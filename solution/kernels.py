"""Our Triton kernels for one Qwen2 + NF4 + LoRA training step on a T4 (fp16, sm_75).

None of them uses tl.dot, so nothing here depends on tensor-core support in Triton for Turing:
the matrix multiplies stay in cuBLAS, the kernels below do everything around them.

  nf4_dequant_kernel   NF4 weights with double-quantized absmax -> fp16, one pass
  rms_fwd_kernel       [residual add] + RMSNorm -> fp16, plus the two LoRA-dropout inputs
  rms_bwd_kernel       RMSNorm backward, merging the LoRA-dropout gradients, + residual grad
  qkv_rope_fwd_kernel  bias'ed qkv + LoRA deltas -> RoPE(q), RoPE(k), v in the attention layout,
                       k and v already repeated for GQA
  qkv_rope_bwd_kernel  the inverse: attention grads -> fused [dq|dk|dv] for one GEMM, GQA sum
  glu_fwd_kernel       silu(g) * u over the fused [g|u] GEMM output
  glu_bwd_kernel       its backward, written in place as fused [dg|du] for one GEMM
  lm_head_ce_kernel    cross-entropy, entropy, argmax and dlogits over a 151,936-word vocabulary,
                       for a chunk of rows, without materializing anything but the fp16 logits;
                       rows without a target (the prompt) get entropy and argmax only

Ideas (no code taken): fused RMSNorm and chunked "fused linear cross-entropy" as in Liger-Kernel
(https://github.com/linkedin/Liger-Kernel) and Unsloth (https://github.com/unslothai/unsloth);
online softmax / logsumexp (Milakov & Gimelshein, 2018, arXiv:1805.02867); NF4 and double
quantization layout from bitsandbytes / QLoRA (Dettmers et al., 2023, arXiv:2305.14314).
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


# --------------------------------------------------------------------------------------------
# NF4 -> fp16. Element i of a [N, K] weight lives in byte i // 2 (high nibble for even i), its
# absmax is code2[absmax_q[i // BS1]] * absmax2[i // (BS1 * BS2)] + offset (double quant).
# --------------------------------------------------------------------------------------------
# The 16 NF4 levels (QLoRA, bitsandbytes' `quant_state.code` for quant_type="nf4"), as fp32.
NF4_LEVELS = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
              -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
              0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
              0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)


@triton.jit
def _nf4_level(n):
    """NF4 code -> level with a select tree on constants: no table loads (on Turing the per-
    element gathers, not the bytes, were what limited the first version of this kernel)."""
    b0 = (n & 1) != 0
    b1 = (n & 2) != 0
    b2 = (n & 4) != 0
    l0 = tl.where(b0, -0.6961928009986877, -1.0)
    l1 = tl.where(b0, -0.39491748809814453, -0.5250730514526367)
    l2 = tl.where(b0, -0.18477343022823334, -0.28444138169288635)
    l3 = tl.where(b0, 0.0, -0.09105003625154495)
    l4 = tl.where(b0, 0.16093020141124725, 0.07958029955625534)
    l5 = tl.where(b0, 0.33791524171829224, 0.24611230194568634)
    l6 = tl.where(b0, 0.5626170039176941, 0.44070982933044434)
    l7 = tl.where(b0, 1.0, 0.7229568362236023)
    m0 = tl.where(b1, l1, l0)
    m1 = tl.where(b1, l3, l2)
    m2 = tl.where(b1, l5, l4)
    m3 = tl.where(b1, l7, l6)
    lo = tl.where(b2, m1, m0)
    hi = tl.where(b2, m3, m2)
    return tl.where((n & 8) != 0, hi, lo)


@triton.jit(do_not_specialize=["n_blocks"])
def nf4_dequant_kernel(packed_ptr, absmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr,
                       out_ptr, n_blocks, NESTED: tl.constexpr, STANDARD_NF4: tl.constexpr,
                       BS1: tl.constexpr, BS2: tl.constexpr, ROWS: tl.constexpr):
    """A tile of ROWS quantization blocks of BS1 values (BS1 / 2 bytes each): one absmax per
    block, both nibbles of every byte from one load, output written as one contiguous tile."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    inside = rows < n_blocks
    HALF: tl.constexpr = BS1 // 2
    cols = tl.arange(0, HALF)
    byte = tl.load(packed_ptr + rows[:, None].to(tl.int64) * HALF + cols[None, :],
                   mask=inside[:, None], other=0).to(tl.int32)
    if NESTED:
        q = tl.load(absmax_ptr + rows, mask=inside, other=0).to(tl.int32)
        # rounded product, then the add: bitsandbytes does them as two operations, and a fused
        # multiply-add would differ in the last bit now and then
        scale = libdevice.mul_rn(tl.load(code2_ptr + q),
                                 tl.load(absmax2_ptr + rows // BS2, mask=inside, other=0.0))
        scale = scale + tl.load(offset_ptr)
    else:
        scale = tl.load(absmax_ptr + rows, mask=inside, other=0.0)
    if STANDARD_NF4:
        first = _nf4_level(byte >> 4)
        second = _nf4_level(byte & 15)
    else:
        first = tl.load(nf4_ptr + (byte >> 4))
        second = tl.load(nf4_ptr + (byte & 15))
    first = first * scale[:, None]
    second = second * scale[:, None]
    vals = tl.reshape(tl.join(first, second), (ROWS, BS1))  # element 2j <- high nibble of byte j
    out = out_ptr + rows[:, None].to(tl.int64) * BS1 + tl.arange(0, BS1)[None, :]
    tl.store(out, vals.to(tl.float16), mask=inside[:, None])


# --------------------------------------------------------------------------------------------
# RMSNorm over rows of H fp32 values. Optionally first adds a fp16 delta to the fp32 residual
# (and writes the sum), optionally also writes dropout(y) for up to two LoRA inputs. The dropout
# mask is a pure function of (seed, row * H + col), so the backward regenerates it instead of
# storing it.
# --------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["seed_q", "seed_v", "has_delta", "write_y", "drop_q",
                                "drop_v"])
def rms_fwd_kernel(x_ptr, delta_ptr, w_ptr, xsum_ptr, y_ptr, yq_ptr, yv_ptr, rstd_ptr,
                   H, eps, p_drop, keep_scale, seed_q, seed_v, has_delta, write_y, drop_q,
                   drop_v, BLOCK: tl.constexpr):
    """The options are runtime flags, not constexprs: one compiled variant serves the input
    norm, the post-attention norm and the final norm (compilation is part of the measured
    time; the branches are uniform across the grid and cost nothing measurable)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    inside = cols < H
    base = row.to(tl.int64) * H
    x = tl.load(x_ptr + base + cols, mask=inside, other=0.0)
    if has_delta != 0:
        x += tl.load(delta_ptr + base + cols, mask=inside, other=0.0).to(tl.float32)
        tl.store(xsum_ptr + base + cols, x, mask=inside)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / H + eps)
    tl.store(rstd_ptr + row, rstd)
    y = tl.load(w_ptr + cols, mask=inside, other=0.0) * (x * rstd)
    if write_y != 0:
        tl.store(y_ptr + base + cols, y.to(tl.float16), mask=inside)
    rand_offs = (row * H + cols).to(tl.int32)
    if drop_q != 0:
        keep = tl.rand(seed_q, rand_offs) >= p_drop
        tl.store(yq_ptr + base + cols, tl.where(keep, y * keep_scale, 0.0).to(tl.float16),
                 mask=inside)
    if drop_v != 0:
        keep = tl.rand(seed_v, rand_offs) >= p_drop
        tl.store(yv_ptr + base + cols, tl.where(keep, y * keep_scale, 0.0).to(tl.float16),
                 mask=inside)


@triton.jit(do_not_specialize=["seed_q", "seed_v", "drop_q", "drop_v", "has_res"])
def rms_bwd_kernel(dy_ptr, tq_ptr, tv_ptr, x_ptr, rstd_ptr, w_ptr, dres_ptr, dx_ptr,
                   H, dy_stride, p_drop, keep_scale, seed_q, seed_v, grad_scale,
                   drop_q, drop_v, has_res, BLOCK: tl.constexpr):
    """dx = dres + RMSNorm'(x)^T (dy + mask_q * tq / (1-p) + mask_v * tv / (1-p)) * grad_scale."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    inside = cols < H
    base = row.to(tl.int64) * H
    dy = tl.load(dy_ptr + row.to(tl.int64) * dy_stride + cols, mask=inside, other=0.0)
    dy = dy.to(tl.float32)
    rand_offs = (row * H + cols).to(tl.int32)
    if drop_q != 0:
        keep = tl.rand(seed_q, rand_offs) >= p_drop
        tq = tl.load(tq_ptr + base + cols, mask=inside, other=0.0).to(tl.float32)
        dy += tl.where(keep, tq * keep_scale, 0.0)
    if drop_v != 0:
        keep = tl.rand(seed_v, rand_offs) >= p_drop
        tv = tl.load(tv_ptr + base + cols, mask=inside, other=0.0).to(tl.float32)
        dy += tl.where(keep, tv * keep_scale, 0.0)
    x = tl.load(x_ptr + base + cols, mask=inside, other=0.0)
    rstd = tl.load(rstd_ptr + row)
    g = dy * tl.load(w_ptr + cols, mask=inside, other=0.0)
    xhat = x * rstd
    dx = rstd * (g - xhat * (tl.sum(g * xhat, axis=0) / H)) * grad_scale
    if has_res != 0:
        dx += tl.load(dres_ptr + base + cols, mask=inside, other=0.0)
    tl.store(dx_ptr + base + cols, dx, mask=inside)


# --------------------------------------------------------------------------------------------
# qkv (one GEMM, bias included) + LoRA deltas -> rotary q and k, plain v. One program per token.
# Qwen2 rotate_half: out[:half] = x1*cos - x2*sin, out[half:] = x2*cos + x1*sin.
# --------------------------------------------------------------------------------------------
@triton.jit
def qkv_rope_fwd_kernel(qkv_ptr, lq_ptr, lk_ptr, lv_ptr, cos_ptr, sin_ptr, pos_ptr,
                        q_ptr, k_ptr, v_ptr, lora_scale,
                        HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                        BHQ: tl.constexpr, BHKV: tl.constexpr, GROUP: tl.constexpr,
                        LORA_Q: tl.constexpr, LORA_K: tl.constexpr, LORA_V: tl.constexpr):
    """k and v are written already repeated for grouped-query attention: [tokens, HQ, D], kv head
    j at heads j * GROUP .. j * GROUP + GROUP - 1 (what repeat_kv does), so the attention kernel
    reads them without a separate expand copy."""
    tok = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = D // 2
    C: tl.constexpr = (HQ + 2 * HKV) * D
    pos = tl.load(pos_ptr + tok)
    d = tl.arange(0, HALF)
    cos = tl.load(cos_ptr + pos * HALF + d)[None, :]
    sin = tl.load(sin_ptr + pos * HALF + d)[None, :]
    row = qkv_ptr + tok * C

    hq = tl.arange(0, BHQ)[:, None]
    mq = hq < HQ
    idx = hq * D + d[None, :]
    q1 = tl.load(row + idx, mask=mq, other=0.0).to(tl.float32)
    q2 = tl.load(row + idx + HALF, mask=mq, other=0.0).to(tl.float32)
    if LORA_Q:
        lrow = lq_ptr + tok * (HQ * D)
        q1 += tl.load(lrow + idx, mask=mq, other=0.0).to(tl.float32) * lora_scale
        q2 += tl.load(lrow + idx + HALF, mask=mq, other=0.0).to(tl.float32) * lora_scale
    out = q_ptr + tok * (HQ * D)
    tl.store(out + idx, (q1 * cos - q2 * sin).to(tl.float16), mask=mq)
    tl.store(out + idx + HALF, (q2 * cos + q1 * sin).to(tl.float16), mask=mq)

    hk = tl.arange(0, BHKV)[:, None]
    mk = hk < HKV
    idx = hk * D + d[None, :]
    k1 = tl.load(row + HQ * D + idx, mask=mk, other=0.0).to(tl.float32)
    k2 = tl.load(row + HQ * D + idx + HALF, mask=mk, other=0.0).to(tl.float32)
    if LORA_K:
        lrow = lk_ptr + tok * (HKV * D)
        k1 += tl.load(lrow + idx, mask=mk, other=0.0).to(tl.float32) * lora_scale
        k2 += tl.load(lrow + idx + HALF, mask=mk, other=0.0).to(tl.float32) * lora_scale
    k1, k2 = (k1 * cos - k2 * sin).to(tl.float16), (k2 * cos + k1 * sin).to(tl.float16)
    for r in tl.static_range(GROUP):
        out = k_ptr + tok * (HQ * D) + (hk * GROUP + r) * D + d[None, :]
        tl.store(out, k1, mask=mk)
        tl.store(out + HALF, k2, mask=mk)

    v1 = tl.load(row + (HQ + HKV) * D + idx, mask=mk, other=0.0).to(tl.float32)
    v2 = tl.load(row + (HQ + HKV) * D + idx + HALF, mask=mk, other=0.0).to(tl.float32)
    if LORA_V:
        lrow = lv_ptr + tok * (HKV * D)
        v1 += tl.load(lrow + idx, mask=mk, other=0.0).to(tl.float32) * lora_scale
        v2 += tl.load(lrow + idx + HALF, mask=mk, other=0.0).to(tl.float32) * lora_scale
    v1, v2 = v1.to(tl.float16), v2.to(tl.float16)
    for r in tl.static_range(GROUP):
        out = v_ptr + tok * (HQ * D) + (hk * GROUP + r) * D + d[None, :]
        tl.store(out, v1, mask=mk)
        tl.store(out + HALF, v2, mask=mk)


@triton.jit
def qkv_rope_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, pos_ptr, dqkv_ptr,
                        HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                        BHQ: tl.constexpr, BHKV: tl.constexpr, GROUP: tl.constexpr):
    """dq, dk, dv come from attention over HQ heads (k and v repeated GROUP times); writes the
    pre-rotary [dq | dk | dv] row that multiplies the fused qkv weight, summing the GQA groups."""
    tok = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = D // 2
    C: tl.constexpr = (HQ + 2 * HKV) * D
    pos = tl.load(pos_ptr + tok)
    d = tl.arange(0, HALF)
    cos = tl.load(cos_ptr + pos * HALF + d)[None, :]
    sin = tl.load(sin_ptr + pos * HALF + d)[None, :]
    out = dqkv_ptr + tok * C

    hq = tl.arange(0, BHQ)[:, None]
    mq = hq < HQ
    idx = hq * D + d[None, :]
    g1 = tl.load(dq_ptr + tok * (HQ * D) + idx, mask=mq, other=0.0).to(tl.float32)
    g2 = tl.load(dq_ptr + tok * (HQ * D) + idx + HALF, mask=mq, other=0.0).to(tl.float32)
    tl.store(out + idx, (g1 * cos + g2 * sin).to(tl.float16), mask=mq)
    tl.store(out + idx + HALF, (g2 * cos - g1 * sin).to(tl.float16), mask=mq)

    hk = tl.arange(0, BHKV)[:, None]
    mk = hk < HKV
    k1 = tl.zeros((BHKV, HALF), dtype=tl.float32)
    k2 = tl.zeros((BHKV, HALF), dtype=tl.float32)
    v1 = tl.zeros((BHKV, HALF), dtype=tl.float32)
    v2 = tl.zeros((BHKV, HALF), dtype=tl.float32)
    for r in tl.static_range(GROUP):
        idx = (hk * GROUP + r) * D + d[None, :]
        k1 += tl.load(dk_ptr + tok * (HQ * D) + idx, mask=mk, other=0.0).to(tl.float32)
        k2 += tl.load(dk_ptr + tok * (HQ * D) + idx + HALF, mask=mk, other=0.0).to(tl.float32)
        v1 += tl.load(dv_ptr + tok * (HQ * D) + idx, mask=mk, other=0.0).to(tl.float32)
        v2 += tl.load(dv_ptr + tok * (HQ * D) + idx + HALF, mask=mk, other=0.0).to(tl.float32)
    idx = hk * D + d[None, :]
    tl.store(out + HQ * D + idx, (k1 * cos + k2 * sin).to(tl.float16), mask=mk)
    tl.store(out + HQ * D + idx + HALF, (k2 * cos - k1 * sin).to(tl.float16), mask=mk)
    tl.store(out + (HQ + HKV) * D + idx, v1.to(tl.float16), mask=mk)
    tl.store(out + (HQ + HKV) * D + idx + HALF, v2.to(tl.float16), mask=mk)


# --------------------------------------------------------------------------------------------
# SwiGLU on the fused [gate | up] output of one GEMM, and its backward in place.
# --------------------------------------------------------------------------------------------
@triton.jit
def glu_fwd_kernel(gu_ptr, out_ptr, I, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = cols < I
    src = gu_ptr + row * (2 * I)
    g = tl.load(src + cols, mask=inside, other=0.0).to(tl.float32)
    u = tl.load(src + I + cols, mask=inside, other=0.0).to(tl.float32)
    act = g / (1.0 + tl.exp(-g))
    tl.store(out_ptr + row * I + cols, (act * u).to(tl.float16), mask=inside)


@triton.jit
def glu_bwd_kernel(dm_ptr, gu_ptr, I, BLOCK: tl.constexpr):
    """Overwrites [g | u] of one row with [dL/dg | dL/du] given dL/d(silu(g) * u)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = cols < I
    src = gu_ptr + row * (2 * I)
    g = tl.load(src + cols, mask=inside, other=0.0).to(tl.float32)
    u = tl.load(src + I + cols, mask=inside, other=0.0).to(tl.float32)
    dm = tl.load(dm_ptr + row * I + cols, mask=inside, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    tl.store(src + cols, (dm * u * sig * (1.0 + g * (1.0 - sig))).to(tl.float16), mask=inside)
    tl.store(src + I + cols, (dm * g * sig).to(tl.float16), mask=inside)


# --------------------------------------------------------------------------------------------
# One row of fp16 logits per program: logsumexp, entropy and argmax in one online pass, then
# the gradient softmax - onehot (unscaled) written over the logits in a second pass. Rows with
# target < 0 get entropy and argmax only (and zeros, if the chunk's gradient GEMM will read them).
# --------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["zero_unsupervised"])
def lm_head_ce_kernel(logits_ptr, target_ptr, loss_ptr, entropy_ptr, correct_ptr, V,
                      zero_unsupervised, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    src = logits_ptr + row * V
    cols = tl.arange(0, BLOCK)
    m = tl.full((), float("-inf"), tl.float32)
    s = tl.zeros((), tl.float32)
    t = tl.zeros((), tl.float32)
    best = tl.full((), float("-inf"), tl.float32)
    best_idx = tl.zeros((), tl.int32)
    for start in range(0, V, BLOCK):
        inside = start + cols < V
        x = tl.load(src + start + cols, mask=inside, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        block_arg = tl.argmax(x, axis=0)
        best_idx = tl.where(block_max > best, start + block_arg, best_idx)  # first max wins
        best = tl.maximum(best, block_max)
        m_new = tl.maximum(m, block_max)
        e = tl.where(inside, tl.exp(x - m_new), 0.0)
        alpha = tl.exp(m - m_new)
        s = s * alpha + tl.sum(e, axis=0)
        t = t * alpha + tl.sum(tl.where(inside, e * x, 0.0), axis=0)
        m = m_new
    lse = m + tl.log(s)
    tl.store(entropy_ptr + row, lse - t / s)
    target = tl.load(target_ptr + row)
    valid = target >= 0
    x_t = tl.load(src + tl.maximum(target, 0)).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(valid, lse - x_t, 0.0))
    tl.store(correct_ptr + row, (valid & (best_idx == target)).to(tl.int32))
    if valid:
        for start in range(0, V, BLOCK):
            inside = start + cols < V
            x = tl.load(src + start + cols, mask=inside, other=0.0).to(tl.float32)
            grad = tl.exp(x - lse) - tl.where(start + cols == target, 1.0, 0.0)
            tl.store(src + start + cols, grad.to(tl.float16), mask=inside)
    elif zero_unsupervised != 0:
        for start in range(0, V, BLOCK):
            inside = start + cols < V
            tl.store(src + start + cols, tl.zeros((BLOCK,), tl.float16), mask=inside)
