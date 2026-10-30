"""The training step of Qwen2 + NF4 + LoRA, rebuilt around our kernels.

Instead of running the HF modules (fp32 residual stream and norms, fp32 copies of every
projection output, (B, T, 151936) logits upcast to fp32), one step goes like this:

1. pad tokens are dropped: the rows of the batch are packed into one (N, H) sequence (N = real
   tokens), attention runs varlen over the packed rows, so no work and no memory goes to padding;
2. every decoder layer is one autograd.Function (`_Layer`) that keeps only its input, the fp32
   residual (6 KiB per token), and recomputes the rest of its forward in backward: about 6x less
   activation memory than keeping the attention and MLP intermediates, for one extra pass of
   the qkv, o and gate/up GEMMs. NF4 weights are dequantized by our kernel once in forward and
   once in backward (as bitsandbytes does); the backward reuses them for the recomputation;
3. the final norm, lm_head, cross-entropy and the metrics TRL / Soup log (entropy, token accuracy,
   per-row loss) are one chunked pass (`_Head`): logits exist for a chunk of rows at a time and the
   gradient w.r.t. the hidden state is formed in the same pass;
4. the packed NF4 weights of the layers wait in pinned host memory and are streamed to the GPU
   one layer ahead of the compute (`WeightStream`).

The math is the base's: the same fp16 roundings at the same places (projection outputs, attention
inputs, the MLP product), fp32 for the residual, the norms, RoPE, softmax and the loss. The LoRA
dropout is regenerated from a seed in backward instead of being stored.
"""

import random

import torch
import triton

from . import kernels as K

NF4_ROWS = 32  # quantization blocks (of 64 values) per program
GLU_BLOCK = 1024
CE_BLOCK = 4096
HEAD_CHUNK = 256  # rows of logits alive at a time: 256 x 151936 fp16 = 78 MiB


def _cdiv(a, b):
    return (a + b - 1) // b


def _pow2(n):
    return triton.next_power_of_2(n)


# ---------------------------------------------------------------------------------------------
# NF4 weights
# ---------------------------------------------------------------------------------------------
class NF4Weight:
    """The packed bytes and quant state of one bitsandbytes Linear4bit weight."""

    def __init__(self, param):
        qs = param.quant_state
        self.param = param
        self.packed = param.data  # the device copy the kernel reads (see WeightStream)
        self.shape = tuple(int(s) for s in qs.shape)
        self.numel = self.shape[0] * self.shape[1]
        self.code = qs.code.float().contiguous()
        self.bs1 = int(qs.blocksize)
        if self.numel % self.bs1:
            raise ValueError("weight size is not a whole number of quantization blocks")
        levels = torch.tensor(K.NF4_LEVELS, dtype=torch.float32, device=self.code.device)
        self.standard = bool(torch.equal(self.code, levels))
        self.nested = bool(qs.nested)
        if self.nested:
            self.absmax = qs.absmax
            self.code2 = qs.state2.code.float().contiguous()
            self.absmax2 = qs.state2.absmax.float().contiguous()
            self.offset = torch.as_tensor(qs.offset, device=self.code.device,
                                          dtype=torch.float32).reshape(1)
            self.bs2 = int(qs.state2.blocksize)
        else:
            self.absmax = qs.absmax.float().contiguous()
            self.code2 = self.absmax2 = self.offset = self.absmax
            self.bs2 = 1

    def dequant_into(self, out):
        blocks = self.numel // self.bs1
        K.nf4_dequant_kernel[(_cdiv(blocks, NF4_ROWS),)](
            self.packed, self.absmax, self.code2, self.absmax2, self.offset, self.code, out,
            blocks, NESTED=self.nested, STANDARD_NF4=self.standard, BS1=self.bs1, BS2=self.bs2,
            ROWS=NF4_ROWS, num_warps=4)
        return out


class WeightStream:
    """The packed NF4 weights of every decoder layer live in pinned host memory during training;
    a layer's bytes are copied into one of three device slots on a side stream, one layer ahead
    of the compute (forward: i+1, backward: i-1). Slot i % 3 is only overwritten after the
    compute stream has finished with the layer that held it before. 24 MiB per Qwen2.5-1.5B
    layer, so 72 MiB on the GPU instead of ~650 MiB, for ~1.3 GB of PCIe traffic per step that
    runs alongside the GEMMs."""

    SLOTS = 3
    NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __init__(self, specs, device):
        self.specs, self.device = specs, device
        self.layout = []  # per layer: (start, size, [[weight, packed off, n, absmax off, n]])
        offset = 0
        for spec in specs:
            entries, start = [], offset
            for name in self.NAMES:
                w = spec.w[name]
                pk = w.packed.reshape(-1).view(torch.uint8)
                am = w.absmax.reshape(-1).view(torch.uint8)
                entry = [w, offset, pk.numel(), 0, am.numel()]
                offset = _align(offset + pk.numel())
                entry[3] = offset
                offset = _align(offset + am.numel())
                entries.append(entry)
            self.layout.append((start, offset - start, entries))
        self.host = torch.empty((offset,), dtype=torch.uint8, pin_memory=True)
        for _, _, entries in self.layout:
            for w, po, pn, ao, an in entries:
                self.host[po:po + pn].copy_(w.packed.reshape(-1).view(torch.uint8))
                self.host[ao:ao + an].copy_(w.absmax.reshape(-1).view(torch.uint8))
        torch.cuda.synchronize(device)
        # Free the device copies (the bitsandbytes parameters point at the host bytes meanwhile),
        # and only then take the three slots, so the peak never holds both.
        self.moved = []
        for _, _, entries in self.layout:
            for w, po, pn, ao, an in entries:
                data, state = w.param.data, w.param.quant_state
                self.moved.append(w)
                w.param.data = self.host[po:po + pn].view(data.dtype).view(data.shape)
                state.absmax = self.host[ao:ao + an].view(state.absmax.dtype).view(
                    state.absmax.shape)
                w.packed = w.absmax = None
        torch.cuda.empty_cache()
        self.slot_bytes = max(size for _, size, _ in self.layout)
        self.slots = [torch.empty((self.slot_bytes,), dtype=torch.uint8, device=device)
                      for _ in range(self.SLOTS)]
        self.slot_layer = [None] * self.SLOTS
        self.copied = [torch.cuda.Event() for _ in range(self.SLOTS)]
        self.released = [torch.cuda.Event() for _ in range(self.SLOTS)]
        self.stream = torch.cuda.Stream(device)

    def _load(self, i):
        slot = i % self.SLOTS
        if self.slot_layer[slot] == i:
            return
        start, size, _ = self.layout[i]
        self.stream.wait_event(self.released[slot])
        with torch.cuda.stream(self.stream):
            self.slots[slot][:size].copy_(self.host[start:start + size], non_blocking=True)
            self.copied[slot].record(self.stream)
        self.slot_layer[slot] = i

    def acquire(self, i, ahead):
        """Make layer i's weights usable on the current stream; start fetching layer `ahead`."""
        self._load(i)
        slot = i % self.SLOTS
        torch.cuda.current_stream(self.device).wait_event(self.copied[slot])
        start, _, entries = self.layout[i]
        buf = self.slots[slot]
        for w, po, pn, ao, an in entries:
            w.packed = buf[po - start:po - start + pn]
            absmax = buf[ao - start:ao - start + an]
            w.absmax = absmax if w.nested else absmax.view(torch.float32)
        if 0 <= ahead < len(self.layout):
            self._load(ahead)

    def release(self, i):
        self.released[i % self.SLOTS].record(torch.cuda.current_stream(self.device))

    def release_slots(self):
        torch.cuda.synchronize(self.device)
        self.slots = []
        self.slot_layer = [None] * self.SLOTS
        torch.cuda.empty_cache()

    def restore(self):
        """Put the weights back on the device."""
        self.release_slots()
        for w in self.moved:
            state = w.param.quant_state
            w.param.data = w.param.data.to(self.device)
            state.absmax = state.absmax.to(self.device)
            w.packed = w.param.data
            w.absmax = state.absmax if w.nested else state.absmax.float()
        self.moved = []
        torch.cuda.synchronize(self.device)


def _align(n, a=256):
    return (n + a - 1) // a * a


def dequant(weights, device):
    """Dequantize one or more [n_i, K] weights stacked into one fp16 [sum n_i, K] buffer."""
    rows = sum(w.shape[0] for w in weights)
    out = torch.empty((rows, weights[0].shape[1]), device=device, dtype=torch.float16)
    start = 0
    for w in weights:
        w.dequant_into(out[start:start + w.shape[0]])
        start += w.shape[0]
    return out


# ---------------------------------------------------------------------------------------------
# Per-step packing
# ---------------------------------------------------------------------------------------------
class Packed:
    """One micro-batch without its padding: real tokens of every row, back to back."""

    def __init__(self, input_ids, attention_mask, labels, rope, dropout_rng, dropout_p):
        B, T = input_ids.shape
        lengths = attention_mask.sum(dim=1)
        lengths_cpu = lengths.tolist()
        expected = torch.arange(T, device=input_ids.device)[None, :] < lengths[:, None]
        if not bool(torch.equal(attention_mask.bool(), expected)):
            raise ValueError("attention_mask is not right-padded")
        flat = expected.reshape(-1).nonzero().squeeze(1)
        self.n = int(flat.numel())
        self.rows = B
        self.lengths = [int(x) for x in lengths_cpu]
        self.max_len = max(self.lengths) if self.lengths else 0
        self.ids = input_ids.reshape(-1)[flat]
        self.pos = (flat % T).to(torch.int32)
        shifted = torch.full_like(labels, -100)
        shifted[:, :-1] = labels[:, 1:]
        self.targets = shifted.reshape(-1)[flat].contiguous()
        supervised = self.targets >= 0
        self.head_order = torch.argsort((~supervised).to(torch.int8), stable=True)
        self.n_supervised = int(supervised.sum())
        cu = [0]
        for length in self.lengths:
            cu.append(cu[-1] + length)
        self.cu_seqlens = torch.tensor(cu, device=input_ids.device, dtype=torch.int32)
        self.row_of_token = torch.repeat_interleave(
            torch.arange(B, device=input_ids.device), lengths)
        self.cos, self.sin = rope.table(self.max_len)
        self.dropout_p = dropout_p
        self.rng = dropout_rng
        self.lora16_views = None  # per layer: fp16 copies of its LoRA weights (FusedQwen2.step)

    def lora16(self, index, weights):
        if self.lora16_views is not None:
            return self.lora16_views[index]
        return tuple(None if t is None else t.to(torch.float16) for t in weights)

    def seed(self):
        return self.rng.randrange(1, 2**31 - 1)


class RopeTable:
    """cos / sin of the model's own rotary embedding (first half of head_dim: Qwen2 repeats it)."""

    def __init__(self, rotary, head_dim, device):
        self.rotary, self.half, self.device = rotary, head_dim // 2, device
        self.cos = self.sin = None

    def table(self, length):
        if self.cos is None or self.cos.shape[0] < length:
            size = max(1024, _pow2(length))
            dummy = torch.zeros(1, device=self.device, dtype=torch.float32)
            pos = torch.arange(size, device=self.device)[None, :]
            with torch.no_grad():
                cos, sin = self.rotary(dummy, pos)
            cos, sin = cos[0].float(), sin[0].float()
            if not (torch.equal(cos[:, :self.half], cos[:, self.half:])
                    and torch.equal(sin[:, :self.half], sin[:, self.half:])):
                raise ValueError("rotary table is not rotate_half style")
            self.cos = cos[:, :self.half].contiguous()
            self.sin = sin[:, :self.half].contiguous()
        return self.cos, self.sin


# ---------------------------------------------------------------------------------------------
# Attention: PyTorch's own memory-efficient kernel in its varlen form (cu_seqlens), causal.
# ---------------------------------------------------------------------------------------------
def _attn_fwd(q, k, v, packed, scale):
    out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
        q[None], k[None], v[None], None, packed.cu_seqlens, packed.cu_seqlens, packed.max_len,
        packed.max_len, 0.0, 1, True, scale=scale)
    return out[0], lse, seed, offset


def _attn_bwd(dout, q, k, v, out, lse, seed, offset, packed, scale):
    dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
        dout[None], q[None], k[None], v[None], None, out[None], packed.cu_seqlens,
        packed.cu_seqlens, packed.max_len, packed.max_len, lse, 0.0, seed, offset, 1, False,
        scale=scale)
    return dq[0], dk[0], dv[0]


def _split_k_mm(a, b, parts):
    """a @ b as `parts` accumulated GEMMs over slices of K. For the [tokens, 17920] x
    [17920, 1536] backward GEMM through the fused gate/up weight, cuBLAS on a T4 picks a
    slower kernel for the single long-K call (1.93 ms) than for four K=4480 ones (1.70 ms)."""
    k = a.shape[1]
    step = k // parts
    if k % parts or step % 64:
        return a @ b
    out = a[:, :step] @ b[:step]
    for i in range(1, parts):
        out.addmm_(a[:, i * step:(i + 1) * step], b[i * step:(i + 1) * step])
    return out


def _expand_kv(x, group):
    if group == 1:
        return x
    n, hkv, d = x.shape
    return x[:, :, None, :].expand(n, hkv, group, d).reshape(n, hkv * group, d)


# ---------------------------------------------------------------------------------------------
# One decoder layer
# ---------------------------------------------------------------------------------------------
class LayerSpec:
    def __init__(self, layer, config, lora_names, index):
        attn, mlp = layer.self_attn, layer.mlp
        self.index = index
        self.stream = None  # a WeightStream, once training starts
        self.H = config.hidden_size
        self.D = getattr(config, "head_dim", None) or self.H // config.num_attention_heads
        self.HQ = config.num_attention_heads
        self.HKV = config.num_key_value_heads
        self.I = config.intermediate_size
        self.eps = float(layer.input_layernorm.variance_epsilon)
        assert float(layer.post_attention_layernorm.variance_epsilon) == self.eps
        self.ln1 = layer.input_layernorm.weight
        self.ln2 = layer.post_attention_layernorm.weight
        self.scale = attn.scaling

        def base_of(module):
            return getattr(module, "base_layer", module)

        projs = {name: getattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
        projs.update({name: getattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")})
        self.base_modules = {name: base_of(m) for name, m in projs.items()}
        self.w = {name: NF4Weight(base_of(m).weight) for name, m in projs.items()}
        biases = [base_of(projs[name]).bias for name in ("q_proj", "k_proj", "v_proj")]
        self.bias_qkv = torch.cat([b.detach().to(torch.float16) for b in biases]).contiguous()
        self.lora = {}
        for short in ("q", "k", "v"):
            module = projs[f"{short}_proj"]
            if short in lora_names:
                name = lora_names[short]
                self.lora[short] = (module.lora_A[name].weight, module.lora_B[name].weight,
                                    float(module.scaling[name]))

    def lora_params(self):
        out = []
        for short in ("q", "k", "v"):
            if short in self.lora:
                out += [self.lora[short][0], self.lora[short][1]]
            else:
                out += [None, None]
        return out


def _lora_scale(spec):
    scales = {value[2] for value in spec.lora.values()}
    assert len(scales) <= 1
    return scales.pop() if scales else 1.0


def _rms_fwd(spec, x, weight, packed, *, delta=None, write_y=True, seeds=(0, 0),
             drop_q=False, drop_v=False):
    n, H = x.shape
    dev = x.device
    xsum = torch.empty_like(x) if delta is not None else x
    y = torch.empty((n, H), device=dev, dtype=torch.float16) if write_y else None
    yq = torch.empty((n, H), device=dev, dtype=torch.float16) if drop_q else None
    yv = torch.empty((n, H), device=dev, dtype=torch.float16) if drop_v else None
    rstd = torch.empty((n,), device=dev, dtype=torch.float32)
    p = packed.dropout_p
    if n:
        K.rms_fwd_kernel[(n,)](
            x, delta if delta is not None else x, weight, xsum, y if y is not None else x,
            yq if yq is not None else x, yv if yv is not None else x, rstd, H, spec.eps, p,
            1.0 / (1.0 - p), seeds[0], seeds[1], HAS_DELTA=delta is not None, WRITE_Y=write_y,
            DROP_Q=drop_q, DROP_V=drop_v, BLOCK=_pow2(H), num_warps=8)
    return xsum, y, yq, yv, rstd


def _rms_bwd(spec, dy, x, rstd, weight, packed, *, dres=None, tq=None, tv=None, seeds=(0, 0),
             grad_scale=1.0, out=None):
    n, H = x.shape
    dx = out if out is not None else torch.empty_like(x)
    p = packed.dropout_p
    if n:
        K.rms_bwd_kernel[(n,)](
            dy, tq if tq is not None else dy, tv if tv is not None else dy, x, rstd, weight,
            dres if dres is not None else x, dx, H, dy.stride(0), p, 1.0 / (1.0 - p),
            seeds[0], seeds[1], grad_scale, DROP_Q=tq is not None, DROP_V=tv is not None,
            HAS_RES=dres is not None, BLOCK=_pow2(H), num_warps=8)
    return dx


def _rope_fwd(spec, qkv, lq, lv, packed, lora_scale):
    n = qkv.shape[0]
    dev = qkv.device
    q = torch.empty((n, spec.HQ, spec.D), device=dev, dtype=torch.float16)
    # k and v come out repeated for the GQA groups ([n, HQ, D]), ready for the attention kernel
    k = torch.empty((n, spec.HQ, spec.D), device=dev, dtype=torch.float16)
    v = torch.empty((n, spec.HQ, spec.D), device=dev, dtype=torch.float16)
    if n:
        K.qkv_rope_fwd_kernel[(n,)](
            qkv, lq if lq is not None else qkv, qkv, lv if lv is not None else qkv, packed.cos,
            packed.sin, packed.pos, q, k, v, lora_scale, HQ=spec.HQ, HKV=spec.HKV, D=spec.D,
            BHQ=_pow2(spec.HQ), BHKV=_pow2(spec.HKV), GROUP=spec.HQ // spec.HKV,
            LORA_Q=lq is not None, LORA_K=False,
            LORA_V=lv is not None, num_warps=4)
    return q, k, v


def _rope_bwd(spec, dq, dk, dv, packed):
    n = dq.shape[0]
    C = (spec.HQ + 2 * spec.HKV) * spec.D
    dqkv = torch.empty((n, C), device=dq.device, dtype=torch.float16)
    if n:
        K.qkv_rope_bwd_kernel[(n,)](
            dq.contiguous(), dk.contiguous(), dv.contiguous(), packed.cos, packed.sin,
            packed.pos, dqkv, HQ=spec.HQ, HKV=spec.HKV, D=spec.D, BHQ=_pow2(spec.HQ),
            BHKV=_pow2(spec.HKV), GROUP=spec.HQ // spec.HKV, num_warps=4)
    return dqkv


def _glu_fwd(gu, I):
    n = gu.shape[0]
    m = torch.empty((n, I), device=gu.device, dtype=torch.float16)
    if n:
        K.glu_fwd_kernel[(n, _cdiv(I, GLU_BLOCK))](gu, m, I, BLOCK=GLU_BLOCK)
    return m


def _glu_bwd(dm, gu, I):
    n = gu.shape[0]
    if n:
        K.glu_bwd_kernel[(n, _cdiv(I, GLU_BLOCK))](dm, gu, I, BLOCK=GLU_BLOCK)


def _ce(logits, targets, tok_loss, entropy, correct, zero_unsupervised=True):
    rows, V = logits.shape
    if rows:
        K.lm_head_ce_kernel[(rows,)](logits, targets, tok_loss, entropy, correct, V,
                                     int(zero_unsupervised), BLOCK=CE_BLOCK, num_warps=8)


def _layer_forward(spec, h, packed, lora16, seeds, keep):
    """One decoder layer forward. With keep=True also returns what the backward reads."""
    n, H = h.shape
    D, HQ, HKV = spec.D, spec.HQ, spec.HKV
    group = HQ // HKV
    dev = h.device
    aq_w, bq_w, av_w, bv_w = lora16
    lora_scale = _lora_scale(spec)
    saved = {}

    # input RMSNorm (+ the dropout'ed copies the LoRA A matrices read)
    _, a, aq, av, rstd1 = _rms_fwd(spec, h, spec.ln1, packed, seeds=seeds,
                                   drop_q=aq_w is not None, drop_v=av_w is not None)

    # q, k, v in one GEMM; LoRA as two thin GEMMs per target
    w = dequant([spec.w["q_proj"], spec.w["k_proj"], spec.w["v_proj"]], dev)
    qkv = torch.addmm(spec.bias_qkv, a, w.t())
    del a
    if keep:
        saved["w_qkv"] = w
    del w
    xq = xv = lq = lv = None
    if aq_w is not None:
        xq = aq @ aq_w.t()
        lq = xq @ bq_w.t()
    if av_w is not None:
        xv = av @ av_w.t()
        lv = xv @ bv_w.t()
    q, k, v = _rope_fwd(spec, qkv, lq, lv, packed, lora_scale)
    del qkv, lq, lv
    k_e, v_e = k, v
    out, lse, aseed, aoffset = _attn_fwd(q, k_e, v_e, packed, spec.scale)

    w = dequant([spec.w["o_proj"]], dev)
    o = out.reshape(n, HQ * D) @ w.t()
    if keep:
        saved["w_o"] = w
    del w
    h2, b, _, _, rstd2 = _rms_fwd(spec, h, spec.ln2, packed, delta=o)
    del o
    w = dequant([spec.w["gate_proj"], spec.w["up_proj"]], dev)
    gu = b @ w.t()
    del b
    if keep:
        saved.update(w_gu=w, rstd1=rstd1, aq=aq, av=av, xq=xq, xv=xv, q=q, k_e=k_e, v_e=v_e,
                     out=out, lse=lse, aseed=aseed, aoffset=aoffset, h2=h2, rstd2=rstd2, gu=gu)
        return None, saved
    del w, aq, av, q, k, v, k_e, v_e, out, lse
    m = _glu_fwd(gu, spec.I)
    del gu
    w = dequant([spec.w["down_proj"]], dev)
    d = m @ w.t()
    del w, m
    return torch.add(h2, d), None


class _Layer(torch.autograd.Function):
    """One decoder layer. Only its input (the fp32 residual) is kept for backward; the backward
    runs the forward again up to the MLP input, with the same dropout seeds, and then goes back
    through it. The LoRA weights are inputs, so autograd hands their gradients to the optimizer."""

    @staticmethod
    def forward(ctx, h, spec, packed, aq_w, bq_w, ak_w, bk_w, av_w, bv_w):
        if ak_w is not None:
            raise NotImplementedError("LoRA on k_proj")
        seeds = (packed.seed() if aq_w is not None else 0,
                 packed.seed() if av_w is not None else 0)
        lora16 = packed.lora16(spec.index, (aq_w, bq_w, av_w, bv_w))
        if spec.stream is not None:
            spec.stream.acquire(spec.index, spec.index + 1)
        h3, _ = _layer_forward(spec, h, packed, lora16, seeds, keep=False)
        if spec.stream is not None:
            spec.stream.release(spec.index)
        ctx.save_for_backward(h, aq_w, bq_w, av_w, bv_w)
        ctx.spec, ctx.packed, ctx.seeds = spec, packed, seeds
        return h3

    @staticmethod
    def backward(ctx, dh3):
        h, aq_w, bq_w, av_w, bv_w = ctx.saved_tensors
        spec, packed, seeds = ctx.spec, ctx.packed, ctx.seeds
        s = _lora_scale(spec)
        n, H = h.shape
        D, HQ, HKV = spec.D, spec.HQ, spec.HKV
        C = (HQ + 2 * HKV) * D
        dev = h.device
        dh3 = dh3.contiguous()
        lora16 = packed.lora16(spec.index, (aq_w, bq_w, av_w, bv_w))
        if spec.stream is not None:
            spec.stream.acquire(spec.index, spec.index - 1)
        with torch.no_grad():
            _, f = _layer_forward(spec, h, packed, lora16, seeds, keep=True)

        # MLP
        w = dequant([spec.w["down_proj"]], dev)
        dm = dh3.to(torch.float16) @ w
        del w
        gu = f.pop("gu")
        _glu_bwd(dm, gu, spec.I)
        del dm
        db = _split_k_mm(gu, f.pop("w_gu"), 4)
        del gu
        dh2 = _rms_bwd(spec, db, f.pop("h2"), f.pop("rstd2"), spec.ln2, packed, dres=dh3)
        del db, dh3

        # attention
        dout = (dh2.to(torch.float16) @ f.pop("w_o")).view(n, HQ, D)
        dq, dk, dv = _attn_bwd(dout, f.pop("q"), f.pop("k_e"), f.pop("v_e"), f.pop("out"),
                               f.pop("lse"), f.pop("aseed"), f.pop("aoffset"), packed, spec.scale)
        del dout
        dqkv = _rope_bwd(spec, dq, dk, dv, packed)
        del dq, dk, dv
        da = dqkv @ f.pop("w_qkv")

        # LoRA: dB = s g^T x, dx = s g B, dA = dx^T drop(a), d drop(a) = dx A
        grads = {"q": (None, None, None), "v": (None, None, None)}
        for short, a_w, b_w, cols in (("q", lora16[0], lora16[1], slice(0, HQ * D)),
                                      ("v", lora16[2], lora16[3], slice((HQ + HKV) * D, C))):
            if a_w is None:
                continue
            x_lo, a_in = f.pop("x" + short), f.pop("a" + short)
            g = dqkv[:, cols]
            db_w = (g.t() @ x_lo).float() * s
            dx_lo = (g @ b_w) * s
            da_w = (dx_lo.t() @ a_in).float()
            grads[short] = (da_w, db_w, dx_lo @ a_w)
            del x_lo, a_in
        del dqkv
        if spec.stream is not None:
            spec.stream.release(spec.index)
        dh = None
        if ctx.needs_input_grad[0]:
            dh = _rms_bwd(spec, da, h, f["rstd1"], spec.ln1, packed, dres=dh2,
                          tq=grads["q"][2], tv=grads["v"][2], seeds=seeds)
        return (dh, None, None, grads["q"][0], grads["q"][1], None, None, grads["v"][0],
                grads["v"][1])


# ---------------------------------------------------------------------------------------------
# Final norm + lm_head + cross-entropy + metrics, chunked
# ---------------------------------------------------------------------------------------------
class HeadStats:
    pass


class _Head(torch.autograd.Function):
    """Final RMSNorm + lm_head + cross-entropy + the logged metrics, in chunks of rows. The rows
    are taken supervised-first: every row needs its logits (entropy and accuracy are logged over
    all of them), but only rows with a target have a gradient, so chunks of prompt rows skip the
    backward GEMM through the 151,936 x 1536 lm_head weight."""

    @staticmethod
    def forward(ctx, h, norm_w, eps, lm_w, packed, denom, stats):
        n, H = h.shape
        dev = h.device
        spec = HeadStats()
        spec.eps = eps
        order = packed.head_order
        n_sup = packed.n_supervised
        hp = h.index_select(0, order)
        targets = packed.targets.index_select(0, order)
        _, hn, _, _, rstd = _rms_fwd(spec, hp, norm_w, packed)
        dhp = torch.empty_like(hp)
        tok_loss = torch.empty((n,), device=dev, dtype=torch.float32)
        entropy = torch.empty((n,), device=dev, dtype=torch.float32)
        correct = torch.empty((n,), device=dev, dtype=torch.int32)
        inv = 1.0 / float(denom) if float(denom) else float("nan")  # 0 / 0, as in the base
        for c0 in range(0, n, HEAD_CHUNK):
            c1 = min(n, c0 + HEAD_CHUNK)
            logits = hn[c0:c1] @ lm_w.t()
            needs_grad = c0 < n_sup
            _ce(logits, targets[c0:c1], tok_loss[c0:c1], entropy[c0:c1], correct[c0:c1],
                zero_unsupervised=needs_grad)
            if needs_grad:
                dhn = logits @ lm_w
                del logits
                _rms_bwd(spec, dhn, hp[c0:c1], rstd[c0:c1], norm_w, packed, grad_scale=inv,
                         out=dhp[c0:c1])
                del dhn
            else:
                del logits
                dhp[c0:c1].zero_()
        dh = torch.empty_like(h)
        dh.index_copy_(0, order, dhp)
        del dhp, hp
        ctx.save_for_backward(dh)
        stats.order, stats.tok_loss, stats.entropy, stats.correct = order, tok_loss, entropy,             correct
        return tok_loss.sum() * inv

    @staticmethod
    def backward(ctx, grad):
        (dh,) = ctx.saved_tensors
        return dh * grad, None, None, None, None, None, None


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------
class FusedQwen2:
    """Holds what the step needs from a PEFT-wrapped, 4-bit Qwen2ForCausalLM."""

    def __init__(self, peft_model, seed=0):
        causal = peft_model.get_base_model()
        inner = causal.model
        config = causal.config
        self.peft_model = peft_model
        self.config = config
        self.device = inner.embed_tokens.weight.device
        lora_names = _lora_targets(inner.layers[0])
        self.layers = [LayerSpec(layer, config, lora_names, i)
                       for i, layer in enumerate(inner.layers)]
        self.base_modules = [spec.base_modules[name] for spec in self.layers
                             for name in spec.base_modules]
        self.stream = None
        self.dropout_p = _lora_dropout(inner.layers[0], lora_names)
        self.norm_w = inner.norm.weight
        self.eps = float(inner.norm.variance_epsilon)
        self.embed = inner.embed_tokens.weight
        self.lm_w = causal.lm_head.weight
        self.rope = RopeTable(inner.rotary_emb, self.layers[0].D, self.device)
        self.rng = random.Random(seed)
        # One fp16 buffer for every LoRA weight: refreshed by a single foreach copy per step
        # instead of four casts per layer in forward and again in backward.
        self.lora_fp32, shapes = [], []
        for spec in self.layers:
            for t in spec.lora_params():
                if t is not None:
                    self.lora_fp32.append(t)
                    shapes.append(t.shape)
        total = sum(int(torch.Size(sh).numel()) for sh in shapes)
        self.lora_buf = torch.empty((total,), device=self.device, dtype=torch.float16)
        self.lora_views, start = [], 0
        for sh in shapes:
            n = int(torch.Size(sh).numel())
            self.lora_views.append(self.lora_buf[start:start + n].view(sh))
            start += n
        self.lora16_by_layer, i = [], 0
        for spec in self.layers:
            row = []
            for t in (spec.lora_params()[0], spec.lora_params()[1], spec.lora_params()[4],
                      spec.lora_params()[5]):
                if t is None:
                    row.append(None)
                else:
                    row.append(self.lora_views[i])
                    i += 1
            self.lora16_by_layer.append(tuple(row))

    def offload_weights(self):
        """Keep the layers' NF4 weights in pinned host memory and stream them (WeightStream)."""
        if self.stream is None:
            self.stream = WeightStream(self.layers, self.device)
            for spec in self.layers:
                spec.stream = self.stream

    def finish_training(self):
        """Training is over: free the device slots. The weights stay in host memory, which is
        all saving a LoRA adapter needs; the first forward call of any of the 4-bit layers
        (if anything runs the model afterwards) brings them back to the GPU first."""
        if self.stream is None:
            return
        for spec in self.layers:
            spec.stream = None
        self.stream.release_slots()
        hooks = []

        def bring_back(module, args):
            for hook in hooks:
                hook.remove()
            self.restore_weights()

        for module in self.base_modules:
            hooks.append(module.register_forward_pre_hook(bring_back))

    def restore_weights(self):
        if self.stream is not None:
            for spec in self.layers:
                spec.stream = None
            self.stream.restore()
            self.stream = None

    def step(self, input_ids, attention_mask, labels, denom, training=True):
        p = self.dropout_p if training else 0.0
        packed = Packed(input_ids, attention_mask, labels, self.rope, self.rng, p)
        if self.lora_views:
            with torch.no_grad():
                torch._foreach_copy_(self.lora_views, self.lora_fp32)
            packed.lora16_views = self.lora16_by_layer
        h = torch.nn.functional.embedding(packed.ids, self.embed).float()
        for spec in self.layers:
            h = _Layer.apply(h, spec, packed, *spec.lora_params())
        stats = HeadStats()
        lm_w = self.lm_w if self.lm_w.dtype == torch.float16 else self.lm_w.to(torch.float16)
        loss = _Head.apply(h, self.norm_w, self.eps, lm_w, packed, denom, stats)
        return loss, packed, stats


def _lora_targets(layer):
    found = {}
    for short, module in (("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj),
                          ("v", layer.self_attn.v_proj)):
        names = getattr(module, "active_adapters", None)
        if names:
            found[short] = names[0]
    return found


def _lora_dropout(layer, lora_names):
    ps = set()
    for short, name in lora_names.items():
        module = getattr(layer.self_attn, f"{short}_proj")
        drop = module.lora_dropout[name]
        ps.add(float(getattr(drop, "p", 0.0)))
    assert len(ps) <= 1
    return ps.pop() if ps else 0.0
