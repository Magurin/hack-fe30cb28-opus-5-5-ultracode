"""Our kernels against plain PyTorch, and the whole fused step against the HF + PEFT model.

Run on a GPU: `python -m pytest solution -q`.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEV = "cuda"


def close(a, b, tol):
    torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", [(256, 1536), (1536, 512), (17920, 1536)])
def test_nf4_dequant_matches_bitsandbytes(nested, shape):
    import bitsandbytes.functional as F

    from solution.engine import NF4Weight

    torch.manual_seed(0)
    w = torch.randn(shape, device=DEV, dtype=torch.float16) * 0.05
    packed, state = F.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=nested)
    ref = F.dequantize_4bit(packed, state)

    class P:  # what NF4Weight reads from a Params4bit
        pass

    p = P()
    p.quant_state, p.data = state, packed
    weight = NF4Weight(p)
    out = torch.empty(shape, device=DEV, dtype=torch.float16)
    weight.dequant_into(out)
    assert torch.equal(out, ref)


class _Spec:
    eps = 1e-6


class _Packed:
    def __init__(self, p):
        self.dropout_p = p


@pytest.mark.parametrize("n,H", [(37, 1536), (5, 100)])
def test_rmsnorm_forward_backward(n, H):
    from solution.engine import _rms_bwd, _rms_fwd

    torch.manual_seed(0)
    x = torch.randn(n, H, device=DEV) * 3
    delta = torch.randn(n, H, device=DEV, dtype=torch.float16)
    w = torch.rand(H, device=DEV) + 0.5
    p = 0.25
    packed, spec = _Packed(p), _Spec()
    xsum, y, yq, yv, rstd = _rms_fwd(spec, x, w, packed, delta=delta, seeds=(11, 12),
                                     drop_q=True, drop_v=True)

    xr = (x + delta.float()).requires_grad_()
    yr = w * (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + spec.eps))
    close(xsum, xr, 1e-6)
    close(y, yr, 2e-3)
    mask_q, mask_v = yq != 0, yv != 0
    assert 0.15 < 1 - mask_q.float().mean().item() < 0.35
    assert not torch.equal(mask_q, mask_v)
    close(yq, torch.where(mask_q, yr / (1 - p), 0.0), 2e-3)

    # the same seeds regenerate the same masks
    _, _, yq2, yv2, _ = _rms_fwd(spec, xsum, w, packed, write_y=False, seeds=(11, 12),
                                 drop_q=True, drop_v=True)
    assert torch.equal(yq, yq2) and torch.equal(yv, yv2)

    dy = torch.randn(n, H, device=DEV, dtype=torch.float16)
    tq = torch.randn(n, H, device=DEV, dtype=torch.float16)
    tv = torch.randn(n, H, device=DEV, dtype=torch.float16)
    dres = torch.randn(n, H, device=DEV)
    dx = _rms_bwd(spec, dy, xsum, rstd, w, packed, dres=dres, tq=tq, tv=tv, seeds=(11, 12),
                  grad_scale=0.5)
    g_total = dy.float() + (mask_q * tq.float() + mask_v * tv.float()) / (1 - p)
    (gx,) = torch.autograd.grad(yr, xr, g_total)
    close(dx, dres + 0.5 * gx, 1e-4)


def _rope_ref(x, cos, sin):
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


def test_qkv_rope_forward_backward():
    from solution import kernels as K

    torch.manual_seed(0)
    n, HQ, HKV, D = 23, 12, 2, 128
    C = (HQ + 2 * HKV) * D
    qkv = torch.randn(n, C, device=DEV, dtype=torch.float16)
    lq = torch.randn(n, HQ * D, device=DEV, dtype=torch.float16)
    lv = torch.randn(n, HKV * D, device=DEV, dtype=torch.float16)
    pos = torch.randint(0, 500, (n,), device=DEV, dtype=torch.int32)
    inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device=DEV).float() / D))
    freqs = torch.arange(512, device=DEV).float()[:, None] * inv[None, :]
    cos_t, sin_t = freqs.cos().contiguous(), freqs.sin().contiguous()
    q = torch.empty(n, HQ, D, device=DEV, dtype=torch.float16)
    group = HQ // HKV
    k = torch.empty(n, HQ, D, device=DEV, dtype=torch.float16)  # repeated for GQA
    v = torch.empty(n, HQ, D, device=DEV, dtype=torch.float16)
    K.qkv_rope_fwd_kernel[(n,)](qkv, lq, qkv, lv, cos_t, sin_t, pos, q, k, v, 2.0, HQ=HQ,
                                HKV=HKV, D=D, BHQ=16, BHKV=2, GROUP=group, LORA_Q=True,
                                LORA_K=False, LORA_V=True)
    cos = torch.cat([cos_t, cos_t], -1)[pos.long()][:, None, :]
    sin = torch.cat([sin_t, sin_t], -1)[pos.long()][:, None, :]
    qf = qkv[:, :HQ * D].float() + 2 * lq.float()
    kf = qkv[:, HQ * D:(HQ + HKV) * D].float()
    vf = qkv[:, (HQ + HKV) * D:].float() + 2 * lv.float()
    qf, kf, vf = (t.view(n, -1, D).requires_grad_() for t in (qf, kf, vf))
    qr, kr = _rope_ref(qf, cos, sin), _rope_ref(kf, cos, sin)
    close(q, qr, 2e-3)
    close(k, kr.repeat_interleave(group, dim=1), 2e-3)
    close(v, vf.repeat_interleave(group, dim=1), 2e-3)

    dq = torch.randn(n, HQ, D, device=DEV, dtype=torch.float16)
    dk = torch.randn(n, HQ, D, device=DEV, dtype=torch.float16)
    dv = torch.randn(n, HQ, D, device=DEV, dtype=torch.float16)
    dqkv = torch.empty(n, C, device=DEV, dtype=torch.float16)
    K.qkv_rope_bwd_kernel[(n,)](dq, dk, dv, cos_t, sin_t, pos, dqkv, HQ=HQ, HKV=HKV, D=D,
                                BHQ=16, BHKV=2, GROUP=group)
    dk_sum = dk.float().view(n, HKV, group, D).sum(2)
    dv_sum = dv.float().view(n, HKV, group, D).sum(2)
    gq, gk = torch.autograd.grad((qr, kr), (qf, kf), (dq.float(), dk_sum))
    close(dqkv[:, :HQ * D], gq.reshape(n, -1), 3e-3)
    close(dqkv[:, HQ * D:(HQ + HKV) * D], gk.reshape(n, -1), 1e-2)
    close(dqkv[:, (HQ + HKV) * D:], dv_sum.reshape(n, -1), 1e-2)


@pytest.mark.parametrize("n,I", [(9, 8960), (3, 1000)])
def test_glu_forward_backward(n, I):
    from solution import kernels as K

    torch.manual_seed(0)
    gu = torch.randn(n, 2 * I, device=DEV, dtype=torch.float16) * 3
    m = torch.empty(n, I, device=DEV, dtype=torch.float16)
    K.glu_fwd_kernel[(n, math.ceil(I / 1024))](gu, m, I, BLOCK=1024)
    g = gu[:, :I].float().requires_grad_()
    u = gu[:, I:].float().requires_grad_()
    ref = torch.nn.functional.silu(g) * u
    close(m, ref, 2e-3)
    dm = torch.randn(n, I, device=DEV, dtype=torch.float16)
    dg, du = torch.autograd.grad(ref, (g, u), dm.float())
    K.glu_bwd_kernel[(n, math.ceil(I / 1024))](dm, gu, I, BLOCK=1024)
    close(gu[:, :I], dg, 1e-2)
    close(gu[:, I:], du, 1e-2)


@pytest.mark.parametrize("V", [151936, 5000])
def test_lm_head_ce(V):
    from solution import kernels as K

    torch.manual_seed(0)
    n = 37
    logits = (torch.randn(n, V, device=DEV) * 4).to(torch.float16)
    target = torch.randint(0, V, (n,), device=DEV)
    target[::5] = -100
    target[1] = logits[1].float().argmax()
    ref = logits.float().requires_grad_()
    lse = torch.logsumexp(ref, -1)
    p = torch.softmax(ref, -1)
    entropy = lse - (p * ref).sum(-1)
    ce = torch.nn.functional.cross_entropy(ref, target, ignore_index=-100, reduction="none")
    (grad,) = torch.autograd.grad(ce.sum(), ref)
    work = logits.clone()
    loss = torch.empty(n, device=DEV)
    ent = torch.empty(n, device=DEV)
    correct = torch.empty(n, device=DEV, dtype=torch.int32)
    K.lm_head_ce_kernel[(n,)](work, target, loss, ent, correct, V, 1, BLOCK=4096, num_warps=8)
    close(loss, ce, 1e-4)
    close(ent, entropy, 1e-3)
    valid = target >= 0
    assert torch.equal(correct.bool(), valid & (logits.float().argmax(-1) == target))
    assert correct[1].item() == 1
    close(work, grad, 1e-3)


def test_varlen_attention_matches_sdpa():
    from solution.engine import _attn_bwd, _attn_fwd, _expand_kv

    torch.manual_seed(0)
    lengths = [7, 130, 1, 64]
    HQ, HKV, D = 12, 2, 128

    class P:
        pass

    packed = P()
    packed.cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                                     device=DEV, dtype=torch.int32)
    packed.max_len = max(lengths)
    n = sum(lengths)
    q = torch.randn(n, HQ, D, device=DEV, dtype=torch.float16)
    k = torch.randn(n, HKV, D, device=DEV, dtype=torch.float16)
    v = torch.randn(n, HKV, D, device=DEV, dtype=torch.float16)
    scale = D ** -0.5
    ke, ve = _expand_kv(k, HQ // HKV), _expand_kv(v, HQ // HKV)
    out, lse, s, o = _attn_fwd(q, ke, ve, packed, scale)
    dout = torch.randn_like(q)
    dq, dk, dv = _attn_bwd(dout, q, ke, ve, out, lse, s, o, packed, scale)

    start = 0
    for length in lengths:
        sl = slice(start, start + length)
        qr = q[sl].float().transpose(0, 1).requires_grad_()
        kr = ke[sl].float().transpose(0, 1).requires_grad_()
        vr = ve[sl].float().transpose(0, 1).requires_grad_()
        ref = torch.nn.functional.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
        g = torch.autograd.grad(ref, (qr, kr, vr), dout[sl].float().transpose(0, 1))
        close(out[sl], ref.transpose(0, 1), 2e-3)
        close(dq[sl], g[0].transpose(0, 1), 1e-2)
        close(dk[sl], g[1].transpose(0, 1), 1e-2)
        close(dv[sl], g[2].transpose(0, 1), 1e-2)
        start += length


# ---------------------------------------------------------------------------------------------
# The whole step on a small random Qwen2, 4-bit + LoRA, against the HF/PEFT path under fp16
# autocast (what Soup runs on a T4).
# ---------------------------------------------------------------------------------------------
def _tiny_model(tmp_path, dropout):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen2Config

    config = Qwen2Config(vocab_size=3000, hidden_size=256, intermediate_size=640,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=512, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(tmp_path / "tiny")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "tiny", quantization_config=quant,
                                                 dtype=torch.float16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=dropout,
                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    for name, param in model.named_parameters():  # B != 0 so every gradient path is exercised
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.02)
    return model


def test_fused_step_matches_hf(tmp_path):
    from solution.engine import FusedQwen2

    model = _tiny_model(tmp_path, dropout=0.0)
    model.train()
    torch.manual_seed(1)
    lengths = [17, 60, 33, 5]
    B, T = len(lengths), max(lengths)
    ids = torch.randint(0, 3000, (B, T), device=DEV)
    mask = (torch.arange(T, device=DEV)[None] < torch.tensor(lengths, device=DEV)[:, None])
    mask = mask.long()
    ids = torch.where(mask.bool(), ids, 0)
    labels = torch.where(mask.bool(), ids, -100)
    labels[:, :3] = -100
    n_items = (labels[:, 1:] != -100).sum()

    params = [p for p in model.parameters() if p.requires_grad]
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask, labels=labels,
                    num_items_in_batch=n_items)
    ref_loss = out.loss.float()
    ref_grads = torch.autograd.grad(ref_loss, params)
    logits = out.logits.float()
    ref_entropy = (torch.logsumexp(logits, -1) - (torch.softmax(logits, -1) * logits).sum(-1))
    ref_entropy = (ref_entropy * mask).sum() / mask.sum()

    engine = FusedQwen2(model)
    loss, packed, stats = engine.step(ids, mask, labels, float(n_items))
    grads = torch.autograd.grad(loss, params)
    assert abs(loss.item() - ref_loss.item()) / ref_loss.item() < 2e-3
    entropy = stats.entropy.sum() / packed.n
    assert abs(entropy.item() - ref_entropy.item()) / ref_entropy.item() < 2e-3
    for g, r in zip(grads, ref_grads):
        rel = (g.float() - r.float()).norm() / r.float().norm().clamp(min=1e-12)
        assert rel < 2e-2, rel.item()


def test_weight_streaming_is_exact(tmp_path):
    from solution.engine import FusedQwen2

    model = _tiny_model(tmp_path, dropout=0.1)
    model.train()
    torch.manual_seed(2)
    B, T = 3, 40
    ids = torch.randint(0, 3000, (B, T), device=DEV)
    mask = torch.ones(B, T, device=DEV, dtype=torch.long)
    mask[1, 25:] = 0
    labels = torch.where(mask.bool(), ids, -100)
    params = [p for p in model.parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.is_cuda}

    def run(offload):
        engine = FusedQwen2(model, seed=7)
        if offload:
            engine.offload_weights()
        loss, _, _ = engine.step(ids, mask, labels, 50.0)
        grads = torch.autograd.grad(loss, params)
        if offload:
            engine.restore_weights()
        return loss, grads

    loss_a, grads_a = run(False)
    loss_b, grads_b = run(True)
    assert torch.equal(loss_a, loss_b)
    for a, b in zip(grads_a, grads_b):
        assert torch.equal(a, b)
    for n, p in model.named_parameters():
        if n in before:
            assert p.is_cuda and torch.equal(p.detach(), before[n]), n


def test_after_training_save_and_forward(tmp_path):
    """After finish_training the weights wait in host memory: saving the adapter works as is,
    and a later forward call brings them back to the GPU and gives the same logits."""
    from solution.engine import FusedQwen2

    model = _tiny_model(tmp_path, dropout=0.0)
    ids = torch.randint(0, 3000, (2, 16), device=DEV)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        before = model(input_ids=ids).logits.float()
    model.save_pretrained(tmp_path / "ref")
    engine = FusedQwen2(model)
    engine.offload_weights()
    engine.finish_training()
    model.save_pretrained(tmp_path / "out")
    from safetensors.torch import load_file

    ref = load_file(tmp_path / "ref" / "adapter_model.safetensors")
    out = load_file(tmp_path / "out" / "adapter_model.safetensors")
    assert ref.keys() == out.keys() and all(torch.equal(ref[k], out[k]) for k in ref)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        after = model(input_ids=ids).logits.float()
    assert torch.equal(before, after)
