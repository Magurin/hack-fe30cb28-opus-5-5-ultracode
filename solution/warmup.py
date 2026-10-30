"""Compile our kernels while Soup loads the model, instead of during the first training step.

Triton compiles a kernel the first time it is launched with a new signature, and on a Kaggle CPU
that is about a second per variant; the measured time includes it. Soup spends several seconds
reading and quantizing the checkpoint before training starts, so a background thread launches
every kernel variant the step uses once, on tiny tensors, through the very helpers the step
calls (so the signatures, and therefore Triton's cache keys, are the same). It also touches
cuBLAS and the attention kernel so their modules are loaded. Nothing it computes is used, and
the training step does not depend on it: a variant it missed is compiled on first use as usual.
"""

import threading

import torch

_thread = None


class _Dims:
    pass


def _run(dims, dropout_p):
    from . import engine as E

    dev = torch.device("cuda", torch.cuda.current_device())
    spec = _Dims()
    spec.H, spec.I, spec.HQ, spec.HKV, spec.D = dims["H"], dims["I"], dims["HQ"], dims["HKV"], \
        dims["D"]
    spec.eps = dims["eps"]
    H, I, HQ, HKV, D, V = spec.H, spec.I, spec.HQ, spec.HKV, spec.D, dims["V"]
    packed = _Dims()
    packed.dropout_p = dropout_p
    packed.cos = torch.ones((1, D // 2), device=dev)
    packed.sin = torch.zeros((1, D // 2), device=dev)
    packed.pos = torch.zeros((1,), device=dev, dtype=torch.int32)
    packed.cu_seqlens = torch.tensor([0, 2], device=dev, dtype=torch.int32)
    packed.max_len = 2
    f16 = dict(device=dev, dtype=torch.float16)

    x = torch.randn((1, H), device=dev)
    w = torch.ones((H,), device=dev)
    h16 = torch.randn((1, H), **f16)
    seeds = (3, 5)
    # input norm with both LoRA dropouts, post-attention norm with residual, final norm
    E._rms_fwd(spec, x, w, packed, seeds=seeds, drop_q=True, drop_v=True)
    E._rms_fwd(spec, x, w, packed, delta=h16)
    _, _, _, _, rstd = E._rms_fwd(spec, x, w, packed)
    E._rms_bwd(spec, h16, x, rstd, w, packed, dres=x)
    E._rms_bwd(spec, h16, x, rstd, w, packed, dres=x, tq=h16, tv=h16, seeds=seeds)
    E._rms_bwd(spec, h16, x, rstd, w, packed, grad_scale=0.5, out=torch.empty_like(x))

    qkv = torch.randn((1, (HQ + 2 * HKV) * D), **f16)
    q, k, v = E._rope_fwd(spec, qkv, torch.randn((1, HQ * D), **f16),
                          torch.randn((1, HKV * D), **f16), packed, 2.0)
    grads = torch.randn((1, HQ, D), **f16)
    E._rope_bwd(spec, grads, grads, grads, packed)

    gu = torch.randn((1, 2 * I), **f16)
    m = E._glu_fwd(gu, I)
    E._glu_bwd(m, gu, I)

    logits = torch.randn((1, V), **f16)
    E._ce(logits, torch.zeros((1,), device=dev, dtype=torch.long),
          torch.empty((1,), device=dev), torch.empty((1,), device=dev),
          torch.empty((1,), device=dev, dtype=torch.int32))

    import bitsandbytes.functional as F

    packed_w, state = F.quantize_4bit(torch.randn((64, 256), **f16), blocksize=64,
                                      quant_type="nf4", compress_statistics=True)
    holder = _Dims()
    holder.quant_state, holder.data = state, packed_w
    weight = E.NF4Weight(holder)
    weight.dequant_into(torch.empty((64, 256), **f16))

    # cuBLAS fp16 GEMMs and the varlen attention kernels, forward and backward
    a = torch.randn((2, H), **f16)
    (a @ torch.randn((H, 64), **f16)).sum()
    qa = torch.randn((2, HQ, D), **f16)
    out, lse, s, o = E._attn_fwd(qa, qa, qa, packed, D ** -0.5)
    E._attn_bwd(torch.randn_like(qa), qa, qa, qa, out, lse, s, o, packed, D ** -0.5)
    torch.cuda.synchronize(dev)


def start(base_model):
    """Start compiling for `base_model` (a Hugging Face id or path) in the background."""
    global _thread
    if _thread is not None or not torch.cuda.is_available():
        return
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_model)
    if getattr(config, "model_type", None) != "qwen2":
        return
    dims = {"H": config.hidden_size, "I": config.intermediate_size, "V": config.vocab_size,
            "HQ": config.num_attention_heads, "HKV": config.num_key_value_heads,
            "D": getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads,
            "eps": float(config.rms_norm_eps)}

    def work():
        try:
            _run(dims, 0.05)
        except Exception as exc:  # noqa: BLE001 - a missed warm-up only costs time
            print(f"[solution] kernel warm-up skipped ({exc.__class__.__name__}: {exc})",
                  flush=True)

    _thread = threading.Thread(target=work, name="solution-kernel-warmup", daemon=True)
    _thread.start()


def wait(timeout=300):
    if _thread is not None:
        _thread.join(timeout)
