"""How the fused step plugs into Soup: one patch of `SFTTrainerWrapper.train`.

When Soup is about to train, the trainer and the 4-bit PEFT model exist. If the model is the
one we know (Qwen2 + bitsandbytes NF4 + LoRA on q/k/v projections), we
- store the frozen tied embedding / lm_head in fp16 instead of fp32 (lossless: the fp32 values
  were upcast from the fp16 checkpoint load, and the base casts the lm_head to fp16 anyway);
- replace the trainer's `compute_loss` with `fused_compute_loss`, which runs `FusedQwen2.step`
  and then does exactly the bookkeeping HF Trainer, TRL's SFTTrainer and Soup's rewind recorder
  do in theirs (same loss normalization, same logged metrics, same rewind rows).
Anything we do not recognize falls through to the original code untouched.
"""

import functools
import sys
import types

import torch

_INSTALLED_ATTR = "_soup_kernels_engine"


def _log(message):
    print(f"[solution] {message}", file=sys.stderr, flush=True)


def _is_supported(model):
    try:
        import bitsandbytes as bnb
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "get_base_model"):
        return False
    causal = model.get_base_model()
    config = getattr(causal, "config", None)
    if getattr(config, "model_type", None) != "qwen2" or getattr(config, "hidden_act", "") != "silu":
        return False
    if getattr(config, "use_sliding_window", False):
        return False
    rope = getattr(config, "rope_parameters", None) or {}
    if rope.get("rope_type", "default") != "default":
        return False
    inner = getattr(causal, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        return False
    if not torch.cuda.is_available() or inner.embed_tokens.weight.device.type != "cuda":
        return False
    for layer in inner.layers:
        attn, mlp = layer.self_attn, layer.mlp
        for name, module in (("q_proj", attn.q_proj), ("k_proj", attn.k_proj),
                             ("v_proj", attn.v_proj), ("o_proj", attn.o_proj),
                             ("gate_proj", mlp.gate_proj), ("up_proj", mlp.up_proj),
                             ("down_proj", mlp.down_proj)):
            base = getattr(module, "base_layer", module)
            if not isinstance(base, bnb.nn.Linear4bit):
                return False
            state = getattr(base.weight, "quant_state", None)
            if state is None or state.quant_type != "nf4":
                return False
            if hasattr(module, "lora_A"):
                if name not in ("q_proj", "v_proj") or getattr(module, "merged", False) \
                        or getattr(module, "disable_adapters", False):
                    return False
                active = list(getattr(module, "active_adapters", []))
                if len(active) != 1 or active[0] not in module.lora_A:
                    return False
                if getattr(module, "use_dora", {}).get(active[0], False):
                    return False
                if active[0] in getattr(module, "lora_variant", {}):
                    return False
            elif name in ("q_proj", "v_proj"):
                return False
    return True


def _embedding_to_fp16(causal):
    """The tied embedding / lm_head weight is frozen and fp32 only because peft upcast every
    non-quantized parameter; if every value is exactly representable in fp16, keep it in fp16."""
    weight = causal.model.embed_tokens.weight
    if weight.dtype != torch.float32:
        return False
    half = weight.data.to(torch.float16)
    if not torch.equal(half.float(), weight.data):
        del half
        return False
    weight.data = half
    torch.cuda.empty_cache()
    return True


def fused_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    engine = getattr(self, _INSTALLED_ATTR, None)
    original = getattr(self, "_soup_original_compute_loss")
    usable = (
        engine is not None
        and getattr(model, "training", False)
        and isinstance(inputs, dict)
        and set(inputs) <= {"input_ids", "attention_mask", "labels", "_prediction_loss_only"}
        and all(key in inputs for key in ("input_ids", "attention_mask", "labels"))
        and self.compute_loss_func is None
        and self.label_smoother is None
    )
    if not usable:
        return original(model, inputs, return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch)
    inputs.pop("_prediction_loss_only", None)
    input_ids, attention_mask, labels = (inputs["input_ids"], inputs["attention_mask"],
                                         inputs["labels"])

    # HF Trainer: sum over the batch / num_items_in_batch when the model takes loss kwargs,
    # otherwise the model's mean over supervised tokens.
    if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
        denom = num_items_in_batch
    else:
        denom = (labels[:, 1:] != -100).sum()
    denom = float(denom.item() if torch.is_tensor(denom) else denom)
    loss, packed, stats = engine.step(input_ids, attention_mask, labels, denom)
    if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs \
            and num_items_in_batch is not None:
        scale = self.accelerator.num_processes
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None:
            scale //= pc.tp_size
        loss = loss * (scale if self.args.n_gpu <= 1 else self.args.n_gpu)

    # TRL SFTTrainer metrics: entropy over every real token, token accuracy over supervised
    # next-token targets, running count of real tokens.
    mode = "train"
    with torch.no_grad():
        valid = packed.targets >= 0
        n_valid = valid.sum()
        summary = torch.stack([stats.entropy.sum(), stats.correct.sum().float(),
                               n_valid.float()]).tolist()
        entropy = summary[0] / max(packed.n, 1)
        accuracy = summary[1] / summary[2] if summary[2] > 0 else 0.0
    self._metrics[mode]["entropy"].append(entropy)
    self._total_train_tokens += int(packed.n)
    self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
    self._metrics[mode]["mean_token_accuracy"].append(accuracy)

    # Soup's rewind recorder: per-row mean cross-entropy over supervised tokens.
    state = getattr(self, "_soup_rewind_state", None)
    if state is not None and not state.failed:
        try:
            with torch.no_grad():
                rows = packed.rows
                totals = torch.zeros(rows, device=labels.device, dtype=torch.float32)
                totals.index_add_(0, packed.row_of_token, stats.tok_loss)
                counts = (labels[:, 1:] != -100).sum(dim=1)
                means = totals / counts.clamp(min=1).to(totals.dtype)
            state.on_micro_batch(step=int(self.state.global_step) + 1,
                                 row_loss=[float(x) for x in means.tolist()],
                                 row_tokens=[int(x) for x in counts.tolist()])
        except Exception as exc:  # the recorder never raises into training
            state.fail(exc)

    if return_outputs:
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return loss, CausalLMOutputWithPast(loss=loss, logits=None)
    return loss


def install(trainer, seed=0):
    model = getattr(trainer, "model", None)
    if getattr(trainer, _INSTALLED_ATTR, None) is not None:
        return True
    if not _is_supported(model):
        _log("model not recognized, training with the original code path")
        return False
    from .engine import FusedQwen2

    causal = model.get_base_model()
    fp16_embedding = _embedding_to_fp16(causal)
    engine = FusedQwen2(model, seed=seed)
    trainer._soup_original_compute_loss = trainer.compute_loss
    setattr(trainer, _INSTALLED_ATTR, engine)
    trainer.compute_loss = types.MethodType(fused_compute_loss, trainer)
    mib = 2**20
    _log(f"fused Qwen2 step installed ({len(engine.layers)} layers, LoRA dropout "
         f"{engine.dropout_p}, embedding in fp16: {fp16_embedding}); CUDA memory before "
         f"training: {torch.cuda.memory_allocated() // mib} MiB allocated, "
         f"{torch.cuda.memory_reserved() // mib} MiB reserved, peak reserved so far "
         f"{torch.cuda.max_memory_reserved() // mib} MiB")
    return True


def patch_soup():
    from soup_cli.trainer import sft

    wrapper = sft.SFTTrainerWrapper
    if getattr(wrapper.train, "_soup_kernels_patched", False):
        return
    original_train = wrapper.train

    original_setup = wrapper.setup

    @functools.wraps(original_setup)
    def setup(self, *args, **kwargs):
        base = getattr(getattr(self, "config", None), "base", None)
        if isinstance(base, str) and torch.cuda.is_available():
            try:
                from . import warmup

                warmup.start(base)
            except Exception as exc:  # noqa: BLE001 - a missed warm-up only costs time
                _log(f"kernel warm-up not started ({exc.__class__.__name__}: {exc})")
        return original_setup(self, *args, **kwargs)

    @functools.wraps(original_train)
    def train(self, *args, **kwargs):
        trainer = getattr(self, "trainer", None)
        if trainer is not None and isinstance(getattr(trainer, "model", None), torch.nn.Module):
            try:
                from . import warmup

                warmup.wait()
                seed = int(getattr(getattr(trainer, "args", None), "seed", 0) or 0)
                install(trainer, seed=seed)
            except Exception as exc:  # noqa: BLE001 - never break Soup
                _log(f"not installed ({exc.__class__.__name__}: {exc})")
        return original_train(self, *args, **kwargs)

    train._soup_kernels_patched = True
    wrapper.setup = setup
    wrapper.train = train
