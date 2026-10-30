"""How the fused step plugs into Soup: patches of `SFTTrainerWrapper.setup` / `.train` and of
peft's `prepare_model_for_kbit_training`.

- `setup`: starts compiling our kernels in the background (solution/warmup.py) while Soup
  loads the model;
- `prepare_model_for_kbit_training`: leaves the frozen, tied embedding / lm_head of a Qwen2 in
  its fp16 load dtype instead of upcasting it to fp32 (933 MiB instead of 467, and a 1.4 GiB
  transient during the cast). Our step reads it in fp16 exactly as the base does: the base's
  fp32 embedding rows are exact copies of fp16 values, and its lm_head runs in fp16 under
  autocast. If the fused step is not installed after all, the fp32 copy is restored;
- `train`: when the trainer and the 4-bit PEFT model exist and are the ones we know (Qwen2 +
  bitsandbytes NF4 + LoRA on q/v projections), we replace the trainer's `compute_loss` with `fused_compute_loss`, which runs `FusedQwen2.step`
  and then does exactly the bookkeeping HF Trainer, TRL's SFTTrainer and Soup's rewind recorder
  do in theirs (same loss normalization, same logged metrics, same rewind rows).
Anything we do not recognize falls through to the original code untouched.
"""

import functools
import sys
import time
import types

import torch

_INSTALLED_ATTR = "_soup_kernels_engine"
OFFLOAD_WEIGHTS = True  # stream the NF4 layer weights from pinned host memory (engine.WeightStream)
# Bytes kept free under the model-load peak when a step decides what to keep. Beyond what PyTorch
# allocates, a long step makes the allocator's reservation grow (fragmentation) and CUDA modules
# load during training; with 64 MiB a T4 run peaked 36 MiB above the load peak, with 112 MiB it
# stays at it, at no measurable cost in time (an empty_cache() per step instead cost ~6%).
KEEP_MARGIN = 112 * 2**20
# Memory the steps may use above the model-load peak to keep activations instead of recomputing
# them (0: the run's peak stays the model load).
EXTRA_MEMORY_MIB = 0


def _restore_callback_base():
    import transformers

    return transformers.TrainerCallback


class _RestoreWeights(_restore_callback_base()):
    """When training ends, frees the weight-streaming slots; the 4-bit weights come back to the
    GPU on the first forward call anyone makes afterwards (saving the adapter makes none)."""

    def __init__(self, engine):
        self.engine = engine

    def on_train_end(self, args, state, control, **kwargs):
        self.engine.finish_training()


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


_KEPT_FP16 = []  # embedding weights prepare_model_for_kbit_training left in fp16


def _embedding_to_fp16(causal):
    """The tied embedding / lm_head weight is frozen; if it is fp32 (peft upcast it) and every
    value is exactly representable in fp16, keep it in fp16. Checked in slices so no second
    full-size copy is ever alive."""
    weight = causal.model.embed_tokens.weight
    if weight.dtype == torch.float16:
        return True
    if weight.dtype != torch.float32:
        return False
    for rows in weight.data.split(8192):
        if not torch.equal(rows.to(torch.float16).float(), rows):
            return False
    weight.data = weight.data.to(torch.float16)
    torch.cuda.empty_cache()
    return True


def _restore_fp32_embeddings():
    while _KEPT_FP16:
        weight = _KEPT_FP16.pop()
        if weight.dtype != torch.float32:
            weight.data = weight.data.to(torch.float32)


def patch_prepare_kbit():
    import peft

    original = peft.prepare_model_for_kbit_training
    if getattr(original, "_soup_kernels_patched", False):
        return

    @functools.wraps(original)
    def prepare(model, *args, **kwargs):
        weight = None
        try:
            config = getattr(model, "config", None)
            embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings")                 else None
            if getattr(config, "model_type", None) == "qwen2" and embed is not None                     and getattr(model, "is_loaded_in_4bit", False)                     and embed.weight.dtype == torch.float16 and embed.weight.is_cuda:
                weight = embed.weight
        except Exception:  # noqa: BLE001
            weight = None
        if weight is None:
            return original(model, *args, **kwargs)
        data = weight.data
        weight.data = torch.empty(0, device=data.device, dtype=data.dtype)
        try:
            result = original(model, *args, **kwargs)
        finally:
            weight.data = data
        _KEPT_FP16.append(weight)
        return result

    prepare._soup_kernels_patched = True
    peft.prepare_model_for_kbit_training = prepare


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
    if OFFLOAD_WEIGHTS:
        engine.offload_weights()  # first step: move the NF4 layer weights to pinned host memory
    loss, packed, stats = engine.step(input_ids, attention_mask, labels, denom)
    if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs \
            and num_items_in_batch is not None:
        scale = self.accelerator.num_processes
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None:
            scale //= pc.tp_size
        loss = loss * (scale if self.args.n_gpu <= 1 else self.args.n_gpu)

    # TRL SFTTrainer metrics and Soup's rewind rows. The running token count is known on the
    # host; entropy, accuracy and per-row losses need the step's results, and reading them here
    # would stall the GPU between forward and backward. They are recorded once backward is
    # queued (on_pre_optimizer_step, after HF Trainer's own check of the loss has synchronized),
    # which is still before the optimizer step and the log - same values, same order.
    mode = "train"
    _flush_metrics(self)  # a previous micro-batch's, if gradient accumulation queued one
    metrics = self._metrics[mode]
    metrics["entropy"], metrics["mean_token_accuracy"]  # noqa: B018 - keys in TRL's order
    self._total_train_tokens += int(packed.n)
    metrics["num_tokens"] = [self._total_train_tokens]
    self._soup_pending_metrics = (packed, stats, labels, int(self.state.global_step) + 1)

    if return_outputs:
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return loss, CausalLMOutputWithPast(loss=loss, logits=None)
    return loss


def _flush_metrics(trainer):
    """Record the queued micro-batch's TRL metrics and Soup rewind row, as TRL's compute_loss and
    Soup's RewindTrainer.compute_loss would have."""
    pending = getattr(trainer, "_soup_pending_metrics", None)
    if pending is None:
        return
    trainer._soup_pending_metrics = None
    packed, stats, labels, step = pending
    with torch.no_grad():
        valid = packed.targets >= 0
        summary = torch.stack([stats.entropy.sum(), stats.correct.sum().float(),
                               valid.sum().float()]).tolist()
    trainer._metrics["train"]["entropy"].append(summary[0] / max(packed.n, 1))
    trainer._metrics["train"]["mean_token_accuracy"].append(
        summary[1] / summary[2] if summary[2] > 0 else 0.0)

    # Soup's rewind recorder: per-row mean cross-entropy over supervised tokens.
    state = getattr(trainer, "_soup_rewind_state", None)
    if state is not None and not state.failed:
        try:
            with torch.no_grad():
                totals = torch.zeros(packed.rows, device=labels.device, dtype=torch.float32)
                totals.index_add_(0, packed.row_of_token.index_select(0, stats.order),
                                  stats.tok_loss)
                counts = (labels[:, 1:] != -100).sum(dim=1)
                means = totals / counts.clamp(min=1).to(totals.dtype)
            state.on_micro_batch(step=step, row_loss=[float(x) for x in means.tolist()],
                                 row_tokens=[int(x) for x in counts.tolist()])
        except Exception as exc:  # the recorder never raises into training
            state.fail(exc)


class _FlushMetrics(_restore_callback_base()):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        _flush_metrics(self.trainer)

    def on_step_end(self, args, state, control, **kwargs):  # in case no optimizer step ran
        _flush_metrics(self.trainer)


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
    # The run's GPU memory peak so far is the model load. Steps may use memory below it (minus a
    # margin for allocator slack) to keep activations instead of recomputing them.
    engine.set_memory_cap(torch.cuda.max_memory_allocated(engine.device) - KEEP_MARGIN
                          + EXTRA_MEMORY_MIB * 2**20)
    trainer._soup_original_compute_loss = trainer.compute_loss
    setattr(trainer, _INSTALLED_ATTR, engine)
    trainer.compute_loss = types.MethodType(fused_compute_loss, trainer)
    trainer.add_callback(_RestoreWeights(engine))
    trainer.add_callback(_FlushMetrics(trainer))
    mib = 2**20
    _log(f"fused Qwen2 step installed ({len(engine.layers)} layers, LoRA dropout "
         f"{engine.dropout_p}, embedding in fp16: {fp16_embedding}); CUDA memory before "
         f"training: {torch.cuda.memory_allocated() // mib} MiB allocated, "
         f"{torch.cuda.memory_reserved() // mib} MiB reserved, peak reserved so far "
         f"{torch.cuda.max_memory_reserved() // mib} MiB")
    return True


def patch_library_overheads():
    """Two pieces of work the `soup train` command does twice or for nothing, done once:

    - `importlib.metadata.packages_distributions()` walks every installed distribution (2.3 s
      on the Kaggle image, and nothing caches it). transformers calls it at import and keeps the
      result in `PACKAGE_DISTRIBUTION_MAPPING`; trl's import calls it again. Later callers get a
      fresh copy of a snapshot of that same mapping;
    - `datasets.Dataset.map` on an in-memory dataset hashes the transform with dill to name a
      cache file it never writes (no `cache_files`); the transforms here close over the
      tokenizer, so that is the whole 151,936-token vocabulary pickled per call. For such a
      dataset we hand `map` a random fingerprint, as `datasets` itself does when hashing fails.
      The mapped rows are the same; only the fingerprint string, which nothing reads, differs.
    """
    import importlib.metadata as metadata

    if not getattr(metadata.packages_distributions, "_soup_kernels_patched", False):
        computed = []

        def packages_distributions():
            if not computed:
                mapping = None
                module = sys.modules.get("transformers.utils.import_utils")
                mapping = getattr(module, "PACKAGE_DISTRIBUTION_MAPPING", None)
                if not isinstance(mapping, dict):
                    mapping = original_packages_distributions()
                # a snapshot, so a caller that mutates transformers' dict (or its own copy)
                # cannot change what the others get
                computed.append({name: list(dists) for name, dists in mapping.items()})
            return {name: list(dists) for name, dists in computed[0].items()}

        original_packages_distributions = metadata.packages_distributions
        packages_distributions._soup_kernels_patched = True
        metadata.packages_distributions = packages_distributions

    try:
        from datasets import Dataset
        from datasets.fingerprint import generate_random_fingerprint
    except Exception:  # noqa: BLE001
        return
    if getattr(Dataset.map, "_soup_kernels_patched", False):
        return
    original_map = Dataset.map

    @functools.wraps(original_map)
    def map_(self, *args, **kwargs):
        if kwargs.get("new_fingerprint") is None and len(args) < 18 and not self.cache_files:
            kwargs["new_fingerprint"] = generate_random_fingerprint()
        return original_map(self, *args, **kwargs)

    map_._soup_kernels_patched = True
    Dataset.map = map_


def patch_soup():
    patch_library_overheads()
    patch_prepare_kbit()
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
            installed = False
            try:
                from . import warmup

                waited, compile_s = warmup.wait()
                started = time.perf_counter()
                seed = int(getattr(getattr(trainer, "args", None), "seed", 0) or 0)
                installed = install(trainer, seed=seed)
                _log(f"kernel warm-up took {compile_s or 0:.1f} s in the background; training "
                     f"waited {waited:.2f} s for it; installing the fused step took "
                     f"{time.perf_counter() - started:.2f} s")
            except Exception as exc:  # noqa: BLE001 - never break Soup
                _log(f"not installed ({exc.__class__.__name__}: {exc})")
            if not installed:
                _restore_fp32_embeddings()
        return original_train(self, *args, **kwargs)

    train._soup_kernels_patched = True
    wrapper.setup = setup
    wrapper.train = train
