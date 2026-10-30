"""The tied embedding / lm_head weight of the 4-bit Qwen2 never sits on the GPU.

Soup loads the model with every parameter on the GPU; the frozen fp16 embedding (151,936 x 1536,
445 MiB) is the first weight loaded, so it is on the GPU while all the NF4 weights are quantized
and the load peaks with both. Our step keeps it in pinned host memory (engine.HostHead), so the
loader puts it in host memory from the start: for the load of a Qwen2 that bitsandbytes quantizes
to 4 bits in fp16, `model.embed_tokens` / `lm_head` are mapped to "cpu" in the device map the
weight loader (and transformers' pre-allocation of device memory for the weights) reads. The
values are the same: the checkpoint's tensor converted to fp16, on the CPU instead of the GPU.
Nothing else of the load changes; `from_pretrained`'s own device map, `hf_device_map` and the
dispatch of the model are untouched.

A model on one device has no `hf_device_map` in transformers 5, so accelerate's `prepare_model`
moves it with `model.to(device)` when training starts. While the weight is in host memory, the
embedding and lm_head modules keep it there through such a move (`_held_apply`: the move is
applied to an empty stand-in, a dtype change is applied to the weight itself), the same way
patch.py hides it from peft's fp32 upcast.

If the fused step is not installed after all, `restore_embeddings()` puts the weight back on the
GPU before Soup's own code trains.
"""

import dataclasses
import functools
import types

import torch

_ON_HOST = []  # embedding weights loaded to host memory, until the fused step takes them


def _offloadable(model, load_config):
    config = getattr(model, "config", None)
    quantizer = load_config.hf_quantizer
    quant = getattr(quantizer, "quantization_config", None)
    device_map = load_config.device_map
    if getattr(config, "model_type", None) != "qwen2" or not getattr(
            config, "tie_word_embeddings", False):
        return False
    if quantizer is None or getattr(quantizer, "pre_quantized", True) \
            or not getattr(quant, "load_in_4bit", False) \
            or getattr(quant, "bnb_4bit_quant_type", None) != "nf4":
        return False
    if load_config.dtype != torch.float16 or not isinstance(device_map, dict) or not device_map:
        return False
    devices = {torch.device(d) if not isinstance(d, int) else torch.device("cuda", d)
               for d in device_map.values()}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        return False
    from .warmup import fp16_path_gpu

    return fp16_path_gpu()  # the GPUs the fused step runs on (patch.install)


def patch_model_loading():
    from transformers import modeling_utils

    cls = modeling_utils.PreTrainedModel
    original = cls._load_pretrained_model
    if getattr(original, "_soup_kernels_patched", False):
        return

    @functools.wraps(original)
    def load(model, state_dict, checkpoint_files, load_config, *args, **kwargs):
        try:
            offload = torch.cuda.is_available() and _offloadable(model, load_config)
        except Exception:  # noqa: BLE001 - anything unexpected: load as Soup asked
            offload = False
        if offload:
            device = next(iter(load_config.device_map.values()))
            device_map = dict(load_config.device_map)
            device_map.update({"model.embed_tokens": "cpu", "lm_head": "cpu"})
            load_config = dataclasses.replace(load_config, device_map=device_map)
        result = original(model, state_dict, checkpoint_files, load_config, *args, **kwargs)
        if offload:
            weight = model.get_input_embeddings().weight
            if weight.device.type == "cpu":
                _ON_HOST.append((weight, device))
                # (lm_head is tied to the embedding only after this, in _finalize_model_loading)
                for module in (model.get_input_embeddings(), model.get_output_embeddings()):
                    if module is not None:
                        module._apply = types.MethodType(_held_apply, module)
        return result

    load._soup_kernels_patched = True
    cls._load_pretrained_model = staticmethod(load)


def _held_apply(module, fn, *args, **kwargs):
    """Module._apply for the embedding / lm_head module: a weight in host memory stays there."""
    weight = getattr(module, "weight", None)
    if weight is None or weight.device.type != "cpu":
        return type(module)._apply(module, fn, *args, **kwargs)
    held = weight.data
    weight.data = torch.empty(0, dtype=held.dtype)
    try:
        return type(module)._apply(module, fn, *args, **kwargs)
    finally:
        dtype = weight.dtype
        weight.data = held if dtype == held.dtype else held.to(dtype)


def restore_embeddings():
    """Put embedding weights the loader left in host memory back on their device (the fused
    step is not used: Soup's own code runs the model)."""
    while _ON_HOST:
        weight, device = _ON_HOST.pop()
        if weight.device.type == "cpu":
            if isinstance(device, int):
                device = torch.device("cuda", device)
            weight.data = weight.data.to(device)
