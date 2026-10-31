"""The NF4 layer weights of the 4-bit Qwen2 go to host memory as they are quantized.

Soup loads the model with every weight on the GPU, and the run's GPU memory peak is the end of
that load: all ~650 MiB of packed NF4 weights at once, quantized on the GPU one by one. Our step
streams them from host memory (engine.WeightStream), so they do not need to be on the GPU at all
after their quantization:

- each Linear4bit weight is quantized by bitsandbytes on the GPU as before (the same code, the
  same values); right after that its packed bytes and its absmax are copied to host memory and
  the GPU copies are freed (the small double-quantization state, which our kernel reads, stays);
- transformers' warm-up of the caching allocator, which reserves GPU memory for every weight the
  load will put there, leaves these weights out (it reserves only what really stays);
- while a Linear4bit weight is in host memory, its module keeps it there through `model.to()`
  (accelerate moves the model to the GPU when training starts) - the LoRA layers peft created on
  the weight's device move as usual;
- the fused step takes the host bytes as they are (patch.install builds the WeightStream at once).

The host copies go straight into one pinned buffer laid out as engine.WeightStream streams it
(`layer_layout`: per layer, each group's packed bytes back to back, then their absmax), without
waiting for each copy: the WeightStream then takes that buffer as it is instead of copying ~650
MiB into one of its own, and the load does not stop at every weight for a copy into fresh
pageable memory. One synchronize at the end of the load. A load that does not fit the plan (a
weight of another size, a layer missing) copies the rest as before; the WeightStream then copies.

Only for the load the fused step is for: a Qwen2 that bitsandbytes quantizes to NF4 on one GPU on
Soup's fp16 path. If the step is not installed after all, `restore()` puts every weight back on the
GPU before Soup's own code trains.
"""

import functools
import re
import types
import weakref

import torch

_LOADING = []  # weakref to the model being loaded with host NF4 weights
_ON_HOST = []  # (module, device) of Linear4bit modules whose weight is in host memory
# the pinned-buffer plan of the load going on (see _plan), or None: copies as before
_PLAN = []
# the groups engine.WeightStream stores back to back (its GROUPS), in its order
GROUPS = (("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"), ("down_proj",))
_WEIGHT = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.weight$")


def _align(n, a=256):
    return (n + a - 1) // a * a


def layer_layout(sizes):
    """{proj: (packed bytes, absmax bytes)} of one layer -> ({proj: (packed offset, absmax
    offset)} from the layer's start, the layer's bytes), as engine.WeightStream lays them out."""
    offsets, offset = {}, 0
    for names in GROUPS:
        packed_at = offset
        absmax_at = _align(packed_at + sum(sizes[n][0] for n in names))
        offset = _align(absmax_at + sum(sizes[n][1] for n in names))
        po, ao = packed_at, absmax_at
        for n in names:
            offsets[n] = (po, ao)
            po, ao = po + sizes[n][0], ao + sizes[n][1]
    return offsets, offset


def _plan(model, names, quant):
    """Where each NF4 weight of a Qwen2 (every layer the same projections) goes in one pinned
    buffer, or None: {"layers", "layer_bytes", "offsets": {proj: (packed, absmax offset)},
    "sizes": {proj: (packed, absmax bytes)}, "buffer" (made at the first weight), "pending"
    (copies not synchronized yet)}. Sizes as bitsandbytes makes them: 4 bits per value, one absmax
    per 64-value block (uint8 with double quantization, else fp32); a weight that turns out
    otherwise ends the plan (_into_plan)."""
    layers = {}
    for name in names:
        match = _WEIGHT.search(name)
        numel = model.get_parameter(name).numel()
        if match is None or numel % 64:
            return None
        absmax = numel // 64 * (1 if getattr(quant, "bnb_4bit_use_double_quant", False) else 4)
        layers.setdefault(int(match.group(1)), {})[match.group(2)] = (numel // 2, absmax)
    projs = {n for names in GROUPS for n in names}
    if not layers or sorted(layers) != list(range(len(layers))) \
            or any(set(sizes) != projs or sizes != layers[0] for sizes in layers.values()):
        return None
    offsets, layer_bytes = layer_layout(layers[0])
    return {"layers": len(layers), "layer_bytes": layer_bytes, "offsets": offsets,
            "sizes": layers[0], "buffer": None, "pending": False}


def _eligible(model, expanded_device_map, hf_quantizer):
    config = getattr(model, "config", None)
    quant = getattr(hf_quantizer, "quantization_config", None)
    if getattr(config, "model_type", None) != "qwen2" or hf_quantizer is None:
        return False
    if getattr(hf_quantizer, "pre_quantized", True) or not getattr(quant, "load_in_4bit", False) \
            or getattr(quant, "bnb_4bit_quant_type", None) != "nf4":
        return False
    if getattr(model, "dtype", None) != torch.float16:
        return False
    devices = {torch.device(d) if not isinstance(d, int) else torch.device("cuda", d)
               for d in expanded_device_map.values()}
    cuda = {d for d in devices if d.type == "cuda"}
    if len(cuda) != 1 or not devices <= cuda | {torch.device("cpu")}:
        return False
    from .warmup import fp16_path_gpu

    return fp16_path_gpu()


def patch_model_loading():
    from transformers import modeling_utils
    from transformers.integrations import bitsandbytes as hf_bnb

    warmup = modeling_utils.caching_allocator_warmup
    if not getattr(warmup, "_soup_kernels_patched", False):
        @functools.wraps(warmup)
        def caching_allocator_warmup(model, expanded_device_map, hf_quantizer):
            _LOADING.clear()
            try:
                eligible = _eligible(model, expanded_device_map, hf_quantizer)
            except Exception:  # noqa: BLE001 - anything unexpected: load as Soup asked
                eligible = False
            _PLAN.clear()
            if eligible:
                _LOADING.append(weakref.ref(model))
                try:
                    _PLAN.append(_plan(model, [name for name in expanded_device_map
                                               if hf_quantizer.param_needs_quantization(
                                                   model, name)],
                                       getattr(hf_quantizer, "quantization_config", None)))
                except Exception:  # noqa: BLE001 - copies as before
                    _PLAN.append(None)
                expanded_device_map = {
                    name: device for name, device in expanded_device_map.items()
                    if not hf_quantizer.param_needs_quantization(model, name)}
            return warmup(model, expanded_device_map, hf_quantizer)

        caching_allocator_warmup._soup_kernels_patched = True
        modeling_utils.caching_allocator_warmup = caching_allocator_warmup

    convert = hf_bnb.Bnb4bitQuantize.convert
    if not getattr(convert, "_soup_kernels_patched", False):
        @functools.wraps(convert)
        def quantize(self, input_dict, full_layer_name=None, model=None, **kwargs):
            result = convert(self, input_dict, full_layer_name=full_layer_name, model=model,
                             **kwargs)
            if model is not None and _LOADING and _LOADING[0]() is model:
                for name, param in result.items():
                    _to_host(model, name, param)
            return result

        quantize._soup_kernels_patched = True
        hf_bnb.Bnb4bitQuantize.convert = quantize

    load = modeling_utils.convert_and_load_state_dict_in_model
    if not getattr(load, "_soup_kernels_hostload", False):
        @functools.wraps(load)
        def convert_and_load_state_dict_in_model(*args, **kwargs):
            try:
                return load(*args, **kwargs)
            finally:
                plan = _PLAN[0] if _PLAN else None
                if plan is not None and plan["pending"]:
                    torch.cuda.synchronize()  # the host copies are done before anyone reads them
                    plan["pending"] = False

        convert_and_load_state_dict_in_model._soup_kernels_hostload = True
        modeling_utils.convert_and_load_state_dict_in_model = convert_and_load_state_dict_in_model


def take_arena():
    """The pinned buffer the NF4 weights of the last load went into as planned, or None; for
    engine.WeightStream, which checks every weight is where it expects it and then streams from it
    (the parameters keep it alive, not this module)."""
    plan = _PLAN.pop() if _PLAN else None
    return None if plan is None or plan["pending"] else plan["buffer"]


def _into_plan(name, param, state):
    """Start copying the packed bytes and absmax of weight `name` into their place in the plan's
    pinned buffer and point the parameter there; False if the plan has no place for it."""
    plan = _PLAN[0] if _PLAN else None
    match = _WEIGHT.search(name) if plan is not None else None
    if match is None or match.group(2) not in plan["offsets"]:
        return False
    proj, layer = match.group(2), int(match.group(1))
    packed, absmax = param.data, state.absmax
    sizes = (packed.numel() * packed.element_size(), absmax.numel() * absmax.element_size())
    if sizes != plan["sizes"][proj] or layer >= plan["layers"] \
            or not packed.is_contiguous() or not absmax.is_contiguous():
        _PLAN[0] = None  # not as planned: the rest as before (and WeightStream copies)
        if plan["pending"]:
            torch.cuda.synchronize()
        return False
    if plan["buffer"] is None:
        plan["buffer"] = torch.empty((plan["layers"] * plan["layer_bytes"],), dtype=torch.uint8,
                                     pin_memory=True)
    start = layer * plan["layer_bytes"]
    po, ao = (start + offset for offset in plan["offsets"][proj])
    host_packed = plan["buffer"][po:po + sizes[0]]
    host_absmax = plan["buffer"][ao:ao + sizes[1]]
    # on the stream that quantized them; the device copies freed below are reused only by work
    # queued after these copies
    host_packed.copy_(packed.reshape(-1).view(torch.uint8), non_blocking=True)
    host_absmax.copy_(absmax.reshape(-1).view(torch.uint8), non_blocking=True)
    plan["pending"] = True
    param.data = host_packed.view(packed.dtype).view(packed.shape)
    state.absmax = host_absmax.view(absmax.dtype).view(absmax.shape)
    return True


def _to_host(model, name, param):
    import bitsandbytes as bnb
    from transformers.quantizers.quantizers_utils import get_module_from_name

    state = getattr(param, "quant_state", None)
    if not isinstance(param, bnb.nn.Params4bit) or state is None or not param.is_cuda:
        return
    module, _ = get_module_from_name(model, name)
    if not isinstance(module, bnb.nn.Linear4bit):
        return
    device = param.device
    if not _into_plan(name, param, state):
        param.data = param.data.cpu()
        state.absmax = state.absmax.cpu()
    # (no empty_cache() here: with the harness's expandable segments the next weight reuses the
    # freed GPU copies and the reservation stays at its load size, while one call per weight
    # synchronized the device 196 times; WeightStream empties the cache once at install)
    _ON_HOST.append((module, device))
    module._apply = types.MethodType(_held_apply, module)


def _held_apply(module, fn, *args, **kwargs):
    """Module._apply for a Linear4bit: a weight in host memory stays there (the rest of the
    module, the bias, moves as usual); once the weight is back on the GPU, as before."""
    params = module._parameters
    weight = params.get("weight")
    if weight is None or weight.device.type != "cpu":
        return type(module)._apply(module, fn, *args, **kwargs)
    order = list(params)
    del params["weight"]
    try:
        return type(module)._apply(module, fn, *args, **kwargs)
    finally:
        rest = dict(params)
        params.clear()
        for key in order:
            params[key] = weight if key == "weight" else rest[key]


def restore(model=None):
    """Put the NF4 weights the loader left in host memory back on their GPU, and the parameters
    peft created next to them (LoRA) with them (the fused step is not used: Soup's own code runs
    the model)."""
    _PLAN.clear()
    device = None
    while _ON_HOST:
        module, device = _ON_HOST.pop()
        weight = module.weight
        if weight.device.type == "cpu":
            weight.data = weight.data.to(device)
            weight.quant_state.absmax = weight.quant_state.absmax.to(device)
    if device is not None and isinstance(model, torch.nn.Module):
        for param in model.parameters():
            if param.requires_grad and param.device.type == "cpu":
                param.data = param.data.to(device)
