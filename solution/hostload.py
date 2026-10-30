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

Only for the load the fused step is for: a Qwen2 that bitsandbytes quantizes to NF4 on one GPU on
Soup's fp16 path. If the step is not installed after all, `restore()` puts every weight back on the
GPU before Soup's own code trains.
"""

import functools
import types
import weakref

import torch

_LOADING = []  # weakref to the model being loaded with host NF4 weights
_ON_HOST = []  # (module, device) of Linear4bit modules whose weight is in host memory


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
            if eligible:
                _LOADING.append(weakref.ref(model))
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
    param.data = param.data.cpu()
    state.absmax = state.absmax.cpu()
    # the freed GPU copies (and the fp16 tensor quantized from) go back to the driver: otherwise
    # the allocator's cache of them grows with the load and the reservation peaks anyway
    torch.cuda.empty_cache()
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
