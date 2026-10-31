"""The NF4 weights put in host memory during the load (solution/hostload.py): the same values,
the same fused step, and Soup's own path after restore()."""
import torch

from solution.test_kernels import _tiny_model

DEV = "cuda"


def _load(tmp_path, host):
    from transformers import modeling_utils
    from transformers.integrations import bitsandbytes as hf_bnb

    from solution import hostload, warmup

    saved = (modeling_utils.caching_allocator_warmup, hf_bnb.Bnb4bitQuantize.convert,
             warmup.fp16_path_gpu)
    try:
        if host:
            warmup.fp16_path_gpu = lambda: True  # the T4 path, also on a bf16 GPU
            hostload.patch_model_loading()
        return _tiny_model(tmp_path, dropout=0.1)
    finally:
        (modeling_utils.caching_allocator_warmup, hf_bnb.Bnb4bitQuantize.convert,
         warmup.fp16_path_gpu) = saved


def _linear4bit(model):
    import bitsandbytes as bnb

    return [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]


def test_host_load_same_step(tmp_path):
    from solution.engine import FusedQwen2

    torch.manual_seed(8)
    ids = torch.randint(0, 3000, (2, 33), device=DEV)
    mask = torch.ones_like(ids)
    mask[1, 20:] = 0
    labels = torch.where(mask.bool(), ids, -100)

    def run(host):
        model = _load(tmp_path / ("h" if host else "d"), host)
        model.train()
        modules = _linear4bit(model)
        assert all(m.weight.device.type == ("cpu" if host else "cuda") for m in modules)
        assert all(m.weight.quant_state.absmax.device.type == ("cpu" if host else "cuda")
                   for m in modules)
        model.to(DEV)  # what accelerate does when training starts: host weights stay there
        assert all(m.weight.device.type == ("cpu" if host else "cuda") for m in modules)
        params = [p for p in model.parameters() if p.requires_grad]
        assert all(p.is_cuda for p in params)
        # the same LoRA values for both loads (_tiny_model draws lora_B on the parameter's device)
        g = torch.Generator().manual_seed(1)
        for p in params:
            p.data = (torch.randn(p.shape, generator=g) * 0.02).to(p.device, p.dtype)
        torch.cuda.manual_seed(2)  # the LoRA dropout masks (TorchDropout)
        engine = FusedQwen2(model)
        assert engine.weights_on_host() == host
        engine.offload_weights()
        loss, _, _ = engine.step(ids, mask, labels, 60.0)
        grads = torch.autograd.grad(loss, params)
        packed = [m.weight.data.clone() for m in modules]
        engine.restore_weights()
        assert all(m.weight.is_cuda and m.weight.quant_state.absmax.is_cuda for m in modules)
        return loss, grads, packed

    loss_a, grads_a, packed_a = run(False)
    loss_b, grads_b, packed_b = run(True)
    assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(packed_a, packed_b))
    assert torch.equal(loss_a, loss_b)
    assert all(torch.equal(a, b) for a, b in zip(grads_a, grads_b))


def test_restore_for_soups_own_path(tmp_path):
    from solution import hostload

    model = _load(tmp_path, True)
    assert all(m.weight.device.type == "cpu" for m in _linear4bit(model))
    hostload.restore(model)
    assert all(m.weight.is_cuda and m.weight.quant_state.absmax.is_cuda
               for m in _linear4bit(model))
    ids = torch.randint(0, 3000, (2, 16), device=DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        model(input_ids=ids)


def _load_to_host(tmp_path, dropout):
    """The tiny model loaded as Soup's run loads it with the solution applied: the NF4 weights
    (hostload) and the embedding (offload) go to host memory during the load."""
    from transformers import modeling_utils
    from transformers.integrations import bitsandbytes as hf_bnb

    from solution import hostload, offload, warmup

    cls = modeling_utils.PreTrainedModel
    saved = (modeling_utils.caching_allocator_warmup, hf_bnb.Bnb4bitQuantize.convert,
             cls.__dict__["_load_pretrained_model"], warmup.fp16_path_gpu)
    try:
        warmup.fp16_path_gpu = lambda: True  # the T4 path, also on a bf16 GPU
        hostload.patch_model_loading()
        offload.patch_model_loading()
        return _tiny_model(tmp_path, dropout=dropout)
    finally:
        (modeling_utils.caching_allocator_warmup, hf_bnb.Bnb4bitQuantize.convert,
         loader, warmup.fp16_path_gpu) = saved
        cls._load_pretrained_model = loader


def test_fallback_after_host_load(tmp_path):
    """Weights and embedding in host memory from the load on: a fused step, Soup's own code for an
    eval call (everything back on the GPU), a fused step again - the same loss as the first."""
    from solution import patch as P
    from solution.test_fallback import _batch, _call, _trainer

    model = _load_to_host(tmp_path, dropout=0.0)
    assert all(m.weight.device.type == "cpu" for m in _linear4bit(model))
    assert model.get_input_embeddings().weight.device.type == "cpu"
    model.to(DEV)  # accelerate, when training starts
    t = _trainer(model)
    engine = t._soup_kernels_engine
    assert engine.weights_on_host()
    engine.offload_weights()  # what patch.install does when the loader left them on the host
    model.train()
    first = _call(t, model, _batch())
    first.backward()
    P._flush_metrics(t)
    model.eval()
    with torch.no_grad():
        _call(t, model, _batch())
    assert all(m.weight.is_cuda for m in _linear4bit(model))
    assert model.get_input_embeddings().weight.is_cuda
    model.train()
    again = _call(t, model, _batch())
    again.backward()
    P._flush_metrics(t)
    assert torch.equal(first.detach(), again.detach())
