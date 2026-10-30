"""The embedding / lm_head weight in pinned host memory (engine.HostHead): the streamed head
against the head with the weight on the GPU, and the whole step with both offloads.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

from solution.test_kernels import _tiny_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEV = "cuda"


class _Packed:  # what the head reads from engine.Packed
    dropout_p = 0.0


def _head_inputs(n, n_prompt, V, H, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    h = torch.randn((n, H), device=DEV, generator=g) * 2.0
    norm_w = 1.0 + 0.1 * torch.randn((H,), device=DEV, generator=g)
    weight = (torch.randn((V, H), device=DEV, generator=g) * 0.03).to(torch.float16)
    targets = torch.randint(0, V, (n,), device=DEV, generator=g)
    targets[torch.randperm(n, device=DEV, generator=g)[:n_prompt]] = -100
    packed = _Packed()
    packed.targets = targets
    supervised = targets >= 0
    packed.head_order = torch.argsort((~supervised).to(torch.int8), stable=True)
    packed.n_supervised = int(supervised.sum())
    return h, norm_w, weight, packed


@pytest.mark.parametrize("n,n_prompt", [(300, 120), (37, 37), (5, 0)])
def test_piece_kernels_match_ce_kernel(n, n_prompt):
    """On the same logits, the piece-by-piece kernels give lm_head_ce_kernel's loss, entropy and
    accuracy to the bit, and the gradient w.r.t. the normed hidden state as closely as the
    one-GEMM gradient of the GPU head."""
    from solution import kernels as K
    from solution.engine import CE_BLOCK, HostHead

    V, H = 151936, 1536
    h, _, weight, packed = _head_inputs(n, n_prompt, V, H, seed=n)
    order, n_sup = packed.head_order, packed.n_supervised
    targets = packed.targets.index_select(0, order)
    hn = h.index_select(0, order).to(torch.float16)
    logits = hn @ weight.t()

    ref = logits.clone()
    loss_a = torch.empty(n, device=DEV)
    ent_a = torch.empty(n, device=DEV)
    cor_a = torch.empty(n, device=DEV, dtype=torch.int32)
    K.lm_head_ce_kernel[(n,)](ref, targets, loss_a, ent_a, cor_a, V, 1, BLOCK=CE_BLOCK,
                              num_warps=8)
    dh_a = ref[:n_sup] @ weight

    state = torch.tensor([float("-inf"), 0.0, 0.0, float("-inf")], device=DEV).repeat(n, 1)
    best_idx = torch.zeros(n, device=DEV, dtype=torch.int32)
    xt = torch.zeros(n, device=DEV)
    alpha = torch.empty(max(n_sup, 1), device=DEV)
    acc = torch.zeros((max(n_sup, 1), H), device=DEV)
    wt = torch.zeros((max(n_sup, 1), H), device=DEV, dtype=torch.float16)
    for v0 in range(0, V, HostHead.PIECE):
        width = min(HostHead.PIECE, V - v0)
        piece = logits[:, v0:v0 + width].contiguous()
        w = weight[v0:v0 + width].contiguous()
        K.lm_head_piece_kernel[(n,)](piece, w, targets, state, best_idx, xt, alpha, wt, v0,
                                     width, n_sup, H, BLOCK=CE_BLOCK, HBLOCK=2048, num_warps=8)
        acc[:n_sup] = acc[:n_sup] * alpha[:n_sup, None] + piece[:n_sup].float() @ w.float()
    loss_b = torch.empty(n, device=DEV)
    ent_b = torch.empty(n, device=DEV)
    cor_b = torch.empty(n, device=DEV, dtype=torch.int32)
    dh_b = torch.empty((max(n_sup, 1), H), device=DEV, dtype=torch.float16)
    K.lm_head_finish_kernel[(n,)](state, best_idx, xt, targets, loss_b, ent_b, cor_b, acc, wt,
                                  dh_b, n_sup, H, HBLOCK=2048, num_warps=4)
    assert torch.equal(loss_a, loss_b)
    assert torch.equal(ent_a, ent_b)
    assert torch.equal(cor_a, cor_b)
    if n_sup:
        exact = (torch.softmax(logits[:n_sup].double(), -1)
                 - torch.nn.functional.one_hot(targets[:n_sup], V).double()) @ weight.double()
        err_a = (dh_a.double() - exact).norm() / exact.norm()
        err_b = (dh_b[:n_sup].double() - exact).norm() / exact.norm()
        assert err_b < 1e-3 and err_b <= 1.5 * err_a, (err_a.item(), err_b.item())


@pytest.mark.parametrize("n,n_prompt", [(820, 300), (300, 120), (37, 37)])
def test_streamed_head_matches_head(n, n_prompt):
    """The streamed head against the GPU head end to end. The logits GEMM runs on pieces of the
    vocabulary, and cuBLAS may sum a [n, 1536] x [1536, piece] product in another order than the
    full-vocabulary one (as the GPU head's own 256-row chunks already do): logits can differ by
    one fp16 ulp, so the comparison is to fp64 rather than to the bit."""
    from solution.engine import HeadStats, HostHead, _Head, _HostHead

    V, H, eps = 151936, 1536, 1e-6
    h, norm_w, weight, packed = _head_inputs(n, n_prompt, V, H, seed=n)
    denom = float(max(packed.n_supervised, 1))
    host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
    host.copy_(weight)
    head = HostHead(torch.nn.Parameter(host, requires_grad=False), torch.device(DEV))
    assert len(head.pieces) > HostHead.BUFFERS

    outs = []
    for streamed in (False, True):
        x = h.clone().requires_grad_(True)
        stats = HeadStats()
        if streamed:
            loss = _HostHead.apply(x, norm_w, eps, head, packed, denom, stats)
        else:
            loss = _Head.apply(x, norm_w, eps, weight, packed, denom, stats)
        (dx,) = torch.autograd.grad(loss, [x])
        outs.append((loss, stats, dx))
    (loss_a, sa, dx_a), (loss_b, sb, dx_b) = outs
    assert head.bufs is None  # the device buffers are released after the head
    torch.testing.assert_close(sa.tok_loss, sb.tok_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sa.entropy, sb.entropy, rtol=1e-5, atol=1e-5)
    assert (sa.correct != sb.correct).sum() <= 1
    torch.testing.assert_close(loss_a, loss_b, rtol=1e-6, atol=1e-6)

    # fp64 reference of the same math (fp16 normed input and weight)
    x = h.double().requires_grad_(True)
    hn = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w.double())
    hn = hn + (hn.to(torch.float16).double() - hn).detach()  # the fp16 GEMM input
    logits = hn @ weight.double().t()
    ref = torch.nn.functional.cross_entropy(logits, packed.targets.clamp(min=0),
                                            reduction="none")
    ref = (ref * (packed.targets >= 0)).sum() / denom
    (dref,) = torch.autograd.grad(ref, [x])
    if packed.n_supervised == 0:
        assert torch.equal(dx_b, torch.zeros_like(dx_b))
        return
    err_a = (dx_a.double() - dref).norm() / dref.norm()
    err_b = (dx_b.double() - dref).norm() / dref.norm()
    assert err_b < 1e-3 and err_b <= 1.5 * err_a, (err_a.item(), err_b.item())


def _batch(seed, lengths, prompt=3):
    torch.manual_seed(seed)
    B, T = len(lengths), max(lengths)
    ids = torch.randint(0, 3000, (B, T), device=DEV)
    mask = (torch.arange(T, device=DEV)[None] < torch.tensor(lengths, device=DEV)[:, None])
    mask = mask.long()
    ids = torch.where(mask.bool(), ids, 0)
    labels = torch.where(mask.bool(), ids, -100)
    labels[:, :prompt] = -100
    return ids, mask, labels


def test_host_head_step(tmp_path):
    """The whole step with the embedding / lm_head weight and the layer weights in host memory:
    the same loss as with everything on the GPU (up to the summation order of the logits GEMM),
    LoRA gradients no further from the HF + PEFT ones than the GPU step's, and every weight back
    on the GPU, unchanged, after restore_weights()."""
    from solution.engine import FusedQwen2
    from solution.patch import _embedding_to_fp16

    model = _tiny_model(tmp_path, dropout=0.0)
    model.train()
    assert _embedding_to_fp16(model.get_base_model())  # as install() does
    ids, mask, labels = _batch(5, [17, 60, 33, 5])
    n_items = (labels[:, 1:] != -100).sum()
    params = [p for p in model.parameters() if p.requires_grad]
    embed = model.get_base_model().model.embed_tokens.weight
    before = embed.detach().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask, labels=labels,
                    num_items_in_batch=n_items)
    ref = torch.autograd.grad(out.loss.float(), params)

    def run(offload):
        engine = FusedQwen2(model, seed=7)
        if offload:
            engine.offload_weights()
            assert engine.offload_head()
            assert embed.device.type == "cpu" and embed.is_pinned()
        loss, packed, stats = engine.step(ids, mask, labels, float(n_items))
        grads = torch.autograd.grad(loss, params)
        if offload:
            engine.restore_weights()
        return loss, stats, grads

    def err(grads):
        return max(((g.float() - r.float()).norm() / r.float().norm()).item()
                   for g, r in zip(grads, ref))

    loss_a, sa, grads_a = run(False)
    loss_b, sb, grads_b = run(True)
    torch.testing.assert_close(loss_a, loss_b, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(sa.entropy, sb.entropy, rtol=1e-5, atol=1e-5)
    assert err(grads_b) < 2e-2 and err(grads_b) <= 1.5 * err(grads_a), (err(grads_a),
                                                                       err(grads_b))
    assert embed.is_cuda and torch.equal(embed.detach(), before)
    assert model.get_base_model().lm_head.weight is embed


def test_host_head_after_training_forward(tmp_path):
    """After finish_training the embedding waits in host memory; a later forward call brings it
    (and the layer weights) back to the GPU and gives the same logits."""
    from solution.engine import FusedQwen2

    from solution.patch import _embedding_to_fp16

    model = _tiny_model(tmp_path, dropout=0.0)
    assert _embedding_to_fp16(model.get_base_model())
    ids = torch.randint(0, 3000, (2, 16), device=DEV)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        before = model(input_ids=ids).logits.float()
    engine = FusedQwen2(model)
    engine.offload_weights()
    assert engine.offload_head()
    engine.finish_training()
    embed = model.get_base_model().model.embed_tokens.weight
    assert embed.device.type == "cpu"
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        after = model(input_ids=ids).logits.float()
    assert embed.is_cuda
    assert torch.equal(before, after)
