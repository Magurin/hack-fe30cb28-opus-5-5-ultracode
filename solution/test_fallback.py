"""The patched compute_loss falls back to the original code (eval, extra input keys, left
padding) without crashing, also after fused steps moved the 4-bit weights to host memory
(written by the testing stream)."""
import types

import pytest
import torch

from solution import patch as P
from solution.test_kernels import _tiny_model

DEV = "cuda"


def _trainer(model):
    from solution.engine import FusedQwen2

    def original(model, inputs, return_outputs=False, num_items_in_batch=None):
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**inputs)
        return out.loss

    t = types.SimpleNamespace()
    t._soup_kernels_engine = FusedQwen2(model, seed=0)
    t._soup_original_compute_loss = original
    t.compute_loss_func = None
    t.label_smoother = None
    t.model_accepts_loss_kwargs = True
    t.args = types.SimpleNamespace(average_tokens_across_devices=False, n_gpu=1)
    t.state = types.SimpleNamespace(global_step=0)
    t._metrics = {"train": {"entropy": [], "mean_token_accuracy": [], "num_tokens": []}}
    t._total_train_tokens = 0
    t._soup_rewind_state = None
    return t


def _batch(left=False):
    torch.manual_seed(3)
    lengths = [20, 31, 9, 31]
    B, T = len(lengths), max(lengths)
    ids = torch.randint(0, 3000, (B, T), device=DEV)
    ar = torch.arange(T, device=DEV)[None]
    L = torch.tensor(lengths, device=DEV)[:, None]
    mask = ((ar >= T - L) if left else (ar < L)).long()
    labels = torch.where(mask.bool(), ids, -100)
    labels[:, :2] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def _call(t, model, inputs):
    n = (inputs["labels"][:, 1:] != -100).sum()
    return P.fused_compute_loss(t, model, dict(inputs), num_items_in_batch=n)


def test_eval_after_training_step(tmp_path):
    model = _tiny_model(tmp_path, dropout=0.05)
    t = _trainer(model)
    model.train()
    _call(t, model, _batch()).backward()
    P._flush_metrics(t)
    model.eval()
    try:
        with torch.no_grad():
            loss = _call(t, model, _batch())
        print("OUTCOME eval fallback ok, loss", float(loss))
    except Exception as exc:  # noqa: BLE001
        print("OUTCOME eval fallback CRASH", type(exc).__name__, exc)
        raise


def test_extra_key_after_training_step(tmp_path):
    model = _tiny_model(tmp_path, dropout=0.05)
    t = _trainer(model)
    model.train()
    _call(t, model, _batch()).backward()
    P._flush_metrics(t)
    inputs = _batch()
    inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[1], device=DEV)[None].expand(4, -1)
    try:
        loss = _call(t, model, inputs)
        loss.backward()
        print("OUTCOME extra-key fallback ok, loss", float(loss))
    except Exception as exc:  # noqa: BLE001
        print("OUTCOME extra-key fallback CRASH", type(exc).__name__, exc)
        raise


def test_left_padding(tmp_path):
    model = _tiny_model(tmp_path, dropout=0.05)
    t = _trainer(model)
    model.train()
    try:
        loss = _call(t, model, _batch(left=True))
        loss.backward()
        print("OUTCOME left padding ok, loss", float(loss))
    except Exception as exc:  # noqa: BLE001
        print("OUTCOME left padding CRASH", type(exc).__name__, exc)
        raise


def test_train_eval_train_matches(tmp_path):
    """Fused step, eval fallback, fused step again: the second fused step re-offloads and gives
    the same loss as a fused step on a fresh engine with the same weights and seed."""
    model = _tiny_model(tmp_path, dropout=0.0)
    t = _trainer(model)
    model.train()
    first = _call(t, model, _batch())
    first.backward()
    P._flush_metrics(t)
    model.eval()
    with torch.no_grad():
        _call(t, model, _batch())
    model.train()
    again = _call(t, model, _batch())
    again.backward()
    P._flush_metrics(t)
    print("OUTCOME train-eval-train", float(first), float(again))
    assert torch.equal(first.detach(), again.detach())
