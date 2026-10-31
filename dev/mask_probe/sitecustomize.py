"""Mask probe (dev only, never part of solution/): with MASK_PROBE_OUT set, records every LoRA
dropout of the first MASK_PROBE_STEPS micro-batches of a `soup train` run and exits.

Base side: nn.Dropout.forward is wrapped - shape, dtype, alignment, the CUDA generator's seed and
offset before / after the call, and the keep mask (out != 0, packed bits).
Solution side (solution.engine imported): TorchDropout.draw and _rms_fwd are wrapped - the
(seed, counter) of every draw and the keep mask of the dropout'ed LoRA inputs our kernel wrote
(yq / yv != 0 where y != 0), on the packed tokens with their padded index.

Used as: PYTHONPATH=dev/mask_probe MASK_PROBE_OUT=/path python harness/run_task.py ...
"""

import importlib.abc
import importlib.util
import os
import pickle
import sys

OUT = os.environ.get("MASK_PROBE_OUT")
STEPS = int(os.environ.get("MASK_PROBE_STEPS", "3"))
CALLS_PER_STEP = int(os.environ.get("MASK_PROBE_CALLS", "56"))  # 28 layers x (q, v)
# MASK_PROBE_MASKS < MASK_PROBE_STEPS: masks for that many steps, then the generator state only
MASKS = int(os.environ.get("MASK_PROBE_MASKS", str(STEPS)))

records = []
seen = set()


def _done():
    path = f"{OUT}.{os.getpid()}.pkl"
    with open(path, "wb") as f:
        pickle.dump(records, f)
    print(f"[mask_probe] {len(records)} records -> {path}", flush=True)
    os._exit(0)


def _gen():
    import torch
    return torch.cuda.default_generators[torch.cuda.current_device()]


def _install_base():
    import numpy as np
    import torch

    orig = torch.nn.Dropout.forward

    def forward(self, x):
        if not (self.training and self.p > 0) or not x.is_cuda:
            return orig(self, x)
        g = _gen()
        off0 = g.get_offset()
        if off0 not in seen and len(seen) >= STEPS * CALLS_PER_STEP:
            _done()
        if len(seen) >= MASKS * CALLS_PER_STEP:
            seen.add(off0)
        out = orig(self, x)
        off1 = g.get_offset()
        rec = dict(kind="base", id=id(self), shape=tuple(x.shape), dtype=str(x.dtype),
                   contig=x.is_contiguous(), align=x.data_ptr() % 64, seed=g.initial_seed(),
                   off0=off0, off1=off1, grad=torch.is_grad_enabled(), p=self.p)
        if off0 not in seen and len(seen) < MASKS * CALLS_PER_STEP:
            seen.add(off0)
            rec["zeros_in"] = int((x == 0).sum())
            rec["xnz"] = np.packbits((x != 0).reshape(-1).cpu().numpy())
            rec["mask"] = np.packbits((out != 0).reshape(-1).cpu().numpy())
        records.append(rec)
        return out

    torch.nn.Dropout.forward = forward


def _patch_engine(E):
    import numpy as np
    import torch

    draw = E.TorchDropout.draw

    def draw_rec(self, nelem):
        seed, ctr, threads = draw(self, nelem)
        records.append(dict(kind="draw", seed=seed, ctr=ctr, threads=threads, nelem=nelem,
                            vec=self.vec))
        return seed, ctr, threads

    E.TorchDropout.draw = draw_rec
    rms_fwd = E._rms_fwd

    def rms_fwd_rec(spec, x, weight, packed, **kw):
        res = rms_fwd(spec, x, weight, packed, **kw)
        if getattr(spec, "index", None) is None:  # the warm-up's and the self-check's calls
            return res
        seeds = kw.get("seeds", (0, 0))
        for which, out, s in (("q", res[2], seeds[0]), ("v", res[3], seeds[1])):
            if out is None or s in seen:
                continue
            if len(seen) >= STEPS * CALLS_PER_STEP:
                _done()
            seen.add(s)
            if len(seen) > MASKS * CALLS_PER_STEP:
                continue
            y = res[1]
            records.append(dict(kind="ours", which=which, layer=spec.index, ctr=s,
                                rows=packed.rows, T=packed.T, H=x.shape[1],
                                tok=packed.tok.cpu().numpy(),
                                ynz=np.packbits((y != 0).reshape(-1).cpu().numpy()),
                                mask=np.packbits((out != 0).reshape(-1).cpu().numpy())))
        return res

    E._rms_fwd = rms_fwd_rec


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "solution.engine":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch_engine(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if OUT and "_train.py" in " ".join(sys.argv):
    _install_base()  # on the solution side it catches the fallbacks to Soup's own code
    if "--solution" in sys.argv:
        sys.meta_path.insert(0, _Finder())
