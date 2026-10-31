"""Our masks in the format of dev/radical/capture_base_masks.py, for its --compare:
python dev/mask_probe/to_radical.py solution.pkl sol.pt [p]

From the solution side of the mask probe (sitecustomize.py): the (seed, counter) of every
TorchDropout.draw, in order, and from them the mask our kernel (torch_dropout_keep_kernel) draws
over the whole padded [B, T, H] LoRA input, packed as capture_base_masks.pack(dropped); and the
generator state (seed, offset) at the end of each step.
"""

import os
import pickle
import sys

import torch

sys.path.insert(0, os.getcwd())
from solution import kernels as K  # noqa: E402

CALLS = 56  # 28 layers x (q, v)


def pack(dropped):  # capture_base_masks.pack
    flat = dropped.reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=flat.device)
    return (flat.view(-1, 8) * weights).sum(dim=1).to(torch.uint8).cpu()


def main(src, dst, p=0.05):
    recs = pickle.load(open(src, "rb"))
    draws = [r for r in recs if r["kind"] == "draw"]
    shapes = {}
    for r in recs:
        if r["kind"] == "ours":
            shapes[r["ctr"]] = (r["rows"], r["T"], r["H"])
    masks, steps = [], []
    for i, d in enumerate(draws):
        step, call = divmod(i, CALLS)
        B, T, H = shapes.get(d["ctr"]) or shapes[draws[step * CALLS]["ctr"]]
        assert B * T * H == d["nelem"]
        out = torch.empty(d["nelem"], device="cuda", dtype=torch.int8)
        K.torch_dropout_keep_kernel[((d["nelem"] + 1023) // 1024,)](
            out, d["nelem"], d["seed"], d["ctr"], d["threads"], 1.0 - p, VEC=d["vec"], BLOCK=1024)
        name = (f"base_model.model.model.layers.{call // 2}.self_attn."
                f"{'qv'[call % 2]}_proj.lora_dropout.default")
        masks.append({"step": step, "name": name, "shape": (B, T, H), "bits": pack(out == 0)})
        blocks = d["threads"] // 256
        increment = ((d["nelem"] - 1) // (256 * blocks * 4) + 1) * 4
        if call == CALLS - 1:
            steps.append({"step": step, "end": (d["seed"], d["ctr"] * 4 + increment)})
    torch.save({"masks": masks, "steps": steps}, dst)
    print(f"{len(masks)} masks, steps {[s['end'] for s in steps]} -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(float, sys.argv[3:4]))
