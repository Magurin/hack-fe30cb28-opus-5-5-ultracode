"""Compares the LoRA dropout masks of a base and a solution run of the mask probe (sitecustomize):
python dev/mask_probe/compare.py base.pkl solution.pkl

For every dropout call of the first steps, in order: the same generator seed and offset, the
same [B, T, H] tensor, and on every real token the same keep mask, bit for bit (positions where
our dropout input y is exactly 0 are left out: the base's out != 0 cannot tell a kept 0 there).
"""

import pickle
import sys

import numpy as np


def main(base_path, sol_path):
    base = [r for r in pickle.load(open(base_path, "rb")) if "mask" in r]
    sol = pickle.load(open(sol_path, "rb"))
    draws = [r for r in sol if r["kind"] == "draw"]
    ours = [r for r in sol if r["kind"] == "ours"]
    fallback = [r for r in sol if r["kind"] == "base"]
    n = min(len(base), len(draws), len(ours))
    print(f"base calls {len(base)}, our draws {len(draws)}, our masks {len(ours)}, "
          f"fallback dropout calls {len(fallback)}; comparing {n}")
    bad = 0
    for i in range(n):
        b, d, o = base[i], draws[i], ours[i]
        B, T, H = b["shape"]
        state_ok = (d["seed"] == b["seed"] and d["ctr"] * 4 == b["off0"]
                    and d["nelem"] == B * T * H and o["ctr"] == d["ctr"]
                    and (o["rows"], o["T"], o["H"]) == (B, T, H))
        bm = np.unpackbits(b["mask"])[:B * T * H].reshape(B * T, H)[o["tok"]]
        k = len(o["tok"]) * H
        om = np.unpackbits(o["mask"])[:k].reshape(-1, H).astype(bool)
        valid = np.unpackbits(o["ynz"])[:k].reshape(-1, H).astype(bool)
        if "xnz" in b:  # and where the base's input is exactly 0
            valid &= np.unpackbits(b["xnz"])[:B * T * H].reshape(B * T, H)[o["tok"]].astype(bool)
        else:
            d10 = int(((bm.astype(bool) & ~om) & valid).sum())
            d01 = int(((~bm.astype(bool) & om) & valid).sum())
            if d10 or d01:
                print(f"  call {i}: base 1 / ours 0: {d10}, base 0 / ours 1: {d01}, base zero "
                      f"inputs {b['zeros_in']}")
        diff = int(((bm.astype(bool) != om) & valid).sum())
        if not state_ok or diff:
            bad += 1
            print(f"call {i} (layer {o['layer']} {o['which']}): state_ok={state_ok} "
                  f"seed {d['seed']}/{b['seed']} off {d['ctr'] * 4}/{b['off0']} "
                  f"differing bits {diff} of {int(valid.sum())}")
        elif i < 3 or i == n - 1:
            print(f"call {i} (layer {o['layer']} {o['which']}): seed {d['seed']} offset "
                  f"{b['off0']} [{B}, {T}, {H}], {int(valid.sum())} bits equal, kept "
                  f"{om[valid].mean():.4f}")
    print("ALL MASKS EQUAL" if not bad and n == len(base) else f"{bad} calls differ")


if __name__ == "__main__":
    main(*sys.argv[1:3])
