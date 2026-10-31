"""Soup Kernel Challenge solution: a fused Qwen2 + NF4 + LoRA training step on our Triton kernels.

`apply()` trims imports Soup does not need (solution/startup.py), starts compiling our kernels in
the background (solution/warmup.py) and patches Soup, peft and the model loader
(solution/patch.py): right before training starts, the trainer's `compute_loss` is replaced by
the fused step of `solution/engine.py` (kernels in `solution/kernels.py`). See README.md for what
changes and why.
"""


def apply():
    import sys

    from . import startup

    startup.small_cublas_workspace()  # before the first cuBLAS call of the process
    startup.import_bitsandbytes()  # before peft imports it, see solution/startup.py
    startup.import_without_sklearn()
    try:  # compile our kernels in the background from now on, while Soup imports and sets up
        from . import warmup

        warmup.start_from_command_line(sys.argv)
    except Exception as exc:  # noqa: BLE001 - a missed warm-up only costs time
        print(f"[solution] early kernel warm-up not started ({exc})", file=sys.stderr, flush=True)
    from .patch import patch_soup

    patch_soup()
