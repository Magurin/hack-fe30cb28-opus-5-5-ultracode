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
    startup.exit_without_teardown()  # at the end: every exit handler, then os._exit (startup.py)
    startup.import_bitsandbytes()  # before peft imports it, see solution/startup.py
    startup.import_without_sklearn()
    from .patch import patch_soup

    patch_soup()
    # compile our kernels in the background from here on, while Soup sets up and loads the model:
    # started earlier, its Python work held the GIL against the imports above (Kaggle: apply()
    # ~0.5 s longer), and it still ends before the first step needs it
    try:
        from . import warmup

        warmup.start_from_command_line(sys.argv)
    except Exception as exc:  # noqa: BLE001 - a missed warm-up only costs time
        print(f"[solution] early kernel warm-up not started ({exc})", file=sys.stderr, flush=True)
