"""Soup Kernel Challenge solution: a fused Qwen2 + NF4 + LoRA training step on our Triton kernels.

`apply()` patches Soup's `SFTTrainerWrapper.train`: right before training starts, the trainer's
`compute_loss` is replaced by the fused step of `solution/engine.py` (kernels in
`solution/kernels.py`). See README.md for what changes and why.
"""


def apply():
    import sys

    from . import startup

    startup.import_bitsandbytes()  # before peft imports it, see solution/startup.py
    startup.import_without_sklearn()
    from .patch import patch_soup

    try:  # compile our kernels in the background from now on, while Soup sets up
        from . import warmup

        warmup.start_from_command_line(sys.argv)
    except Exception as exc:  # noqa: BLE001 - a missed warm-up only costs time
        print(f"[solution] early kernel warm-up not started ({exc})", file=sys.stderr)
    patch_soup()
