"""Soup Kernel Challenge solution: a fused Qwen2 + NF4 + LoRA training step on our Triton kernels.

`apply()` patches Soup's `SFTTrainerWrapper.train`: right before training starts, the trainer's
`compute_loss` is replaced by the fused step of `solution/engine.py` (kernels in
`solution/kernels.py`). See README.md for what changes and why.
"""


def apply():
    from .patch import patch_soup

    patch_soup()
