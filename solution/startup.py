"""Less import work before training, on the paths `soup train` takes anyway:

- bitsandbytes wraps a few CPU fallback ops in `torch.compile` when it is imported; that imports
  TorchInductor's compiler (~150 modules, ~1.2 s on a Kaggle CPU), which this run never uses (on
  CUDA every bitsandbytes op has its own kernel). We import bitsandbytes with
  `TORCHDYNAMO_DISABLE=1` set for exactly that import and restore the variable right after:
  `torch.compile` then returns the functions as they are, and later `torch.compile` calls behave
  as usual (the variable is read at call time, nothing caches it). The organizers allowed this
  (Telegram Q&A #834).
"""

import os
import sys


def import_bitsandbytes():
    if "bitsandbytes" in sys.modules:
        return
    previous = os.environ.get("TORCHDYNAMO_DISABLE")
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    try:
        import bitsandbytes  # noqa: F401
    except Exception:  # noqa: BLE001 - whoever needs it imports it again and sees the error
        pass
    finally:
        if previous is None:
            os.environ.pop("TORCHDYNAMO_DISABLE", None)
        else:
            os.environ["TORCHDYNAMO_DISABLE"] = previous
