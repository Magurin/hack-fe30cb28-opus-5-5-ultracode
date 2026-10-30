"""Less import work before training, on the paths `soup train` takes anyway:

- bitsandbytes wraps a few CPU fallback ops in `torch.compile` when it is imported; that imports
  TorchInductor's compiler (~150 modules, ~1.2 s on a Kaggle CPU), which this run never uses (on
  CUDA every bitsandbytes op has its own kernel). We import bitsandbytes with
  `TORCHDYNAMO_DISABLE=1` set for exactly that import and restore the variable right after:
  `torch.compile` then returns the functions as they are, and later `torch.compile` calls behave
  as usual (the variable is read at call time, nothing caches it). The organizers allowed this
  (Telegram Q&A #834);
- transformers imports scikit-learn in two modules only for features a training run does not use
  (assisted-generation threshold tuning, GLUE metrics), and trl imports wandb in three modules at
  import time even when nothing reports to it (Soup reports to wandb only with `--wandb` /
  `--tracker`). While exactly those modules are imported, the availability check they bind
  reports the library as missing; the check is restored afterwards for everyone else. Nothing is
  stubbed: the libraries are simply not imported, as on a machine without them (organizers'
  answer #622).
"""

import contextlib
import importlib
import os
import sys


def _log(message):
    print(f"[solution] {message}", file=sys.stderr, flush=True)


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


def _unavailable():
    return False


@contextlib.contextmanager
def _reported_missing(checks):
    """Replace the availability functions `checks` ((module name, attribute) pairs) by one that
    returns False, and put the original attributes back afterwards."""
    saved = []
    try:
        for module_name, attr in checks:
            module = importlib.import_module(module_name)
            had = attr in vars(module)
            saved.append((module, attr, had, vars(module).get(attr)))
            setattr(module, attr, _unavailable)
        yield
    finally:
        for module, attr, had, value in reversed(saved):
            if had:
                setattr(module, attr, value)
            else:
                delattr(module, attr)


def _import_without(library, checks, modules):
    if library in sys.modules or any(name in sys.modules for name in modules):
        return
    try:
        with _reported_missing(checks):
            for name in modules:
                importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - the modules are imported again as usual later
        _log(f"importing without {library} failed ({exc.__class__.__name__}: {exc})")


def import_without_sklearn():
    _import_without("sklearn", [("transformers.utils", "is_sklearn_available")],
                    ["transformers.generation.candidate_generator", "transformers.data.metrics"])


def _reports_to_trackers(argv):
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv
               for flag in ("--wandb", "--tracker"))


def import_without_wandb(argv):
    """Only when the command line asks Soup for no tracker (report_to="none")."""
    if _reports_to_trackers(argv):
        return
    _import_without("wandb", [("transformers", "is_wandb_available"),
                              ("transformers.integrations", "is_wandb_available"),
                              ("accelerate.utils", "is_wandb_available")],
                    ["trl.extras.profiling", "trl.trainer.callbacks", "trl.trainer.base_trainer"])
