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
  answer #622);
- PyTorch gives every cuBLAS handle a workspace of 4 MiB x 2 + 16 KiB x 8 by default
  (CUBLAS_WORKSPACE_CONFIG unset); with `:16:8` (the setting NVIDIA documents for deterministic
  cuBLAS) a T4 run peaks 20 MiB lower in nvidia-smi at the same step time. cuBLAS then picks other
  GEMM algorithms for some shapes, so the products round differently in the last bits (the loss
  stays within the checks of the base, as with any GEMM). It must be set before the first cuBLAS
  call of the process, which in a `soup train` run comes with the first training step (cuBLASLt
  gets the same 128 KiB: CUBLASLT_WORKSPACE_SIZE, else it warns and is cut to it anyway). Allowed
  by the organizers (t.me/souptasterschat/2880).
"""

import contextlib
import importlib
import os
import sys


def _log(message):
    print(f"[solution] {message}", file=sys.stderr, flush=True)


# CUBLAS_WORKSPACE_CONFIG for this process unless the user set one; None: PyTorch's default
CUBLAS_WORKSPACE = ":16:8"


def small_cublas_workspace():
    if CUBLAS_WORKSPACE is None or "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    # cuBLASLt shares that workspace and would otherwise ask for 1 MiB, warn and get 128 KiB anyway
    size, count = (int(x) for x in CUBLAS_WORKSPACE.strip(":").split(":"))
    os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", str(size * count))  # KiB


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


# At the end of a run, once every exit handler has run (the files they write are written), the
# process leaves without the interpreter's own teardown (module and object finalization, the
# CUDA context and the pinned host memory torn down object by object, ~2 s): os._exit with the
# exit code the program asked for (organizers' answers t.me/souptasterschat/5199, 5190 p.2,
# 5202). Only on a normal end through sys.exit with an int (or None) code and no uncaught
# exception; anything else exits as usual.
FAST_EXIT = True


def exit_without_teardown():
    import atexit
    import threading

    if not FAST_EXIT or getattr(sys.exit, "_soup_kernels_patched", False):
        return
    state = {"code": None, "error": False}
    original_exit, original_hook = sys.exit, sys.excepthook
    original_thread_hook = threading.excepthook

    def exit(code=None):
        state["code"] = 0 if code is None else code
        original_exit(code)

    def excepthook(*args):
        state["error"] = True
        original_hook(*args)

    def thread_excepthook(args):
        state["error"] = True
        original_thread_hook(args)

    def finish():
        code = state["code"]
        if state["error"] or not isinstance(code, int) or isinstance(code, bool):
            return
        atexit.unregister(finish)
        atexit._run_exitfuncs()  # every other exit handler, as the interpreter would run them
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()  # nothing still runs on the GPU
        except Exception:  # noqa: BLE001 - then the usual teardown, which reports it
            return
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        os._exit(code)

    exit._soup_kernels_patched = True
    sys.exit = exit
    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook
    atexit.register(finish)
