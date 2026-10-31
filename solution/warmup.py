"""Compile our kernels while Soup sets up and loads the model, not during the first step.

Triton compiles a kernel the first time it is launched with a new signature, and on a Kaggle CPU
that is about a second per variant; the measured time includes it. From `apply()` on, background
threads compile every kernel variant the step uses through the very helpers the step calls (so
the signatures, and therefore Triton's cache keys, are the same) - compile only: nothing is
launched, so no CUDA module is loaded onto the GPU while the model loads (module code is device
memory outside PyTorch's allocator and would add to the run's peak). The first real launch in
the training step loads each module in a few milliseconds. A variant the warm-up missed is
compiled on first use as usual.
"""

import contextlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

# One compile thread: on Kaggle's 2 cores x 2 threads, every further one slows Soup's model load
# more than it shortens the compilation (3 threads: 0.8-1.3 s more outside the training loop).
COMPILE_THREADS = 1
_thread = None
_timing = {}


class _Dims:
    pass


def _lower_priority():
    """Run this thread (and the ptxas / gcc processes it starts, which inherit its priority) at
    the lowest CPU priority: the compilation then takes the CPU time the main thread leaves idle
    instead of slowing down Soup's model load (4 vCPUs on Kaggle are 2 physical cores). On Linux a
    thread's nice value is its own; elsewhere this does nothing."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    except (AttributeError, OSError):
        pass


@contextlib.contextmanager
def _unoptimized_c_builds():
    """Build Triton's small C modules (the kernel launchers, the driver helpers) with -O0 while
    the warm-up compiles, through Triton's `knobs.build.impl` hook: gcc takes the last -O flag,
    and -O0 makes the warm-up ~1.5 s shorter on a Kaggle CPU (the first step waits for it less
    often). A launcher only unpacks the arguments and calls cuLaunchKernelEx, so the training
    loop does not get measurably slower. A hook someone else installed is left alone."""
    from triton import knobs
    from triton.runtime import build

    def impl(name, src, srcdir, library_dirs, include_dirs, libraries):
        knobs.build.impl = None  # Triton's own build below, not this hook again
        try:
            return build._build(name, src, srcdir, library_dirs, include_dirs, libraries,
                                ["-O0"])
        except TypeError:  # another Triton version: its own build as it is
            return build._build(name, src, srcdir, library_dirs, include_dirs, libraries)
        finally:
            knobs.build.impl = impl

    if knobs.build.impl is not None:
        yield
        return
    knobs.build.impl = impl
    try:
        yield
    finally:
        knobs.build.impl = None


def _run(dims, dropout_p):
    from . import engine as E
    from . import kernels as K

    dev = torch.device("cuda", torch.cuda.current_device())
    spec = _Dims()
    spec.H, spec.I, spec.HQ, spec.HKV, spec.D = dims["H"], dims["I"], dims["HQ"], dims["HKV"], \
        dims["D"]
    spec.eps = dims["eps"]
    H, I, HQ, HKV, D, V = spec.H, spec.I, spec.HQ, spec.HKV, spec.D, dims["V"]
    packed = _Dims()
    packed.dropout_p = dropout_p
    # Only torch.empty below: the warm-up compiles, it launches nothing, so no CUDA module (ours
    # or PyTorch's) is loaded onto the GPU while the model is being loaded - module code is
    # device memory outside PyTorch's allocator and would add to the run's nvidia-smi peak.
    empty = lambda *shape, dtype=torch.float32: torch.empty(shape, device=dev, dtype=dtype)  # noqa: E731
    packed.cos, packed.sin = empty(1, D // 2), empty(1, D // 2)
    packed.pos = empty(1, dtype=torch.int32)
    f16 = torch.float16
    x, w, rstd = empty(1, H), empty(H), empty(1)
    h16 = empty(1, H, dtype=f16)
    seeds = (3, 5)
    qkv = empty(1, (HQ + 2 * HKV) * D, dtype=f16)
    grads = empty(1, HQ, D, dtype=f16)

    def rms_forward():  # input norm with both LoRA dropouts, post-attention norm, final norm
        E._rms_fwd(spec, x, w, packed, seeds=seeds, drop_q=True, drop_v=True)
        E._rms_fwd(spec, x, w, packed, delta=h16)
        E._rms_fwd(spec, x, w, packed)

    def rms_backward():
        E._rms_bwd(spec, h16, x, rstd, w, packed, dres=x)
        E._rms_bwd(spec, h16, x, rstd, w, packed, dres=x, tq=h16, tv=h16, seeds=seeds)
        E._rms_bwd(spec, h16, x, rstd, w, packed, grad_scale=0.5, out=empty(1, H))

    def glu():
        gu = empty(1, 2 * I, dtype=f16)
        E._glu_bwd(E._glu_fwd(gu, I), gu, I)

    def ce():
        E._ce(empty(1, V, dtype=f16), empty(1, dtype=torch.long), empty(1), empty(1),
              empty(1, dtype=torch.int32))

    def ce_pieces():  # the lm_head streamed from host memory (engine.HostHead)
        E._ce_piece(empty(1, 8192, dtype=f16), empty(8192, H, dtype=f16),
                    empty(1, dtype=torch.long), empty(1, 4), empty(1, dtype=torch.int32),
                    empty(1), empty(1), empty(1, H, dtype=f16), 8192, 5)
        E._ce_finish(empty(1, 4), empty(1, dtype=torch.int32), empty(1),
                     empty(1, dtype=torch.long), empty(1), empty(1), empty(1, dtype=torch.int32),
                     empty(1, H), empty(1, H, dtype=f16), empty(1, H, dtype=f16), 5)

    def nf4():  # the argument types NF4Weight.dequant_into passes for double-quantized weights
        E._launch(K.nf4_dequant_kernel, (1,), empty(4096, dtype=torch.uint8),
                  empty(128, dtype=torch.uint8), empty(256), empty(1), empty(1), empty(16),
                  empty(64, 128, dtype=f16), 128, NESTED=True, STANDARD_NF4=True, BS1=64,
                  BS2=256, ROWS=E.NF4_ROWS, num_warps=4)

    def compile_only(job):
        E._COMPILE_ONLY.kernels = []
        try:
            job()
            # build the C launchers too (CPU work, cached on disk): the first real launch then
            # only loads the module onto the GPU
            from triton.runtime.driver import driver

            for compiled in E._COMPILE_ONLY.kernels:
                driver.active.launcher_cls(compiled.src, compiled.metadata)
        finally:
            E._COMPILE_ONLY.kernels = None

    # One task per @triton.jit function (its variants one after the other), so with several
    # threads no two compile into the same kernel's cache.
    jobs = [rms_forward, rms_backward, glu, ce, ce_pieces, nf4,
            lambda: E._rope_fwd(spec, qkv, empty(1, HQ * D, dtype=f16),
                                empty(1, HKV * D, dtype=f16), packed, 2.0),
            lambda: E._rope_bwd(spec, grads, grads, grads, packed)]
    with (_unoptimized_c_builds(),
          ThreadPoolExecutor(max_workers=COMPILE_THREADS, thread_name_prefix="solution-compile",
                             initializer=_lower_priority) as pool):
        for future in [pool.submit(compile_only, job) for job in jobs]:
            future.result()

    # Pin the host memory the NF4 weights will be streamed from (WeightStream) now: pinning
    # ~650 MB costs ~0.15 s, and PyTorch's caching host allocator hands the freed block (rounded
    # up to a power of two) to the same-sized request at the first training step.
    values = dims["L"] * H * (HQ * D + 2 * HKV * D + HQ * D + 3 * I)
    pinned = torch.empty((values // 2 + values // 64,), dtype=torch.uint8, pin_memory=True)
    del pinned
    # ... and the layer inputs' (engine.InputStream). The embedding / lm_head weight needs none:
    # engine.HostHead pins it where the loader left it.
    pinned = torch.empty((dims["L"], E.InputStream.HOST_ROWS, H), dtype=torch.float32,
                         pin_memory=True)
    del pinned


def fp16_path_gpu():
    """The GPUs the fused step is for: Soup's fp16 path (no bf16, so Soup trains in fp16, as on a
    T4) on sm_70 or newer (Triton's minimum). On a bf16 GPU Soup trains in bf16, and the fused step,
    which computes in fp16, would change the computation dtype, so it stays off there."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() < (7, 0):
        return False
    try:
        from soup_cli.utils.gpu import cuda_supports_bf16
        return not cuda_supports_bf16()
    except Exception:  # noqa: BLE001 - same rule as Soup's own check
        return torch.cuda.get_device_capability() < (8, 0)


def start(base_model):
    """Start compiling for `base_model` (a Hugging Face id or path) in the background."""
    global _thread
    if _thread is not None or not fp16_path_gpu():
        return
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_model)
    if getattr(config, "model_type", None) != "qwen2":
        return
    dims = {"H": config.hidden_size, "I": config.intermediate_size, "V": config.vocab_size,
            "HQ": config.num_attention_heads, "HKV": config.num_key_value_heads,
            "D": getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads,
            "eps": float(config.rms_norm_eps), "L": config.num_hidden_layers}

    def work():
        _lower_priority()
        started = time.perf_counter()
        try:
            _run(dims, 0.05)
        except Exception as exc:  # noqa: BLE001 - a missed warm-up only costs time
            print(f"[solution] kernel warm-up skipped ({exc.__class__.__name__}: {exc})",
                  file=sys.stderr, flush=True)
        _timing["compile_s"] = time.perf_counter() - started

    _thread = threading.Thread(target=work, name="solution-kernel-warmup", daemon=True)
    _thread.start()


def start_from_command_line(argv):
    """Start as early as possible: `soup train --config <file>` names the base model in its
    config, which is all the warm-up needs (Soup reads the same file a moment later)."""
    if "--config" not in argv or argv.index("--config") + 1 >= len(argv):
        return
    import yaml

    with open(argv[argv.index("--config") + 1], encoding="utf-8") as handle:
        base = (yaml.safe_load(handle) or {}).get("base")
    if isinstance(base, str):
        start(base)


def wait(timeout=300):
    """Join the warm-up thread; returns (seconds waited here, seconds the warm-up took)."""
    started = time.perf_counter()
    if _thread is not None:
        _thread.join(timeout)
    return time.perf_counter() - started, _timing.get("compile_s")
