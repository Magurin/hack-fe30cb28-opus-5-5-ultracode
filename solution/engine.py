"""The training step of Qwen2 + NF4 + LoRA, rebuilt around our kernels.

Instead of running the HF modules (fp32 residual stream and norms, fp32 copies of every
projection output, (B, T, 151936) logits upcast to fp32), one step goes like this:

1. pad tokens are dropped: the rows of the batch are packed into one (N, H) sequence (N = real
   tokens), attention runs varlen over the packed rows, so no work and no memory goes to padding;
2. every decoder layer is one autograd.Function (`_Layer`). Within the step's memory cap
   (`FusedQwen2.keep_plan`) the last layers keep their [gate | up] GEMM output, then every
   intermediate; the others recompute their forward in backward from their input, the fp32
   residual (6 KiB per token and layer). A step too long for the cap runs lean
   (`FusedQwen2.lean`): nothing kept, the MLP over MLP_CHUNK tokens at a time. NF4 weights are
   dequantized by our kernel once in forward and once in backward (as bitsandbytes does); the
   backward reuses them for the recomputation;
3. the final norm, lm_head, cross-entropy and the metrics TRL / Soup log (entropy, token accuracy,
   per-row loss) are one chunked pass (`_Head`): logits exist for a chunk of rows at a time and the
   gradient w.r.t. the hidden state is formed in the same pass;
4. the packed NF4 weights of the layers wait in pinned host memory and are streamed to the GPU
   one layer ahead of the compute (`WeightStream`);
5. the frozen, tied embedding / lm_head weight (445 MiB in fp16) stays in pinned host memory
   (`HostHead`): the step's embedding rows are gathered on the CPU, the lm_head is streamed in
   pieces of the vocabulary while the head runs over them (`_head_streamed`);
6. the layer inputs (`InputStream`) and the [gate | up] outputs of the layers that do not keep
   them (`GuStream`) wait in pinned host memory between forward and backward, copied out and
   back on side streams while the layers compute (`_HostRing`).

The math is the base's: the same fp16 roundings at the same places (projection outputs, attention
inputs, the MLP product), fp32 for the residual, the norms, RoPE, softmax and the loss. The LoRA
dropout masks are the base's own, bit for bit: those PEFT's nn.Dropout would draw from the CUDA
generator, in the same order (`TorchDropout`); our kernels compute them from the generator's seed
and offset, in backward again instead of storing them.
"""

import threading
import warnings

import torch
import triton

from . import kernels as K

NF4_ROWS = 32  # quantization blocks (of 64 values) per program
GLU_BLOCK = 1024  # columns of the MLP output per SwiGLU program
CE_BLOCK = 4096  # vocabulary entries per loop iteration of the cross-entropy kernel
HEAD_CHUNK = 256  # rows of logits alive at a time: 256 x 151936 fp16 = 74 MiB
MLP_CHUNK = 512  # tokens of [gate | up] alive at a time in a lean step (FusedQwen2.lean)


_COMPILE_ONLY = threading.local()  # set by solution/warmup.py in its compile threads


def _launch(kernel, grid, *args, **kwargs):
    """Launch a Triton kernel; in a warm-up compile thread, only compile it (no launch, and no
    module loaded onto the GPU yet) and prepare its launcher."""
    if getattr(_COMPILE_ONLY, "kernels", None) is None:
        return kernel[grid](*args, **kwargs)
    compiled = kernel.warmup(*args, grid=grid, **kwargs)
    _COMPILE_ONLY.kernels.append(compiled)
    return compiled


def _cdiv(a, b):
    return (a + b - 1) // b


def _pow2(n):
    return triton.next_power_of_2(n)


# ---------------------------------------------------------------------------------------------
# NF4 weights
# ---------------------------------------------------------------------------------------------
class NF4Weight:
    """The packed bytes and quant state of one bitsandbytes Linear4bit weight."""

    def __init__(self, param):
        qs = param.quant_state
        self.param = param
        self.packed = param.data  # the device copy the kernel reads (see WeightStream)
        self.group = None  # an NF4Group, while streamed
        self.shape = tuple(int(s) for s in qs.shape)
        self.numel = self.shape[0] * self.shape[1]
        self.code = qs.code.float().contiguous()
        self.bs1 = int(qs.blocksize)
        if self.numel % self.bs1:
            raise ValueError("weight size is not a whole number of quantization blocks")
        levels = torch.tensor(K.NF4_LEVELS, dtype=torch.float32, device=self.code.device)
        self.standard = bool(torch.equal(self.code, levels))
        self.nested = bool(qs.nested)
        if self.nested:
            self.absmax = qs.absmax
            self.code2 = qs.state2.code.float().contiguous()
            self.absmax2 = qs.state2.absmax.float().contiguous()
            self.offset = torch.as_tensor(qs.offset, device=self.code.device,
                                          dtype=torch.float32).reshape(1)
            self.bs2 = int(qs.state2.blocksize)
        else:
            self.absmax = qs.absmax.float().contiguous()
            self.code2 = self.absmax2 = self.offset = self.absmax
            self.bs2 = 1

    def dequant_into(self, out):
        blocks = self.numel // self.bs1
        _launch(K.nf4_dequant_kernel, (_cdiv(blocks, NF4_ROWS),),
            self.packed, self.absmax, self.code2, self.absmax2, self.offset, self.code, out,
            blocks, NESTED=self.nested, STANDARD_NF4=self.standard, BS1=self.bs1, BS2=self.bs2,
            ROWS=NF4_ROWS, num_warps=4)
        return out


class NF4Group:
    """Weights dequantized into one buffer and stored back to back in a WeightStream slot: one
    kernel launch for all of them. Only the double-quantization offset differs between them (per
    segment); their absmax2 are concatenated, which needs every weight but the last to end on a
    whole block of absmax2."""

    @classmethod
    def of(cls, weights):
        if not 2 <= len(weights) <= 3:
            return None
        w0 = weights[0]
        for w in weights:
            if (w.shape[1] != w0.shape[1] or w.bs1 != w0.bs1 or w.bs2 != w0.bs2
                    or w.nested != w0.nested or w.standard != w0.standard
                    or not torch.equal(w.code, w0.code)
                    or (w.nested and not torch.equal(w.code2, w0.code2))):
                return None
        if any((w.numel // w.bs1) % w.bs2 for w in weights[:-1]):
            return None
        return cls(weights)

    def __init__(self, weights):
        w0 = weights[0]
        self.weights = weights
        self.nested, self.standard, self.bs1, self.bs2 = w0.nested, w0.standard, w0.bs1, w0.bs2
        self.code, self.code2 = w0.code, w0.code2
        self.blocks = sum(w.numel for w in weights) // self.bs1
        bounds = []
        for w in weights[:-1]:
            bounds.append((bounds[-1] if bounds else 0) + w.numel // self.bs1)
        self.seg1, self.seg2 = (bounds + [2147483647, 2147483647])[:2]
        if self.nested:
            self.absmax2 = torch.cat([w.absmax2 for w in weights])
            self.offsets = torch.cat([w.offset for w in weights])
        self.packed = self.absmax = None  # set while the group's layer sits in a slot

    def dequant_into(self, out):
        if self.nested:
            code2, absmax2, offsets = self.code2, self.absmax2, self.offsets
        else:
            code2 = absmax2 = offsets = self.absmax
        _launch(K.nf4_dequant_kernel, (_cdiv(self.blocks, NF4_ROWS),),
            self.packed, self.absmax, code2, absmax2, offsets, self.code, out, self.blocks,
            NESTED=self.nested, STANDARD_NF4=self.standard, BS1=self.bs1, BS2=self.bs2,
            ROWS=NF4_ROWS, seg1=self.seg1, seg2=self.seg2, num_warps=4)
        return out


class WeightStream:
    """The packed NF4 weights of every decoder layer live in pinned host memory during training;
    a layer's bytes are copied into one of three device slots on a side stream, one layer ahead
    of the compute (forward: i+1, backward: i-1). Slot i % 3 is only overwritten after the
    compute stream has finished with the layer that held it before. 23 MiB per Qwen2.5-1.5B
    layer, so 69 MiB on the GPU instead of ~645 MiB, for ~1.3 GB of PCIe traffic per step that
    runs alongside the GEMMs."""

    SLOTS = 3
    # weights dequantized together into one buffer (one GEMM each): stored back to back, so one
    # kernel launch dequantizes the whole group
    GROUPS = (("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"),
              ("down_proj",))

    def __init__(self, specs, device):
        self.device = device
        # per layer: (start, size, [[weight, packed off, n, absmax off, n]],
        #             [(group, packed off, n, absmax off, n)])
        self.layout = []
        self.groups = []
        offset = 0
        for spec in specs:
            entries, groups, start = [], [], offset
            for names in self.GROUPS:
                ws = [spec.w[name] for name in names]
                pks = [w.packed.reshape(-1).view(torch.uint8).numel() for w in ws]
                ams = [w.absmax.reshape(-1).view(torch.uint8).numel() for w in ws]
                packed_at = offset
                absmax_at = _align(packed_at + sum(pks))
                offset = _align(absmax_at + sum(ams))
                po, ao = packed_at, absmax_at
                for w, pn, an in zip(ws, pks, ams):
                    entries.append([w, po, pn, ao, an])
                    po, ao = po + pn, ao + an
                group = NF4Group.of(ws)
                if group is not None:
                    groups.append((group, packed_at, sum(pks), absmax_at, sum(ams)))
                    self.groups.append(group)
                    for w in ws:
                        w.group = group
            self.layout.append((start, offset - start, entries, groups))
        self.host = torch.empty((offset,), dtype=torch.uint8, pin_memory=True)
        for _, _, entries, _ in self.layout:
            for w, po, pn, ao, an in entries:
                self.host[po:po + pn].copy_(w.packed.reshape(-1).view(torch.uint8))
                self.host[ao:ao + an].copy_(w.absmax.reshape(-1).view(torch.uint8))
        torch.cuda.synchronize(device)
        # Free the device copies (the bitsandbytes parameters point at the host bytes meanwhile),
        # and only then take the three slots, so the peak never holds both.
        self.moved = []
        for _, _, entries, _ in self.layout:
            for w, po, pn, ao, an in entries:
                data, state = w.param.data, w.param.quant_state
                self.moved.append(w)
                w.param.data = self.host[po:po + pn].view(data.dtype).view(data.shape)
                state.absmax = self.host[ao:ao + an].view(state.absmax.dtype).view(
                    state.absmax.shape)
                w.packed = w.absmax = None
        torch.cuda.empty_cache()
        self.slot_bytes = max(layer[1] for layer in self.layout)
        self.slots = [torch.empty((self.slot_bytes,), dtype=torch.uint8, device=device)
                      for _ in range(self.SLOTS)]
        self.slot_layer = [None] * self.SLOTS
        self.copied = [torch.cuda.Event() for _ in range(self.SLOTS)]
        self.released = [torch.cuda.Event() for _ in range(self.SLOTS)]
        self.stream = torch.cuda.Stream(device)

    def _load(self, i):
        slot = i % self.SLOTS
        if self.slot_layer[slot] == i:
            return
        start, size, _, _ = self.layout[i]
        self.stream.wait_event(self.released[slot])
        with torch.cuda.stream(self.stream):
            self.slots[slot][:size].copy_(self.host[start:start + size], non_blocking=True)
            self.copied[slot].record(self.stream)
        self.slot_layer[slot] = i

    def acquire(self, i, ahead):
        """Make layer i's weights usable on the current stream; start fetching layer `ahead`."""
        self._load(i)
        slot = i % self.SLOTS
        torch.cuda.current_stream(self.device).wait_event(self.copied[slot])
        start, _, entries, groups = self.layout[i]
        buf = self.slots[slot]
        for w, po, pn, ao, an in entries:
            w.packed = buf[po - start:po - start + pn]
            absmax = buf[ao - start:ao - start + an]
            w.absmax = absmax if w.nested else absmax.view(torch.float32)
        for group, po, pn, ao, an in groups:
            absmax = buf[ao - start:ao - start + an]
            group.packed = buf[po - start:po - start + pn]
            group.absmax = absmax if group.nested else absmax.view(torch.float32)
        if 0 <= ahead < len(self.layout):
            self._load(ahead)

    def release(self, i):
        self.released[i % self.SLOTS].record(torch.cuda.current_stream(self.device))

    def release_slots(self):
        torch.cuda.synchronize(self.device)
        for group in self.groups:
            group.packed = group.absmax = None
        self.slots = []
        self.slot_layer = [None] * self.SLOTS
        torch.cuda.empty_cache()

    def restore(self):
        """Put the weights back on the device."""
        self.release_slots()
        for w in self.moved:
            state = w.param.quant_state
            w.param.data = w.param.data.to(self.device)
            state.absmax = state.absmax.to(self.device)
            w.packed = w.param.data
            w.absmax = state.absmax if w.nested else state.absmax.float()
        self.moved = []
        torch.cuda.synchronize(self.device)


class _HostRing:
    """What InputStream and GuStream share: a tensor of each layer that waits in pinned host
    memory (`host`, [L, rows, cols]) between the layer's forward and its backward, and a ring of
    SLOTS device buffers it is written into and copied back into (layer i's in slot i % SLOTS).

    A copy out runs on its own stream (a separate copy engine) while the layers compute on; a
    slot is written again only after the compute stream has waited for the copy out of what it
    held (`saved`), and read in backward after the copy back (`fetched`), so nothing needs
    record_stream (which grew the allocator's reservation). The tensors of the last SLOTS layers
    below `upto` are still in the ring when their backward runs; the others are copied back
    LOOKAHEAD layers ahead of the backward that reads them, into a slot whose previous tensor
    belongs to a layer whose backward has already run."""

    SLOTS = 2
    LOOKAHEAD = 1
    HOST_ROWS = 2048  # tokens the host buffer is first made for (warmup.py pins its block early)

    def __init__(self, layers, device):
        self.L, self.device = layers, device
        self.out_stream = torch.cuda.Stream(device)
        self.in_stream = torch.cuda.Stream(device)
        self.host = None  # pinned [L, rows, cols]
        self.ring = None  # [SLOTS, rows, cols] on the device
        self.saved = [torch.cuda.Event() for _ in range(layers)]
        self.fetched = [torch.cuda.Event() for _ in range(layers)]
        self.in_slot = [None] * self.SLOTS  # which layer's tensor each slot holds
        self.n = 0
        self.upto = layers  # the layers below it use the ring

    def _host_rows(self, n, cols, dtype):
        if self.host is None or self.host.shape[1] < n:
            torch.cuda.synchronize(self.device)  # nothing still reads the old buffer
            self.host = None
            # one buffer: PyTorch hands out pinned memory in powers of two
            self.host = torch.empty((self.L, max(n, self.HOST_ROWS), cols), dtype=dtype,
                                    pin_memory=True)

    def _stays(self, i):
        """Whether layer i's tensor is still in the ring when its backward runs."""
        return i >= self.upto - self.SLOTS

    def _slot(self, i):
        """Forward: the slot layer i's tensor is written into."""
        s = i % self.SLOTS
        old = self.in_slot[s]
        if old is not None and not self._stays(old):
            torch.cuda.current_stream(self.device).wait_event(self.saved[old])  # copied out
        self.in_slot[s] = i
        return self.ring[s, :self.n]

    def _copy_out(self, i, t):
        """Forward: copy layer i's tensor t out once the compute stream has written it."""
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        self.out_stream.wait_event(ready)
        with torch.cuda.stream(self.out_stream):
            self.host[i, :self.n].copy_(t, non_blocking=True)
            self.saved[i].record(self.out_stream)

    def fetch(self, i):
        """Backward: start copying layer i's tensor back into its slot."""
        s = i % self.SLOTS
        if i < 0 or self._stays(i) or self.in_slot[s] == i:
            return
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))  # the slot's last reader is done
        self.in_stream.wait_event(ready)
        with torch.cuda.stream(self.in_stream):
            self.ring[s, :self.n].copy_(self.host[i, :self.n], non_blocking=True)
            self.fetched[i].record(self.in_stream)
        self.in_slot[s] = i

    def load(self, i):
        """Backward of layer i: its tensor on the current stream; starts fetching the next
        ones (never into the slot this backward reads: j < SLOTS)."""
        if not self._stays(i):
            self.fetch(i)
            torch.cuda.current_stream(self.device).wait_event(self.fetched[i])
        out = self.ring[i % self.SLOTS, :self.n]
        for j in range(1, min(self.LOOKAHEAD, self.SLOTS - 1) + 1):
            self.fetch(i - j)
        return out


class InputStream(_HostRing):
    """The input of every decoder layer (the fp32 residual, L x H x 4 bytes per token: the bulk
    of what a step keeps for backward) waits in pinned host memory instead of on the GPU.

    Layer i writes its output (the input of layer i + 1) straight into slot (i + 1) % SLOTS, so
    the inputs never come from the caching allocator (freed one by one between larger blocks,
    they fragmented its reservation); its input is copied out while it computes. The first
    layer's input (the embedding output) is not a slot: it is kept until its copy is done."""

    SLOTS = 3  # > LOOKAHEAD: a slot fetched into is never the one the backward reads
    LOOKAHEAD = 2
    ROWS = 256  # the ring grows (rarely) in steps of this many tokens

    def __init__(self, layers, device):
        super().__init__(layers, device)
        self.first = None  # the first layer's input (not a ring slot), until its copy is done

    def begin(self, n, H):
        self._host_rows(n, H, torch.float32)
        if self.ring is None or self.ring.shape[1] < n:
            torch.cuda.synchronize(self.device)
            self.ring = None
            self.ring = torch.empty((self.SLOTS, _cdiv(n, self.ROWS) * self.ROWS, H),
                                    device=self.device, dtype=torch.float32)
        self.n = n
        self.first = None
        self.in_slot = [None] * self.SLOTS

    def save(self, i, h):
        """Forward of layer i, before its compute: copy its input out (unless it stays in the
        ring). Returns whether the layer must not keep `h` for backward."""
        if self._stays(i) and i == 0:
            return False  # not a ring slot: the layer keeps it
        if i == 0:
            self.first = h
        if not self._stays(i):
            self._copy_out(i, h)
        return True

    def output(self, i):
        """The slot layer i writes its output (layer i + 1's input) into, or None for the last
        layer (its output goes to the head)."""
        if i + 1 >= self.L:
            return None
        slot = self._slot(i + 1)
        if self.first is not None and i >= 1:
            torch.cuda.current_stream(self.device).wait_event(self.saved[0])
            self.first = None
        return slot

    def finish_forward(self):
        if self.first is not None:
            torch.cuda.current_stream(self.device).wait_event(self.saved[0])
            self.first = None


class GuStream(_HostRing):
    """The [gate | up] GEMM output (2 * I fp16 per token) of the layers below `upto` (those
    that do not keep it on the GPU, FusedQwen2.keep_plan) waits in pinned host memory instead of
    being recomputed in backward: that GEMM is most of the backward's recomputation.

    The layers write their [gate | up] GEMM straight into their slot. The ring is sized to each
    step's tokens and allocated before the step's plan, so the layers keeping [gate | up] on the
    GPU are fewer by its size: their forward, with the whole ring allocated, is where a step
    with the layer inputs in host memory peaks.

    The 1.9 GiB host buffer is costly to pin (~0.95 s on a T4 host, most of it faulting the pages
    in, which holds up the training even on another thread): a thread of warmup.py faults it in
    as plain memory while the model loads (`pages`), and the first step pins it in place
    (~0.1 s)."""

    pages = None  # [L, HOST_ROWS, 2 * I] fp16 in pageable memory, from warmup.py

    def __init__(self, layers, device):
        super().__init__(layers, device)
        self.registered = None  # `host`, if pinned in place (unregistered before it is freed)
        self.chunk = None  # tokens per [gate | up] chunk (begin_chunks)

    def unregister(self):
        if self.registered is not None:
            torch.cuda.synchronize(self.device)  # no copy still reads it
            torch.cuda.cudart().cudaHostUnregister(self.registered.data_ptr())
            self.registered = self.host = None

    def ring_bytes(self, n, cols):
        return self.SLOTS * n * cols * 2

    def chunk_ring_bytes(self, chunk, cols):
        return self.SLOTS * chunk * cols * 2

    def release(self):
        """Free the ring: the compute stream has waited for every copy into or out of it."""
        self.ring = None

    def _host(self, n, cols):
        if self.host is None and GuStream.pages is not None:
            pages, GuStream.pages = GuStream.pages, None
            if pages.shape[1] >= n and pages.shape[2] == cols \
                    and _pin_in_place(pages, self.device):
                self.host = self.registered = pages
        if self.host is not None and self.host.shape[1] < n:
            self.unregister()  # _host_rows makes a bigger one
        self._host_rows(n, cols, torch.float16)  # 1.9 GiB for 2048 tokens

    def begin(self, n, cols):
        self._host(n, cols)
        self.ring = torch.empty((self.SLOTS, n, cols), device=self.device, dtype=torch.float16)
        self.n = n
        self.upto = 0
        self.in_slot = [None] * self.SLOTS
        self.chunk = None

    def begin_chunks(self, n, cols, chunk):
        """A step too long for the ring of whole outputs (but not lean): the [gate | up] of the
        layers below `upto`, in chunks of `chunk` tokens (their MLP runs in chunks, as a lean
        step's), goes to host memory through a ring of SLOTS chunk buffers; the backward fetches
        the chunks back one ahead, in its order (the layers from the last, a layer's chunks
        from the first)."""
        self._host(n, cols)
        self.ring = torch.empty((self.SLOTS, chunk, cols), device=self.device,
                                dtype=torch.float16)
        self.n, self.chunk, self.upto = n, chunk, 0
        self.k = 0  # chunks written so far
        self.copied = [None] * self.SLOTS  # per slot: its last copy out
        self.loaded = {}  # (layer, first token) -> (slot, event of its copy in)

    def chunk_slot(self, rows):
        """Forward: the slot the next chunk's [gate | up] GEMM writes into."""
        s = self.k % self.SLOTS
        self.k += 1
        if self.copied[s] is not None:
            torch.cuda.current_stream(self.device).wait_event(self.copied[s])
        return s, self.ring[s, :rows]

    def chunk_save(self, s, i, c0, c1):
        """Forward: copy chunk c0:c1 of layer i (in slot s) out."""
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        self.out_stream.wait_event(ready)
        with torch.cuda.stream(self.out_stream):
            self.host[i, c0:c1].copy_(self.ring[s, :c1 - c0], non_blocking=True)
            self.copied[s] = torch.cuda.Event()
            self.copied[s].record(self.out_stream)

    def _chunk_fetch(self, key, s):
        i, c0 = key
        c1 = min(self.n, c0 + self.chunk)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))  # the slot's last reader is done
        self.in_stream.wait_event(ready)
        if self.copied[s] is not None:  # (the forward's copy out of the slot)
            self.in_stream.wait_event(self.copied[s])
        with torch.cuda.stream(self.in_stream):
            self.ring[s, :c1 - c0].copy_(self.host[i, c0:c1], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.in_stream)
        self.loaded[key] = (s, done)

    def chunk_load(self, i, c0, c1):
        """Backward: chunk c0:c1 of layer i's [gate | up] on the current stream; starts
        fetching the next one."""
        key = (i, c0)
        if key not in self.loaded:
            self._chunk_fetch(key, 0)
        s, done = self.loaded.pop(key)
        torch.cuda.current_stream(self.device).wait_event(done)
        nxt = (i, c1) if c1 < self.n else (i - 1, 0)
        if nxt[0] >= 0:
            self._chunk_fetch(nxt, 1 - s)
        return self.ring[s, :c1 - c0]

    def holds(self, i):
        return i < self.upto

    def slot(self, i):
        """Forward of layer i: the buffer its [gate | up] goes into."""
        return self._slot(i)

    def save(self, i):
        """Forward of layer i, after its [gate | up] GEMM: copy it out."""
        if not self._stays(i):
            self._copy_out(i, self.ring[i % self.SLOTS, :self.n])


def _pin_in_place(t, device):
    """Page-lock a contiguous CPU tensor's memory where it is (cudaHostRegister, as
    torch.cuda._pin_memory_utils does) instead of copying it into a new pinned buffer: for the
    445 MiB embedding 22 ms on a T4 host, against 45 ms for the copy alone. Returns whether it
    did; the memory must be unregistered (cudaHostUnregister) before it is freed."""
    if t.device.type != "cpu" or not t.is_contiguous() or t.is_pinned():
        return False
    rc = int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0))
    if rc != 0:
        try:  # the failed call left its error as the runtime's last one; a launch check takes it
            torch.zeros(1, device=device)
        except RuntimeError:
            pass
    return rc == 0


def _align(n, a=256):
    return (n + a - 1) // a * a


def dequant(weights, device):
    """Dequantize one or more [n_i, K] weights stacked into one fp16 [sum n_i, K] buffer."""
    rows = sum(w.shape[0] for w in weights)
    out = torch.empty((rows, weights[0].shape[1]), device=device, dtype=torch.float16)
    group = weights[0].group
    if group is not None and group.packed is not None and group.weights == weights:
        return group.dequant_into(out)
    start = 0
    for w in weights:
        w.dequant_into(out[start:start + w.shape[0]])
        start += w.shape[0]
    return out


# ---------------------------------------------------------------------------------------------
# Per-step packing
# ---------------------------------------------------------------------------------------------
class UnsupportedBatch(ValueError):
    """The fused step does not take this batch; raised before the step changes anything, so the
    caller can run the original code instead."""


class Packed:
    """One micro-batch without its padding: real tokens of every row, back to back."""

    def __init__(self, input_ids, attention_mask, labels, rope, dropout, dropout_p):
        if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape \
                or labels.shape != input_ids.shape:
            raise UnsupportedBatch("inputs are not [rows, tokens]")
        B, T = input_ids.shape
        lengths = attention_mask.sum(dim=1)
        lengths_cpu = lengths.tolist()
        expected = torch.arange(T, device=input_ids.device)[None, :] < lengths[:, None]
        if not bool(torch.equal(attention_mask.bool(), expected)):
            raise UnsupportedBatch("attention_mask is not right-padded")
        flat = expected.reshape(-1).nonzero().squeeze(1)
        self.n = int(flat.numel())
        self.tok = flat  # each packed token's index in the padded [B, T] batch (int64)
        self.T = T
        self.rows = B
        self.lengths = [int(x) for x in lengths_cpu]
        self.max_len = max(self.lengths) if self.lengths else 0
        self.ids = input_ids.reshape(-1)[flat]
        self.pos = (flat % T).to(torch.int32)
        shifted = torch.full_like(labels, -100)
        shifted[:, :-1] = labels[:, 1:]
        self.targets = shifted.reshape(-1)[flat].contiguous()
        supervised = self.targets >= 0
        self.head_order = torch.argsort((~supervised).to(torch.int8), stable=True)
        self.n_supervised = int(supervised.sum())
        cu = [0]
        for length in self.lengths:
            cu.append(cu[-1] + length)
        self.cu_seqlens = torch.tensor(cu, device=input_ids.device, dtype=torch.int32)
        self.row_of_token = torch.repeat_interleave(
            torch.arange(B, device=input_ids.device), lengths)
        self.cos, self.sin = rope.table(self.max_len)
        self.dropout_p = dropout_p
        self.dropout = dropout  # a TorchDropout (the base's masks), or None without dropout
        self.drop_seed = self.drop_threads = None  # its seed and threads in this step (draw)
        self.drop_vec = dropout.vec if dropout is not None else 4
        self.lora16_views = None  # per layer: fp16 copies of its LoRA weights (FusedQwen2.step)
        self.keep_gu_from = 1 << 30  # layers from this index on keep [gate | up] for backward
        self.keep_all_from = 1 << 30  # ... and from this one every intermediate
        self.inputs = None  # an InputStream, if the layer inputs wait in host memory
        self.gus = None  # a GuStream, if [gate | up] of the layers below gus.upto does
        self.lean = False  # recompute and run the MLP in parts (FusedQwen2.lean)
        self.mlp_chunk = MLP_CHUNK

    def lora16(self, index, weights):
        if self.lora16_views is not None:
            return self.lora16_views[index]
        return tuple(None if t is None else t.to(torch.float16) for t in weights)

    def draw(self, H):
        """The counter of the next LoRA dropout mask of this step (the base's next nn.Dropout
        call, on its padded [B, T, H] LoRA input)."""
        seed, ctr, threads = self.dropout.draw(self.rows * self.T * H)
        assert self.drop_seed in (None, seed) and self.drop_threads in (None, threads)
        self.drop_seed, self.drop_threads = seed, threads
        return ctr


class TorchDropout:
    """The LoRA dropout masks of the base, bit for bit. PEFT's Linear4bit runs
    lora_B(lora_A(dropout(x))) on the layer's LoRA input x: an nn.Dropout(p) on a contiguous
    [B, T, H] tensor of `dtype`, which (torch.nn.functional.dropout -> native_dropout ->
    dropout_cuda in aten/src/ATen/native/cuda/Dropout.cu) takes a Philox state (seed, offset) from
    the default CUDA generator and advances its offset. `draw` takes the same state for the next
    call and advances the generator the same way, so every later draw (ours, or the base's own
    code after a fallback) sees the offset it would see in the base; the kernels compute the
    mask from it (kernels._torch_dropout_keep). The base calls dropout once per LoRA module, in
    the order the forward reaches them: layer 0 q_proj, layer 0 v_proj, layer 1 q_proj, ... (its
    gradient checkpointing, if any, replays under a forked generator state)."""

    BLOCK, UNROLL = 256, 4  # Dropout.cu: block_size, UNROLL

    CHECK_SEED = 20261031  # the self-check's own generator
    CHECK_TOKENS = 64  # tokens per row in the self-check

    def __init__(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.ok = None  # the self-check's verdict (check), once the first step has run it
        props = torch.cuda.get_device_properties(device)
        self.max_blocks = props.multi_processor_count \
            * (props.max_threads_per_multi_processor // self.BLOCK)
        self.gen = torch.cuda.default_generators[device.index if device.index is not None
                                                  else torch.cuda.current_device()]
        # the vectorized kernel (a contiguous, 16-byte aligned input with numel % VEC == 0)
        self.vec = 16 // torch.empty((), dtype=dtype).element_size()
        assert self.vec in (4, 8), dtype

    def draw(self, nelem):
        """(seed, counter, threads) of a dropout call on nelem elements; advances the generator."""
        assert nelem % self.vec == 0, nelem
        blocks = min(self.max_blocks, (nelem + self.BLOCK - 1) // self.BLOCK)
        increment = ((nelem - 1) // (self.BLOCK * blocks * self.UNROLL) + 1) * self.UNROLL
        seed, offset = self.gen.initial_seed(), self.gen.get_offset()
        self.gen.set_offset(offset + increment)
        if seed >= 2**63:  # the kernels take it as int64, its bits as they are
            seed -= 2**64
        return seed, offset // 4, blocks * self.BLOCK

    def check(self, B, T, H, p):
        """Whether our kernel's mask is PyTorch's on this GPU and build: the rms kernel's
        dropout of ones over [B, T, H] (a step's LoRA input) against the dropout kernel
        nn.Dropout runs (dropout_cuda, here through torch._fused_dropout, which takes a
        generator), both on a generator of their own - the default one is not touched."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.CHECK_SEED)
        ones = torch.ones((B, T, H), device=self.device, dtype=self.dtype)
        _, ref = torch._fused_dropout(ones, 1.0 - p, gen)
        nelem = B * T * H
        blocks = min(self.max_blocks, (nelem + self.BLOCK - 1) // self.BLOCK)

        class _Check:
            dropout_p, drop_seed, drop_threads, drop_vec = p, self.CHECK_SEED, \
                blocks * self.BLOCK, self.vec
            tok = torch.arange(B * T, device=self.device)
            eps = 1e-6

        x = ones.view(B * T, H).float()
        _, _, yq, _, _ = _rms_fwd(_Check, x, torch.ones(H, device=self.device), _Check,
                                  write_y=False, seeds=(0, 0), drop_q=True)
        return bool(torch.equal((yq != 0).view(B, T, H), ref.bool()))


class RopeTable:
    """cos / sin of the model's own rotary embedding (first half of head_dim: Qwen2 repeats it)."""

    def __init__(self, rotary, head_dim, device):
        self.rotary, self.half, self.device = rotary, head_dim // 2, device
        self.cos = self.sin = None

    def table(self, length):
        if self.cos is None or self.cos.shape[0] < length:
            size = max(1024, _pow2(length))
            dummy = torch.zeros(1, device=self.device, dtype=torch.float32)
            pos = torch.arange(size, device=self.device)[None, :]
            with torch.no_grad():
                cos, sin = self.rotary(dummy, pos)
            cos, sin = cos[0].float(), sin[0].float()
            if not (torch.equal(cos[:, :self.half], cos[:, self.half:])
                    and torch.equal(sin[:, :self.half], sin[:, self.half:])):
                raise ValueError("rotary table is not rotate_half style")
            self.cos = cos[:, :self.half].contiguous()
            self.sin = sin[:, :self.half].contiguous()
        return self.cos, self.sin


# ---------------------------------------------------------------------------------------------
# Attention: PyTorch's own memory-efficient kernel in its varlen form (cu_seqlens), causal.
# ---------------------------------------------------------------------------------------------
def _attn_fwd(q, k, v, packed, scale):
    out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
        q[None], k[None], v[None], None, packed.cu_seqlens, packed.cu_seqlens, packed.max_len,
        packed.max_len, 0.0, 1, True, scale=scale)
    return out[0], lse, seed, offset


def _attn_bwd(dout, q, k, v, out, lse, seed, offset, packed, scale):
    dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
        dout[None], q[None], k[None], v[None], None, out[None], packed.cu_seqlens,
        packed.cu_seqlens, packed.max_len, packed.max_len, lse, 0.0, seed, offset, 1, False,
        scale=scale)
    return dq[0], dk[0], dv[0]


# ---------------------------------------------------------------------------------------------
# One decoder layer
# ---------------------------------------------------------------------------------------------
class LayerSpec:
    def __init__(self, layer, config, lora_names, index):
        attn, mlp = layer.self_attn, layer.mlp
        self.index = index
        self.stream = None  # a WeightStream, once training starts
        self.H = config.hidden_size
        self.D = getattr(config, "head_dim", None) or self.H // config.num_attention_heads
        self.HQ = config.num_attention_heads
        self.HKV = config.num_key_value_heads
        self.I = config.intermediate_size
        self.eps = float(layer.input_layernorm.variance_epsilon)
        assert float(layer.post_attention_layernorm.variance_epsilon) == self.eps
        self.ln1 = layer.input_layernorm.weight
        self.ln2 = layer.post_attention_layernorm.weight
        self.scale = attn.scaling

        def base_of(module):
            return getattr(module, "base_layer", module)

        projs = {name: getattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
        projs.update({name: getattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")})
        self.base_modules = {name: base_of(m) for name, m in projs.items()}
        self.w = {name: NF4Weight(base_of(m).weight) for name, m in projs.items()}
        biases = [base_of(projs[name]).bias for name in ("q_proj", "k_proj", "v_proj")]
        self.bias_qkv = torch.cat([b.detach().to(torch.float16) for b in biases]).contiguous()
        self.lora = {}
        for short in ("q", "k", "v"):
            module = projs[f"{short}_proj"]
            if short in lora_names:
                name = lora_names[short]
                self.lora[short] = (module.lora_A[name].weight, module.lora_B[name].weight,
                                    float(module.scaling[name]))

    def lora_params(self):
        out = []
        for short in ("q", "k", "v"):
            if short in self.lora:
                out += [self.lora[short][0], self.lora[short][1]]
            else:
                out += [None, None]
        return out


def _lora_scale(spec):
    scales = {value[2] for value in spec.lora.values()}
    assert len(scales) <= 1
    return scales.pop() if scales else 1.0


def _dummy16(device, dtype=torch.float16):
    """A small fp16 tensor standing in for an output or input a launch does not use, so
    every launch of a kernel has the same argument types (one compiled variant)."""
    t = _DUMMIES.get((device, dtype))
    if t is None:
        t = _DUMMIES[device, dtype] = torch.empty((8,), device=device, dtype=dtype)
    return t


def _dropout_args(packed, x, seeds):
    """The rms kernels' dropout arguments: the tokens' indices in the padded batch, the keep
    probability and scale, the generator's seed, the counters of the q and v masks
    (TorchDropout.draw), its threads."""
    tok = getattr(packed, "tok", None)
    p = packed.dropout_p
    return (tok if tok is not None else _dummy16(x.device, torch.int64), 1.0 - p,
            1.0 / (1.0 - p), getattr(packed, "drop_seed", None) or 0, seeds[0], seeds[1],
            getattr(packed, "drop_threads", None) or 1)


_DUMMIES = {}


def _rms_fwd(spec, x, weight, packed, *, delta=None, write_y=True, seeds=(0, 0),
             drop_q=False, drop_v=False):
    n, H = x.shape
    dev = x.device
    xsum = torch.empty_like(x) if delta is not None else x
    y = torch.empty((n, H), device=dev, dtype=torch.float16) if write_y else None
    yq = torch.empty((n, H), device=dev, dtype=torch.float16) if drop_q else None
    yv = torch.empty((n, H), device=dev, dtype=torch.float16) if drop_v else None
    rstd = torch.empty((n,), device=dev, dtype=torch.float32)
    dummy = _dummy16(dev)
    if n:
        tok, p_keep, keep_scale, seed, ctr_q, ctr_v, threads = _dropout_args(packed, x, seeds)
        _launch(K.rms_fwd_kernel, (n,),
            x, delta if delta is not None else dummy, weight, xsum,
            y if y is not None else dummy, yq if yq is not None else dummy,
            yv if yv is not None else dummy, rstd, tok, H, spec.eps, p_keep, keep_scale, seed,
            ctr_q, ctr_v, threads, int(delta is not None), int(write_y), int(drop_q),
            int(drop_v), VEC=getattr(packed, "drop_vec", 4), BLOCK=_pow2(H), num_warps=8)
    return xsum, y, yq, yv, rstd


def _rms_bwd(spec, dy, x, rstd, weight, packed, *, dres=None, tq=None, tv=None, seeds=(0, 0),
             grad_scale=1.0, out=None):
    n, H = x.shape
    dx = out if out is not None else torch.empty_like(x)
    if n:
        tok, p_keep, keep_scale, seed, ctr_q, ctr_v, threads = _dropout_args(packed, x, seeds)
        _launch(K.rms_bwd_kernel, (n,),
            dy, tq if tq is not None else dy, tv if tv is not None else dy, x, rstd, weight,
            dres if dres is not None else x, dx, tok, H, dy.stride(0), p_keep, keep_scale,
            seed, ctr_q, ctr_v, threads, grad_scale, int(tq is not None),
            int(tv is not None), int(dres is not None), VEC=getattr(packed, "drop_vec", 4),
            BLOCK=_pow2(H), num_warps=8)
    return dx


def _rope_fwd(spec, qkv, lq, lv, packed, lora_scale):
    n = qkv.shape[0]
    dev = qkv.device
    q = torch.empty((n, spec.HQ, spec.D), device=dev, dtype=torch.float16)
    # k and v come out repeated for the GQA groups ([n, HQ, D]), ready for the attention kernel
    k = torch.empty((n, spec.HQ, spec.D), device=dev, dtype=torch.float16)
    v = torch.empty((n, spec.HQ, spec.D), device=dev, dtype=torch.float16)
    if n:
        _launch(K.qkv_rope_fwd_kernel, (n,),
            qkv, lq if lq is not None else qkv, lv if lv is not None else qkv, packed.cos,
            packed.sin, packed.pos, q, k, v, lora_scale, HQ=spec.HQ, HKV=spec.HKV, D=spec.D,
            BHQ=_pow2(spec.HQ), BHKV=_pow2(spec.HKV), GROUP=spec.HQ // spec.HKV,
            LORA_Q=lq is not None, LORA_V=lv is not None, num_warps=4)
    return q, k, v


def _rope_bwd(spec, dq, dk, dv, packed):
    n = dq.shape[0]
    C = (spec.HQ + 2 * spec.HKV) * spec.D
    dqkv = torch.empty((n, C), device=dq.device, dtype=torch.float16)
    if n:
        _launch(K.qkv_rope_bwd_kernel, (n,),
            dq.contiguous(), dk.contiguous(), dv.contiguous(), packed.cos, packed.sin,
            packed.pos, dqkv, HQ=spec.HQ, HKV=spec.HKV, D=spec.D, BHQ=_pow2(spec.HQ),
            BHKV=_pow2(spec.HKV), GROUP=spec.HQ // spec.HKV, num_warps=4)
    return dqkv


def _glu_fwd(gu, I, out=None):
    n = gu.shape[0]
    m = out if out is not None else torch.empty((n, I), device=gu.device, dtype=torch.float16)
    if n:
        _launch(K.glu_fwd_kernel, (n, _cdiv(I, GLU_BLOCK)), gu, m, I, BLOCK=GLU_BLOCK)
    return m


def _glu_bwd(dm, gu, I):
    n = gu.shape[0]
    if n:
        _launch(K.glu_bwd_kernel, (n, _cdiv(I, GLU_BLOCK)), dm, gu, I, BLOCK=GLU_BLOCK)


def _ce(logits, targets, tok_loss, entropy, correct, zero_unsupervised=True):
    rows, V = logits.shape
    if rows:
        _launch(K.lm_head_ce_kernel, (rows,), logits, targets, tok_loss, entropy, correct, V,
                int(zero_unsupervised), BLOCK=CE_BLOCK, num_warps=8)


def _ce_piece(logits, w, targets, state, best_idx, xt, alpha, wt, v0, n_sup):
    rows, width = logits.shape
    if rows:
        _launch(K.lm_head_piece_kernel, (rows,), logits, w, targets, state, best_idx, xt, alpha,
                wt, v0, width, n_sup, w.shape[1], BLOCK=CE_BLOCK, HBLOCK=_pow2(w.shape[1]),
                num_warps=8)


def _ce_finish(state, best_idx, xt, targets, tok_loss, entropy, correct, acc, wt, dh, n_sup):
    rows, H = state.shape[0], acc.shape[1]
    if rows:
        _launch(K.lm_head_finish_kernel, (rows,), state, best_idx, xt, targets, tok_loss, entropy,
                correct, acc, wt, dh, n_sup, H, HBLOCK=_pow2(H), num_warps=4)


def _layer_output(spec, packed, h2, d):
    """h2 + d, the layer's output; written into its InputStream slot if there is one."""
    out = packed.inputs.output(spec.index) if packed.inputs is not None else None
    return torch.add(h2, d) if out is None else torch.add(h2, d, out=out)


def _layer_forward(spec, h, packed, lora16, seeds, keep, gu=None, return_gu=False, full=False,
                   part=None):
    """One decoder layer forward. With keep=True (the backward's recomputation) returns what the
    backward reads, dequantized weights included; a `gu` kept from the forward replaces the
    gate/up GEMM. Otherwise returns h3 and, with return_gu, the [gate | up] GEMM output, or with
    full, every intermediate the backward reads (without the weights).

    A lean step (packed.lean) recomputes in two parts, so the attention intermediates and the
    [gate | up] output never coexist: part="mlp_input" returns the residual after attention, its
    rstd and the MLP input b; part="attention" returns what the attention and LoRA backward read.
    Its forward runs the MLP over MLP_CHUNK tokens at a time."""
    n, H = h.shape
    D, HQ = spec.D, spec.HQ
    dev = h.device
    aq_w, bq_w, av_w, bv_w = lora16
    lora_scale = _lora_scale(spec)
    saved = {}

    # input RMSNorm (+ the dropout'ed copies the LoRA A matrices read)
    _, a, aq, av, rstd1 = _rms_fwd(spec, h, spec.ln1, packed, seeds=seeds,
                                   drop_q=aq_w is not None, drop_v=av_w is not None)

    # q, k, v in one GEMM; LoRA as two thin GEMMs per target
    w = dequant([spec.w["q_proj"], spec.w["k_proj"], spec.w["v_proj"]], dev)
    qkv = torch.addmm(spec.bias_qkv, a, w.t())
    del a
    if keep:
        saved["w_qkv"] = w
    del w
    xq = xv = lq = lv = None
    if aq_w is not None:
        xq = aq @ aq_w.t()
        lq = xq @ bq_w.t()
    if av_w is not None:
        xv = av @ av_w.t()
        lv = xv @ bv_w.t()
    q, k, v = _rope_fwd(spec, qkv, lq, lv, packed, lora_scale)
    del qkv, lq, lv
    k_e, v_e = k, v
    out, lse, aseed, aoffset = _attn_fwd(q, k_e, v_e, packed, spec.scale)
    if part == "attention":
        saved.update(rstd1=rstd1, aq=aq, av=av, xq=xq, xv=xv, q=q, k_e=k_e, v_e=v_e, out=out,
                     lse=lse, aseed=aseed, aoffset=aoffset)
        return None, saved
    if part == "mlp_input":
        saved.pop("w_qkv", None)
        del aq, av, xq, xv, q, k, v, k_e, v_e, lse

    w = dequant([spec.w["o_proj"]], dev)
    o = out.reshape(n, HQ * D) @ w.t()
    if keep:
        saved["w_o"] = w
    del w
    h2, b, _, _, rstd2 = _rms_fwd(spec, h, spec.ln2, packed, delta=o)
    del o
    if part == "mlp_input":
        return None, {"h2": h2, "rstd2": rstd2, "b": b}
    w = dequant([spec.w["gate_proj"], spec.w["up_proj"]], dev)
    chunks = packed.gus is not None and packed.gus.chunk and packed.gus.holds(spec.index)
    if (packed.lean or chunks) and not keep and gu is None and not full and not return_gu:
        # the MLP over chunks of tokens: [gate | up] of MLP_CHUNK tokens at a time (to host
        # memory, if the GuStream takes this layer's in chunks)
        del aq, av, q, k, v, k_e, v_e, out, lse
        m = torch.empty((n, spec.I), device=dev, dtype=torch.float16)
        gus = packed.gus if chunks else None
        for c0 in range(0, n, packed.mlp_chunk):
            c1 = min(n, c0 + packed.mlp_chunk)
            if gus is None:
                _glu_fwd(b[c0:c1] @ w.t(), spec.I, out=m[c0:c1])
            else:  # into a GuStream chunk slot, copied to host memory for the backward
                slot, gu = gus.chunk_slot(c1 - c0)
                _glu_fwd(torch.mm(b[c0:c1], w.t(), out=gu), spec.I, out=m[c0:c1])
                gus.chunk_save(slot, spec.index, c0, c1)
        del w, b
        w = dequant([spec.w["down_proj"]], dev)
        d = m @ w.t()
        del w, m
        return _layer_output(spec, packed, h2, d), None
    if gu is None and not (keep and chunks):
        gus = packed.gus if not keep and packed.gus is not None \
            and packed.gus.holds(spec.index) else None
        if gus is None:
            gu = b @ w.t()
        else:  # into its GuStream slot, copied to host memory for the backward
            gu = torch.mm(b, w.t(), out=gus.slot(spec.index))
            gus.save(spec.index)
    del b
    if keep or full:
        saved.update(rstd1=rstd1, aq=aq, av=av, xq=xq, xv=xv, q=q, k_e=k_e, v_e=v_e,
                     out=out, lse=lse, aseed=aseed, aoffset=aoffset, h2=h2, rstd2=rstd2, gu=gu)
    if keep:
        saved["w_gu"] = w
        return None, saved
    del w, aq, av, q, k, v, k_e, v_e, out, lse
    m = _glu_fwd(gu, spec.I)
    if not return_gu:
        gu = None
    w = dequant([spec.w["down_proj"]], dev)
    d = m @ w.t()
    del w, m
    return _layer_output(spec, packed, h2, d), (saved if full else gu)


class _Layer(torch.autograd.Function):
    """One decoder layer. Its input (the fp32 residual) is kept for backward, on the GPU or in
    host memory (InputStream), and, as the step's plan allows (FusedQwen2.keep_plan), its
    [gate | up] GEMM output (on the GPU or in host memory, GuStream) or every intermediate. The
    backward runs the forward again, with the same dropout seeds, as far as it has to (up to the
    MLP input, or none of it), in two parts in a lean step, and then goes back through it. The
    LoRA weights are inputs, so autograd hands their gradients to the optimizer."""

    @staticmethod
    def forward(ctx, h, spec, packed, aq_w, bq_w, ak_w, bk_w, av_w, bv_w):
        if ak_w is not None:
            raise NotImplementedError("LoRA on k_proj")
        # the base's next two dropout calls: this layer's q_proj, then v_proj
        drop = packed.dropout is not None
        seeds = (packed.draw(spec.H) if drop and aq_w is not None else 0,
                 packed.draw(spec.H) if drop and av_w is not None else 0)
        lora16 = packed.lora16(spec.index, (aq_w, bq_w, av_w, bv_w))
        offloaded = packed.inputs is not None and packed.inputs.save(spec.index, h)
        if spec.stream is not None:
            spec.stream.acquire(spec.index, spec.index + 1)
        # What this layer keeps for backward depends on the room the step has (keep_plan):
        # the [gate | up] GEMM output first (most recomputation saved per byte), then every
        # intermediate; the rest is recomputed in backward.
        ctx.kept_gu = ctx.kept_all = None
        if spec.index >= packed.keep_all_from:
            h3, ctx.kept_all = _layer_forward(spec, h, packed, lora16, seeds, keep=False,
                                              full=True)
        else:
            h3, ctx.kept_gu = _layer_forward(spec, h, packed, lora16, seeds, keep=False,
                                             return_gu=spec.index >= packed.keep_gu_from)
        if spec.stream is not None:
            spec.stream.release(spec.index)
        ctx.save_for_backward(None if offloaded else h, aq_w, bq_w, av_w, bv_w)
        ctx.spec, ctx.packed, ctx.seeds = spec, packed, seeds
        return h3

    @staticmethod
    def backward(ctx, dh3):
        h, aq_w, bq_w, av_w, bv_w = ctx.saved_tensors
        spec, packed, seeds = ctx.spec, ctx.packed, ctx.seeds
        if h is None:  # in the InputStream
            h = packed.inputs.load(spec.index)
        s = _lora_scale(spec)
        n, H = h.shape
        D, HQ, HKV = spec.D, spec.HQ, spec.HKV
        C = (HQ + 2 * HKV) * D
        dev = h.device
        dh3 = dh3.contiguous()
        lora16 = packed.lora16(spec.index, (aq_w, bq_w, av_w, bv_w))
        if spec.stream is not None:
            spec.stream.acquire(spec.index, spec.index - 1)
        # MLP output gradient first: the down_proj weight is freed before the recomputation
        # dequantizes the other three, so one layer never holds all four at once.
        w = dequant([spec.w["down_proj"]], dev)
        dm = dh3.to(torch.float16) @ w
        del w
        if packed.lean:
            # recompute up to the MLP input only, go back through the MLP a chunk of tokens at a
            # time, then recompute the attention for its backward
            with torch.no_grad():
                _, f = _layer_forward(spec, h, packed, lora16, seeds, keep=True,
                                      part="mlp_input")
            b = f.pop("b")
            w = dequant([spec.w["gate_proj"], spec.w["up_proj"]], dev)
            db = torch.empty((n, H), device=dev, dtype=torch.float16)
            for c0 in range(0, n, packed.mlp_chunk):
                c1 = min(n, c0 + packed.mlp_chunk)
                gu = b[c0:c1] @ w.t()
                _glu_bwd(dm[c0:c1], gu, spec.I)
                torch.mm(gu, w, out=db[c0:c1])
                del gu
            del w, b, dm
            dh2 = _rms_bwd(spec, db, f.pop("h2"), f.pop("rstd2"), spec.ln2, packed, dres=dh3)
            del db, dh3
            with torch.no_grad():
                _, f = _layer_forward(spec, h, packed, lora16, seeds, keep=True,
                                      part="attention")
        else:
            if ctx.kept_all is not None:  # nothing to recompute; weights dequantized below
                f, ctx.kept_all = ctx.kept_all, None
            else:
                gu = ctx.kept_gu
                if gu is None and packed.gus is not None and packed.gus.holds(spec.index) \
                        and not packed.gus.chunk:
                    gu = packed.gus.load(spec.index)
                with torch.no_grad():
                    _, f = _layer_forward(spec, h, packed, lora16, seeds, keep=True, gu=gu)
                ctx.kept_gu = gu = None

            # MLP
            gu = f.pop("gu", None)
            w = f.pop("w_gu") if "w_gu" in f else dequant(
                [spec.w["gate_proj"], spec.w["up_proj"]], dev)
            if gu is None:  # in host memory in chunks (GuStream.begin_chunks)
                db = torch.empty((n, H), device=dev, dtype=torch.float16)
                for c0 in range(0, n, packed.mlp_chunk):
                    c1 = min(n, c0 + packed.mlp_chunk)
                    gu = packed.gus.chunk_load(spec.index, c0, c1)
                    _glu_bwd(dm[c0:c1], gu, spec.I)
                    torch.mm(gu, w, out=db[c0:c1])
            else:
                _glu_bwd(dm, gu, spec.I)
                db = gu @ w
            del w, dm
            del gu
            dh2 = _rms_bwd(spec, db, f.pop("h2"), f.pop("rstd2"), spec.ln2, packed, dres=dh3)
            del db, dh3

        # attention
        w = f.pop("w_o") if "w_o" in f else dequant([spec.w["o_proj"]], dev)
        dout = (dh2.to(torch.float16) @ w).view(n, HQ, D)
        del w
        dq, dk, dv = _attn_bwd(dout, f.pop("q"), f.pop("k_e"), f.pop("v_e"), f.pop("out"),
                               f.pop("lse"), f.pop("aseed"), f.pop("aoffset"), packed, spec.scale)
        del dout
        dqkv = _rope_bwd(spec, dq, dk, dv, packed)
        del dq, dk, dv
        # the first layer's input is the frozen embedding: no gradient through the norm, the qkv
        # weight or the LoRA A matrices to its input
        need_dh = ctx.needs_input_grad[0]
        w = f.pop("w_qkv", None)
        if need_dh:
            if w is None:
                w = dequant([spec.w["q_proj"], spec.w["k_proj"], spec.w["v_proj"]], dev)
            da = dqkv @ w
        del w

        # LoRA: dB = s g^T x, dx = s g B, dA = dx^T drop(a), d drop(a) = dx A
        grads = {"q": (None, None, None), "v": (None, None, None)}
        for short, a_w, b_w, cols in (("q", lora16[0], lora16[1], slice(0, HQ * D)),
                                      ("v", lora16[2], lora16[3], slice((HQ + HKV) * D, C))):
            if a_w is None:
                continue
            x_lo, a_in = f.pop("x" + short), f.pop("a" + short)
            g = dqkv[:, cols]
            db_w = (g.t() @ x_lo).float() * s
            dx_lo = (g @ b_w) * s
            da_w = (dx_lo.t() @ a_in).float()
            grads[short] = (da_w, db_w, dx_lo @ a_w if need_dh else None)
            del x_lo, a_in
        del dqkv
        if spec.stream is not None:
            spec.stream.release(spec.index)
        dh = None
        if need_dh:
            dh = _rms_bwd(spec, da, h, f["rstd1"], spec.ln1, packed, dres=dh2,
                          tq=grads["q"][2], tv=grads["v"][2], seeds=seeds)
        return (dh, None, None, grads["q"][0], grads["q"][1], None, None, grads["v"][0],
                grads["v"][1])


# ---------------------------------------------------------------------------------------------
# Final norm + lm_head + cross-entropy + metrics, chunked
# ---------------------------------------------------------------------------------------------
class HeadStats:
    """What the head hands back for the logged metrics (row order, per-token loss, entropy,
    correct). The head also passes one to _rms_fwd / _rms_bwd as the layer spec, of which they
    read only `eps`."""


class _Head(torch.autograd.Function):
    """Final RMSNorm + lm_head + cross-entropy + the logged metrics, in chunks of rows. The rows
    are taken supervised-first: every row needs its logits (entropy and accuracy are logged over
    all of them), but only rows with a target have a gradient, so chunks of prompt rows skip the
    backward GEMM through the 151,936 x 1536 lm_head weight."""

    @staticmethod
    def forward(ctx, h, norm_w, eps, lm_w, packed, denom, stats):
        n, H = h.shape
        dev = h.device
        spec = HeadStats()
        spec.eps = eps
        order = packed.head_order
        n_sup = packed.n_supervised
        hp = h.index_select(0, order)
        targets = packed.targets.index_select(0, order)
        _, hn, _, _, rstd = _rms_fwd(spec, hp, norm_w, packed)
        dhp = torch.empty_like(hp)
        tok_loss = torch.empty((n,), device=dev, dtype=torch.float32)
        entropy = torch.empty((n,), device=dev, dtype=torch.float32)
        correct = torch.empty((n,), device=dev, dtype=torch.int32)
        inv = 1.0 / float(denom) if float(denom) else float("nan")  # 0 / 0, as in the base
        for c0 in range(0, n, HEAD_CHUNK):
            c1 = min(n, c0 + HEAD_CHUNK)
            logits = hn[c0:c1] @ lm_w.t()
            needs_grad = c0 < n_sup
            _ce(logits, targets[c0:c1], tok_loss[c0:c1], entropy[c0:c1], correct[c0:c1],
                zero_unsupervised=needs_grad)
            if needs_grad:
                dhn = logits @ lm_w
                del logits
                _rms_bwd(spec, dhn, hp[c0:c1], rstd[c0:c1], norm_w, packed, grad_scale=inv,
                         out=dhp[c0:c1])
                del dhn
            else:
                del logits
                dhp[c0:c1].zero_()
        dh = torch.empty_like(h)
        dh.index_copy_(0, order, dhp)
        del dhp, hp
        ctx.save_for_backward(dh)
        stats.order, stats.tok_loss = order, tok_loss
        stats.entropy, stats.correct = entropy, correct
        return tok_loss.sum() * inv

    @staticmethod
    def backward(ctx, grad):
        (dh,) = ctx.saved_tensors
        return dh * grad, None, None, None, None, None, None


# ---------------------------------------------------------------------------------------------
# The tied embedding / lm_head weight in pinned host memory
# ---------------------------------------------------------------------------------------------
class HostHead:
    """The frozen, tied embedding / lm_head weight (151,936 x 1536 fp16, 445 MiB) stays in pinned
    host memory for the whole run. The embedding rows of a step are gathered on the CPU and copied
    over; the lm_head is streamed to the GPU in pieces of PIECE vocabulary rows on a side stream
    into BUFFERS device buffers, the first ones during the last layers of the forward, the next
    one as soon as the head is done with a buffer. The buffers only exist from that prefetch to
    the end of the head."""

    PIECE = 8192  # vocabulary rows per piece (24 MiB); a multiple of the CE kernels' BLOCK
    BUFFERS = 3
    PREFETCH_LAYERS = 3  # the copies start once the forward reaches the last this many layers

    def __init__(self, weight, device):
        self.weight, self.device = weight, device
        self.V, self.H = weight.shape
        self.pieces = [(v0, min(self.PIECE, self.V - v0)) for v0 in range(0, self.V, self.PIECE)]
        self.stream = torch.cuda.Stream(device)
        self.copied = [torch.cuda.Event() for _ in range(self.BUFFERS)]
        self.released = [torch.cuda.Event() for _ in range(self.BUFFERS)]
        self.start = torch.cuda.Event()
        self.bufs = None
        self.rows = None  # pinned staging buffer of the embedding rows
        self.rows_copied = torch.cuda.Event()

    @property
    def host(self):
        return self.weight.data

    def embed(self, ids):
        """fp32 hidden states of the token ids: the fp16 rows gathered on the CPU, copied over."""
        ids_cpu = ids.cpu()
        n = ids_cpu.numel()
        if self.rows is None or self.rows.shape[0] < n:
            self.rows_copied.synchronize()
            self.rows = torch.empty((max(n, _HostRing.HOST_ROWS), self.H),
                                    dtype=self.host.dtype, pin_memory=True)
        self.rows_copied.synchronize()  # the previous step's copy has read the staging buffer
        rows = self.rows[:n]
        torch.index_select(self.host, 0, ids_cpu, out=rows)
        out = rows.to(self.device, non_blocking=True)
        self.rows_copied.record(torch.cuda.current_stream(self.device))
        return out.float()

    def _copy(self, j):
        b = j % self.BUFFERS
        v0, width = self.pieces[j]
        self.stream.wait_event(self.released[b])
        with torch.cuda.stream(self.stream):
            self.bufs[b][:width].copy_(self.host[v0:v0 + width], non_blocking=True)
            self.copied[b].record(self.stream)

    def prefetch(self):
        """Start copying the first pieces once the compute stream gets here."""
        if self.bufs is not None:
            return
        self.bufs = [torch.empty((self.PIECE, self.H), device=self.device, dtype=self.host.dtype)
                     for _ in range(self.BUFFERS)]
        self.start.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(self.start)
        for j in range(min(self.BUFFERS, len(self.pieces))):
            self._copy(j)

    def stream_pieces(self):
        """Yield (first row, rows, device piece) for the whole vocabulary, in order."""
        self.prefetch()
        current = torch.cuda.current_stream(self.device)
        for j, (v0, width) in enumerate(self.pieces):
            b = j % self.BUFFERS
            current.wait_event(self.copied[b])
            yield v0, width, self.bufs[b][:width]
            self.released[b].record(current)
            if j + self.BUFFERS < len(self.pieces):
                self._copy(j + self.BUFFERS)
        self.bufs = None  # every copy has been waited for: the buffers are free again

    def peak_bytes(self, n):
        """Device memory of the streamed head over its input: buffers, a piece of logits and
        the per-row work tensors."""
        return self.BUFFERS * self.PIECE * self.H * 2 + n * (self.PIECE * 2 + self.H * 24 + 64)


def _head_streamed(h, norm_w, eps, head, packed, denom, stats):
    """_Head.forward with the lm_head weight streamed in pieces of the vocabulary (HostHead):
    every row's logits of a piece, the online CE state carried over (lm_head_piece_kernel), and
    for the rows with a target, sum_v exp(logit_v - max) W_v accumulated in fp32, rescaled
    whenever the running max grows. Loss, entropy and accuracy are lm_head_ce_kernel's to the bit;
    the gradient is formed with the same fp16 GEMM inputs and fp32 accumulation, in a different
    order (piece by piece instead of one GEMM over the whole vocabulary)."""
    n, H = h.shape
    dev = h.device
    spec = HeadStats()
    spec.eps = eps
    order = packed.head_order
    n_sup = packed.n_supervised
    hp = h.index_select(0, order)
    targets = packed.targets.index_select(0, order)
    _, hn, _, _, rstd = _rms_fwd(spec, hp, norm_w, packed)
    # every row's (max, sum of exp, sum of exp * logit, best logit) = (-inf, 0, 0, -inf), filled
    # on the device: a torch.tensor from host memory would wait for the whole forward first
    state = torch.full((n, 4), float("-inf"), device=dev, dtype=torch.float32)
    state[:, 1:3] = 0.0
    best_idx = torch.zeros((n,), device=dev, dtype=torch.int32)
    xt = torch.zeros((n,), device=dev, dtype=torch.float32)
    alpha = torch.empty((max(n_sup, 1),), device=dev, dtype=torch.float32)
    acc = torch.zeros((max(n_sup, 1), H), device=dev, dtype=torch.float32)
    wt = torch.zeros((max(n_sup, 1), H), device=dev, dtype=torch.float16)
    for v0, width, w in head.stream_pieces():
        logits = hn @ w.t()
        _ce_piece(logits, w, targets, state, best_idx, xt, alpha, wt, v0, n_sup)
        if n_sup:
            acc[:n_sup].mul_(alpha[:n_sup, None])
            torch.addmm(acc[:n_sup], logits[:n_sup], w, out_dtype=torch.float32,
                        out=acc[:n_sup])
        del logits
    tok_loss = torch.empty((n,), device=dev, dtype=torch.float32)
    entropy = torch.empty((n,), device=dev, dtype=torch.float32)
    correct = torch.empty((n,), device=dev, dtype=torch.int32)
    dhn = torch.empty((max(n_sup, 1), H), device=dev, dtype=torch.float16)
    _ce_finish(state, best_idx, xt, targets, tok_loss, entropy, correct, acc, wt, dhn, n_sup)
    del state, best_idx, xt, alpha, acc, wt, hn
    inv = 1.0 / float(denom) if float(denom) else float("nan")  # 0 / 0, as in the base
    dhp = torch.empty_like(hp)
    _rms_bwd(spec, dhn[:n_sup], hp[:n_sup], rstd[:n_sup], norm_w, packed, grad_scale=inv,
             out=dhp[:n_sup])
    dhp[n_sup:].zero_()
    del dhn, hp
    dh = torch.empty_like(h)
    dh.index_copy_(0, order, dhp)
    stats.order, stats.tok_loss = order, tok_loss
    stats.entropy, stats.correct = entropy, correct
    return dh, tok_loss.sum() * inv


class _HostHead(torch.autograd.Function):
    """_Head with the lm_head weight in host memory (HostHead)."""

    @staticmethod
    def forward(ctx, h, norm_w, eps, head, packed, denom, stats):
        dh, loss = _head_streamed(h, norm_w, eps, head, packed, denom, stats)
        ctx.save_for_backward(dh)
        return loss

    @staticmethod
    def backward(ctx, grad):
        (dh,) = ctx.saved_tensors
        return dh * grad, None, None, None, None, None, None


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------
class FusedQwen2:
    """Holds what the step needs from a PEFT-wrapped, 4-bit Qwen2ForCausalLM."""

    def __init__(self, peft_model, seed=0):
        # (seed: unused - the LoRA dropout masks come from the CUDA generator, as the base's)
        causal = peft_model.get_base_model()
        inner = causal.model
        config = causal.config
        self.device = inner.norm.weight.device  # the embedding may be in host memory
        lora_names = _lora_targets(inner.layers[0])
        self.layers = [LayerSpec(layer, config, lora_names, i)
                       for i, layer in enumerate(inner.layers)]
        self.base_modules = [spec.base_modules[name] for spec in self.layers
                             for name in spec.base_modules]
        self.stream = None
        self.head = None  # a HostHead while the embedding / lm_head weight is in host memory
        self.registered = None  # the embedding's host memory, if pinned where the loader put it
        self.inputs = None  # an InputStream while the layer inputs go to host memory
        self.gus = None  # a GuStream while [gate | up] outputs may go to host memory
        self.embed_module = inner.embed_tokens
        self.dropout_p = _lora_dropout(inner.layers[0], lora_names)
        self.norm_w = inner.norm.weight
        self.eps = float(inner.norm.variance_epsilon)
        self.embed = inner.embed_tokens.weight
        self.lm_w = causal.lm_head.weight
        self.rope = RopeTable(inner.rotary_emb, self.layers[0].D, self.device)
        # the LoRA dropout masks: the base's (TorchDropout), on its LoRA input's dtype
        self.dropout = TorchDropout(self.device, _lora_input_dtype(inner.layers[0], lora_names)) \
            if self.dropout_p > 0 else None
        self.mlp_chunk = MLP_CHUNK
        # One fp16 buffer for every LoRA weight: refreshed by a single foreach copy per step
        # instead of four casts per layer in forward and again in backward.
        self.lora_fp32, shapes = [], []
        for spec in self.layers:
            for t in spec.lora_params():
                if t is not None:
                    self.lora_fp32.append(t)
                    shapes.append(t.shape)
        total = sum(int(torch.Size(sh).numel()) for sh in shapes)
        self.lora_buf = torch.empty((total,), device=self.device, dtype=torch.float16)
        self.lora_views, start = [], 0
        for sh in shapes:
            n = int(torch.Size(sh).numel())
            self.lora_views.append(self.lora_buf[start:start + n].view(sh))
            start += n
        self.lora16_by_layer, i = [], 0
        for spec in self.layers:
            row = []
            for t in (spec.lora_params()[0], spec.lora_params()[1], spec.lora_params()[4],
                      spec.lora_params()[5]):
                if t is None:
                    row.append(None)
                else:
                    row.append(self.lora_views[i])
                    i += 1
            self.lora16_by_layer.append(tuple(row))

    def set_memory_cap(self, cap_bytes):
        """The step may keep activations as long as its predicted peak stays under this."""
        self.memory_cap = cap_bytes

    def keep_plan(self, n):
        """(layers keeping [gate | up], layers keeping every intermediate) for a step of n
        tokens: [gate | up] first (its GEMM is most of the recomputation), and once every layer
        has it, the remaining intermediates (attention inputs and output, the residual after
        attention, the dropout'ed LoRA inputs) - each within the cap."""
        k_gu = self.layers_to_keep(n)
        L = len(self.layers)
        if k_gu < L or not n:
            return k_gu, 0
        spec = self.layers[0]
        cap = self.memory_cap
        left = cap - self._predicted_peak(n) - L * n * 2 * spec.I * 2
        per_layer = n * (2 * spec.H * 2 + 4 * spec.HQ * spec.D * 2 + spec.H * 4 + 4 * spec.HQ
                         + 256)
        return k_gu, max(0, min(L, left // per_layer))

    def _predicted_peak(self, n):
        """The step's peak if no layer keeps anything: the memory allocated now plus one
        recomputing layer's backward (its dequantized weights and a working set per token), or
        the streamed head if that is larger, plus the layer inputs if they stay on the GPU.

        Calibrated with the layer inputs on the GPU (they dominated the peak at the start of
        backward): measured over a whole run, 65 MiB + 218 KiB per token above the memory
        allocated when the step starts; this prediction stays 46 MiB + 14 KiB per token above it.
        With the inputs in host memory a step peaks in the forward of the layers keeping
        [gate | up]; on a T4 it stays a few MiB under the cap."""
        spec = self.layers[0]
        H, I, L = spec.H, spec.I, len(self.layers)
        # the dequantized qkv, o and [gate | up] weights a recomputing layer's backward holds
        # (down_proj is used and freed before the recomputation)
        weights = 2 * H * ((spec.HQ + 2 * spec.HKV) * spec.D + spec.HQ * spec.D + 2 * I)
        work = weights + 48 * 2**20 + n * 64 * 1024
        if self.head is not None:  # the streamed head runs between the forward and the backward
            work = max(work, self.head.peak_bytes(n))
        # the layer inputs, unless they wait in host memory (their InputStream ring is allocated
        # by then: step() begins it before the plan)
        inputs = 0 if self.inputs is not None else n * L * H * 4
        return torch.cuda.memory_allocated(self.device) + work + inputs

    def lean(self, n, extra=0):
        """Whether a step of n tokens needs the lean recomputation to stay under the cap (with
        `extra` bytes more allocated): on the longest batches the plain step's peak (one layer's
        [gate | up] output and dm on top of what the step holds) would go above it. A lean step
        keeps nothing and recomputes in two parts (_layer_forward), with [gate | up] alive for
        MLP_CHUNK tokens at a time."""
        cap = getattr(self, "memory_cap", None)
        return bool(cap) and n > self.mlp_chunk and self._predicted_peak(n) + extra > cap

    def layers_to_keep(self, n):
        """How many layers can keep their [gate | up] output (2 * I fp16 values per token) on the
        GPU instead of recomputing that GEMM in backward (or, with GuStream, instead of copying
        it to host memory and back), without the step's predicted peak reaching the cap."""
        cap = getattr(self, "memory_cap", None)
        if not cap or not n:
            return 0
        I, L = self.layers[0].I, len(self.layers)
        per_layer = n * 2 * I * 2
        return max(0, min(L, (cap - self._predicted_peak(n)) // per_layer))

    def nf4_bytes(self):
        """Bytes of the layers' packed NF4 weights and their absmax."""
        return sum(w.packed.numel() * w.packed.element_size()
                   + w.absmax.numel() * w.absmax.element_size()
                   for spec in self.layers for w in spec.w.values())

    def weights_on_host(self):
        """Whether the loader already left the NF4 weights in host memory (solution/hostload)."""
        return any(w.packed.device.type == "cpu" for spec in self.layers for w in spec.w.values())

    def offload_weights(self):
        """Keep the layers' NF4 weights in pinned host memory and stream them (WeightStream)."""
        if self.stream is None:
            self.stream = WeightStream(self.layers, self.device)
            for spec in self.layers:
                spec.stream = self.stream

    def offload_head(self):
        """Keep the tied embedding / lm_head weight in pinned host memory (HostHead): pinned where
        the loader left it (solution/offload) or copied into a pinned buffer. Only an fp16 weight
        (the one the head's GEMM reads): returns whether it is offloaded."""
        if self.head is not None:
            return True
        weight = self.embed
        if weight.dtype != torch.float16 or self.lm_w is not weight:
            return False
        if _pin_in_place(weight.data, self.device):  # where the loader put it (solution/offload)
            self.registered = weight.data
        elif not weight.data.is_pinned():  # on the GPU (or not pinnable in place): copied over
            host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
            host.copy_(weight.data)
            weight.data = host
            torch.cuda.empty_cache()  # the GPU copy goes back to the driver
        self.head = HostHead(weight, self.device)
        return True

    def offload_inputs(self):
        """Keep the layer inputs for backward in pinned host memory (InputStream)."""
        if self.inputs is None:
            self.inputs = InputStream(len(self.layers), self.device)

    def offload_gus(self):
        """Keep the [gate | up] outputs of the layers that would recompute them in backward in
        pinned host memory (GuStream)."""
        if self.gus is None:
            self.gus = GuStream(len(self.layers), self.device)

    def restore_head(self):
        """Put the embedding / lm_head weight back on the GPU."""
        if self.embed.device != self.device:
            torch.cuda.synchronize(self.device)
            self.embed.data = self.embed.data.to(self.device)
        if self.registered is not None:  # nothing reads it any more
            torch.cuda.cudart().cudaHostUnregister(self.registered.data_ptr())
            self.registered = None
        self.head = None

    def finish_training(self):
        """Training is over: free the device slots, drop the HostHead and unregister GuStream's
        host buffer. The weights stay in host memory, which is all saving a LoRA adapter needs;
        the first forward call of any of the 4-bit layers or of the embedding (if anything runs
        the model afterwards) brings them back to the GPU first."""
        self.head = None
        GuStream.pages = None  # if no step took it
        if self.gus is not None:
            self.gus.unregister()
        if self.stream is None and self.embed.device == self.device:
            return
        for spec in self.layers:
            spec.stream = None
        if self.stream is not None:
            self.stream.release_slots()
        hooks = []

        def bring_back(module, args):
            for hook in hooks:
                hook.remove()
            self.restore_weights()

        for module in self.base_modules + [self.embed_module]:
            hooks.append(module.register_forward_pre_hook(bring_back))

    def restore_weights(self):
        """Back to Soup's own code (a batch the fused step does not take): the NF4 weights and the
        embedding on the GPU again, the layer inputs' buffers and GuStream's ring freed."""
        self.restore_head()
        self.inputs = None  # its device ring and pinned buffer (offload_inputs makes them again)
        if self.gus is not None:
            self.gus.release()  # its device ring (its pinned buffers stay for the next steps)
        if self.stream is not None:
            for spec in self.layers:
                spec.stream = None
            self.stream.restore()
            self.stream = None

    def step(self, input_ids, attention_mask, labels, denom):
        """One training step's forward: (loss, packed, stats); autograd runs the backward. The
        InputStream and GuStream rings are allocated before the plan (keep_plan, lean), so the
        plan counts them; the streamed head fetches its first pieces HostHead.PREFETCH_LAYERS
        layers before the end of the forward."""
        packed = Packed(input_ids, attention_mask, labels, self.rope, self.dropout,
                        self.dropout_p)
        if self.dropout is not None and self.dropout.ok is None:
            # the step's rows, up to CHECK_TOKENS tokens each (4 x 64 x 1536 elements: every
            # one of a T4's 40960 threads draws for 2-3 iterations; 4 MiB, nothing at the peak)
            self.dropout.ok = self.dropout.check(
                packed.rows, min(packed.T, TorchDropout.CHECK_TOKENS), self.layers[0].H,
                self.dropout_p)
            if not self.dropout.ok:
                warnings.warn("solution: our LoRA dropout masks are not PyTorch's on this GPU / "
                              "build; training with the original code path")
        if self.dropout is not None and not self.dropout.ok:
            raise UnsupportedBatch("LoRA dropout masks differ from PyTorch's (TorchDropout.check)")
        if self.lora_views:
            with torch.no_grad():
                torch._foreach_copy_(self.lora_views, self.lora_fp32)
            packed.lora16_views = self.lora16_by_layer
        if self.inputs is not None:  # before the plan: its device ring counts as allocated
            self.inputs.begin(packed.n, self.layers[0].H)
            packed.inputs = self.inputs
        L = len(self.layers)
        cols = 2 * self.layers[0].I
        if self.gus is not None:
            self.gus.release()
            if self.layers_to_keep(packed.n) < L:
                # some layer would recompute its [gate | up]: a ring of whole [gate | up]
                # outputs if the step with it still needs no lean recomputation, else a ring of
                # MLP_CHUNK-token chunks (the MLP of those layers runs in chunks) if the step
                # with that one does not either; allocated before the plan, which counts it. A
                # lean step recomputes: its [gate | up] copies (every layer, 2 GB each way for
                # 2048 tokens) made a T4 instance's 4 x 512-token run 10 s slower.
                chunks = self.gus.chunk_ring_bytes(self.mlp_chunk, cols)
                if not self.lean(packed.n, self.gus.ring_bytes(packed.n, cols)):
                    self.gus.begin(packed.n, cols)
                    packed.gus = self.gus
                elif not self.lean(packed.n, chunks):
                    self.gus.begin_chunks(packed.n, cols, self.mlp_chunk)
                    packed.gus = self.gus
        k_gu, k_all = self.keep_plan(packed.n)
        packed.keep_gu_from = L - k_gu
        packed.keep_all_from = L - k_all
        packed.mlp_chunk = self.mlp_chunk
        packed.lean = k_gu == 0 and self.lean(packed.n)
        if packed.gus is not None:
            packed.gus.upto = 0 if packed.lean else packed.keep_gu_from
        head = self.head
        if head is not None:
            h = head.embed(packed.ids)
        else:
            h = torch.nn.functional.embedding(packed.ids, self.embed).float()
        prefetch_at = len(self.layers) - HostHead.PREFETCH_LAYERS
        for spec in self.layers:
            if head is not None and spec.index == prefetch_at:
                head.prefetch()
            h = _Layer.apply(h, spec, packed, *spec.lora_params())
        if packed.inputs is not None:
            packed.inputs.finish_forward()
        stats = HeadStats()
        if head is not None:
            loss = _HostHead.apply(h, self.norm_w, self.eps, head, packed, denom, stats)
            return loss, packed, stats
        lm_w = self.lm_w if self.lm_w.dtype == torch.float16 else self.lm_w.to(torch.float16)
        loss = _Head.apply(h, self.norm_w, self.eps, lm_w, packed, denom, stats)
        return loss, packed, stats


def _lora_targets(layer):
    found = {}
    for short, module in (("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj),
                          ("v", layer.self_attn.v_proj)):
        names = getattr(module, "active_adapters", None)
        if names:
            found[short] = names[0]
    return found


def _lora_input_dtype(layer, lora_names):
    """The dtype of the tensor PEFT's LoRA dropout runs on: x cast to lora_A's dtype, or under
    autocast (Soup's fp16 training) x itself, input_layernorm's output weight * x.to(residual
    dtype) - with prepare_model_for_kbit_training's fp32 norm weights, residual and LoRA weights,
    fp32 either way."""
    dtypes = {layer.input_layernorm.weight.dtype}
    for short, name in lora_names.items():
        dtypes.add(getattr(layer.self_attn, f"{short}_proj").lora_A[name].weight.dtype)
    out = dtypes.pop()
    for d in dtypes:
        out = torch.promote_types(out, d)
    return out


def _lora_dropout(layer, lora_names):
    ps = set()
    for short, name in lora_names.items():
        module = getattr(layer.self_attn, f"{short}_proj")
        drop = module.lora_dropout[name]
        ps.add(float(getattr(drop, "p", 0.0)))
    assert len(ps) <= 1
    return ps.pop() if ps else 0.0
