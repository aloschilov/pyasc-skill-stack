---
name: pyasc-api-patterns
description: Current pyasc v2 AscTile API and kernel patterns. Use when selecting asctile APIs, implementing tiled memory access, tails, elementwise kernels, reductions, normalization, transpose, or matmul.
---

# Current pyasc v2 API patterns

The authoritative source is `compiler-team/pyasc` branch `v2`. The CANNBench
campaign uses commit `0a631f70968c3cb7c33ce45330a85768dd5a6f06` with
a matching runtime. The historical high-level GeLU campaign used
`adadd7d66ed0ee16d33d79487bf584899a26ef1e`; that pin is not CANNBench authority.
Other campaigns must state their own source/runtime pin;
older snapshot evidence is not automatically current.
The public tile package is `asctile`; `asc2` and `tensor/load/store` belong to
older snapshots and must not appear in new kernels.

## Surface to use

```python
from asc.experimental import asctile   # legacy source builds: import asctile

@asctile.jit
def kernel(x_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress,
           size: int, tile_size: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    for tile_id in asctile.range(asctile.block_idx(),
                                 asctile.ceildiv(size, tile_size),
                                 asctile.block_num(), unroll_factor=1):
        offset = tile_id * tile_size
        valid = min(size - offset, tile_size)
        x = asctile.copy_in(
            x_gm, [offset], [tile_size], real_shape=[valid], pad_value=0
        )
        out = asctile.exp(x)
        asctile.copy_out(out, out_gm, [offset], real_shape=[valid])
```

The exact memory signatures are:

- `global_tensor(base, shape)` creates a ranked GM descriptor.
- `copy_in(src, offsets, shape, location=..., *, real_shape=None,
  pad_value=None)` returns a local tensor. Omitting `shape` loads one scalar.
- `copy_out(src, dst, offsets, *, real_shape=None)` stores a local tensor or
  scalar.
- `copy(src, offsets=None, shape=None, location=None)` moves/slices a local
  tensor between local memory levels.

Offsets always precede shape. Swapping them —
`copy_in(g, [rows, cols], [off0, off1])` — fails with `TypeError: All
values in '2' must be integers` because the shape list lands in the offset
position (measured on pinned-runtime engram_gate_fusion).

`copy_in`/`copy_out` consume the `GlobalTensor` descriptor, never the raw
`GlobalAddress` kernel parameter. Passing the parameter directly —
`asctile.copy_in(x_ptr, ...)` — fails with `TypeError: 'src' argument must
be GlobalTensor, got GlobalAddress` (measured on pinned-runtime
conv_3d_backprop_filter). Always materialize `x_gm =
asctile.global_tensor(x_ptr, [numel])` in the kernel body first.

`asctile.concat` is variadic: `asctile.concat(t0, t1, ...)`, not
`asctile.concat([t0, t1])`. Passing one Python list fails compile with
`TypeError`-class diagnostics because the list is not a tile (measured on
pinned-runtime conv_3d_backprop_filter). Splat a built list:
`asctile.concat(*tiles)`.

Use `asctile.TensorLocation.UB/L1/L0A/L0B/L0C`, not the removed
`TileLocation` spelling. Shape rank, offsets rank, and `real_shape` rank must
match. Static local shapes must be literals or compile-time values.

## Tails and padding

For a partial tile, allocate an aligned physical `shape` and pass the logical
extent through `real_shape`. The final physical dimension of a 2-D-or-higher
UB transfer must be 32-byte aligned. Padded lanes still participate in vector
math, so select an operation-neutral `pad_value`:

- add/sub/sum: `0`
- mul/prod: `1`
- divisor: `1`
- max: `-inf`
- min: `+inf`
- softmax: a sufficiently negative finite value when the builtin cannot
  accept `-inf` on the route being tested

`copy_out(..., real_shape=...)` limits the logical store but does not undo
exceptions produced while computing padded lanes.

## Host/JIT boundary

- Host code owns shape normalization, allocation, dtype/attribute dispatch,
  metadata-only `view`/`reshape`, and kernel launches.
- Numerical work belongs inside one or more `@asctile.jit` kernels.
- Pass a `torch.Tensor`/`numpy.ndarray` directly to a JIT launch. Do not call
  `Tensor.data_ptr()`; the JIT needs pointer dtype specialization.
- Never call `asctile.global_tensor` (or `copy_in`/`copy_out`) at host level
  or pass its result across the host/JIT boundary. These calls require the
  JIT builder context and fail at launch with `RuntimeError:
  '<name>' cannot be called without initialization of global builder`.
  Declare kernel parameters as `GlobalAddress` and construct every
  `asctile.global_tensor(ptr, [shape])` inside the `@asctile.jit` kernel
  body; the launch converts the raw host tensors itself. Measured on the
  pinned runtime: host-side `global_tensor` construction fails all cases at
  dispatch.
- Use `asctile.ConstExpr[T]` for parameters that affect local shapes, compile-time
  branches, or unrolling.
- The local compile gate replays the host callable with shape/dtype-only
  tensor stubs. Data-dependent host reads such as `t.item()`,
  `int(t[0])`, or host branching on tensor values fail at dispatch with
  `AttributeError: 'FakeTensor' object has no attribute 'item'` (measured
  on pinned-runtime nms, all 20 cases, with every kernel specialization
  compiling cleanly). Prefer shape/metadata-derived host logic. A genuinely
  data-dependent read (e.g. a device-computed count that sizes the output)
  must tolerate the trace without changing real-runtime semantics:
  ```python
  try:
      m = int(cnt[0].item())      # real path: always reads the true value
  except AttributeError:          # shape-only gate replay stub only
      m = n                       # conservative shape-derived bound
  ```
  Constraints on this pattern, all mandatory:
  - Catch only the measured replay signature (`AttributeError`). Catching
    `RuntimeError`/`TypeError` would also swallow genuine runtime failures
    and silently replace the true value with the bound on hardware.
  - The fallback branch must be unreachable on the real evaluator; if it
    ever executes there, the candidate is defective — fix the read, do not
    widen the except.
  - The fallback may only supply output-shape metadata (an allocation
    bound), never a numerical value that enters computation.
  - A compile-gate pass under the fallback proves dispatch completeness
    only; it does NOT qualify the data-dependent path. CaModel numerical
    qualification on real tensors remains required and is the oracle that
    would reject any fallback leaking onto the real path.
- Launch as `kernel[core_num](...)` or, with the qualified active-device context,
  `kernel[core_num, stream](...)`. At the GeLU pin AscTile inherits public
  `LaunchOptions(core_num, stream, ...)`; stream is not a removed v1 feature.
  Use the evaluator's current stream when required; do not force a default stream.

Supported current compile options include `always_compile`, `insert_sync`,
`reuse_alloc` (`0`, `1`, `2`), `vf_fusion`, and `matmul_cube_only`. Do not use
the removed `run_asc2_passes` option. Prefer an evidenced `reuse_alloc` value;
`always_compile=True` is for development rather than a submission requirement.

The pinned JIT has a measured base-first keyword-discovery cache hazard, not
a blanket failure of call-time options. Read the exact reproducer in
[JIT options](references/api-jit-options.md). Test public decorators/overrides
in fresh processes; do not use a concrete-options adapter or monkeypatch.

## Elementwise pattern

Flatten contiguous inputs to one logical dimension, grid-stride over tiles,
promote sensitive f16/bf16 arithmetic to f32, and cast once before storing.
Compare tile sizes using measured compiled UB, useful work per available block,
DMA size, overlap, and scalar overhead. The largest fitting tile is not always
fastest; UB fill is not the objective. Read
[top-down tuning from CANN samples](references/cann-performance-tuning.md)
before a performance iteration.

Current-v2 dtype constraints observed by the CANNBench integration:

- Direct int8 arithmetic/cast-to-f32 is incomplete; the measured route is
  int8 -> f16 -> f32.
- A single specialization mixing active f16 and bf16 temporaries can generate
  conflicting scalar declarations. Convert each 16-bit input directly toward
  f32 and keep all arithmetic in f32.
- Compare/select destinations can overwrite to a vector-repeat boundary when
  their aligned byte size is not a multiple of 256. Prefer a 256-byte-safe
  destination shape or a counted operation such as `maximum`/`minimum` when it
  expresses the same operation. See `references/api-jit-options.md`.

## Reduction pattern

Flatten a last-axis reduction to `[rows, cols]`. Spread rows over the full core
grid and copy several contiguous rows per iteration when they fit UB. Do not
pad every row merely to align `cols`; that wastes bandwidth. When a physical
aligned shape is required, preserve the logical width with `real_shape` and a
correct identity pad.

Reduction results may be `PlainValue`, while tensor-only unary/binary APIs
require `LocalTensor`. Seed loop-carried accumulators from a pyasc reduction or
materialize an aligned local tensor with `full`; do not seed with a Python
float and later assign a pyasc value. For cross-core reduction use supported
atomics or per-core slots plus a second kernel. The complete tiling selector is
in `references/reduction-tiling.md`.

## Normalization and softmax

- RMSNorm: flatten leading dimensions into rows, accumulate squares in f32,
  preserve gamma dtype dispatch, and use `asctile.rms_norm` only for the shape
  and dtype routes it accepts. Current target tests may require launch-time
  `asctile.ConstExpr` wrappers even when the JIT signature is unannotated.
- Softmax: normalize negative axes on the host and view data as
  `[outer, axis_size, inner]`. The builtin is a last-axis operation. For
  `inner != 1`, tile/transpose locally or use another measured layout; never
  use a host `torch.softmax` or materialized device permutation as the
  implementation.

## Transpose

Collapse adjacent dimensions that remain adjacent under the permutation, then
select a supported rank/tiling path. Transpose axes are variadic:
`tile.transpose(*perm)` or `tile.transpose(1, 0)`. Passing a list as one axis is
invalid. Both input and output physical final dimensions need transfer
alignment; preserve logical tails with `real_shape`. Current v2 has explicit
limitations for int64 local transpose and rank-greater-than-four stores; route
those cases through word-preserving supported views and rank-collapsed tiles.

## Matmul placement

Stage reusable operands in L1, then move slices to L0A/L0B with `copy`:

```python
a_l1 = asctile.copy_in(a_gm, [0, 0], [m, k], asctile.TensorLocation.L1)
b_l1 = asctile.copy_in(b_gm, [0, 0], [k, n], asctile.TensorLocation.L1)
a_l0 = asctile.copy(a_l1, [i, 0], [m_tile, k], asctile.TensorLocation.L0A)
b_l0 = asctile.copy(b_l1, [0, j], [k, n_tile], asctile.TensorLocation.L0B)
acc = asctile.matmul(a_l0, b_l0)
out = asctile.copy(acc, location=asctile.TensorLocation.UB)
```

Budget every live L0 operand/accumulator and actual buffering against the
relevant capacity; half-capacity is only an estimate for two equal buffers.
Compile-time Python loops are appropriate for fully unrolled M/N/K tile loops;
use `asctile.range(..., unroll_factor=2)` only where software pipelining is
intended and memory permits it.

## Validation

1. `python3 -m py_compile` and static contract checks.
2. Exact-v2 compile/lowering for every host dispatch and specialization.
3. Camodel numerics for ordinary and adversarial routes.
4. CANNBench real-NPU correctness and profiler performance.

Compile-only/QEMU evidence proves lowering and memory-budget checks, not
numerical correctness or performance.

## References

- [JIT options and measured hazards](references/api-jit-options.md)
- [Reduction tiling](references/reduction-tiling.md)
- [CANN performance techniques and evidence-driven experiments](references/cann-performance-tuning.md)
- `pyasc-cannbench-kernel` for the benchmark contract and evidence labels
