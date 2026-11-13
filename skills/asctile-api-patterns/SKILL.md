---
name: asctile-api-patterns
description: Current PyASC/AscTile AscTile API and kernel patterns. Use when selecting asctile APIs, implementing tiled memory access, tails, elementwise kernels, reductions, normalization, transpose, or matmul.
---

# Current PyASC/AscTile API patterns

Use the runtime selected by `reference/images/ENV.json` and the matching
source recorded in `reference/asctile/PIN.json`. For an explicitly selected
campaign, use its own source/runtime pin and keep its evidence separate.
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

For a sum reduction, the pinned API is `asctile.reduce_sum(tile, axis)`;
`asctile.sum` is not exported. Reducing a padded local tile of shape `[1, C]`
along axis `1` produces shape `[1]`, suitable for a one-element GM output.
For a contiguous vector of length `n` that fits in one padded row, declare
`x_gm = asctile.global_tensor(x_ptr, [1, n])` **before** the 2-D load.
Use `copy_in(x_gm, [0, 0], [1, width], real_shape=[1, n], pad_value=0)`
with an aligned compile-time `width >= n`. A descriptor declared with `[n]`
accepts only one offset. Reducing a rank-one local tile on its only axis
fails to lower on this pin (`Reduction along all dimensions is not supported`).
The leading singleton preserves one result per row. Preserve the output dtype
and logical reduction axis.

## Tails and padding

For a partial tile, allocate an aligned physical `shape` and pass the logical
extent through `real_shape`. The final physical dimension of a 2-D-or-higher
UB transfer must be 32-byte aligned. Padded lanes still participate in vector
math, so select an operation-neutral `pad_value`:

- add/sub/sum: `0`
- mul/prod: `1`
- divisor: `1`
- max/min reduction over floating lanes: `-inf` / `+inf` (use representable
  dtype extrema for integers)
- elementwise maximum: zero is safe when padded lanes are independent and
  the logical store excludes them
- softmax: a sufficiently negative finite value when the builtin cannot
  accept `-inf` on the route being tested

`copy_out(..., real_shape=...)` limits the logical store but does not undo
exceptions produced while computing padded lanes.

## Host/JIT boundary

- Check the caller's host signature first. For a fixed-shape task that requires
  one call with literal launch parameters, pass the declared tensors unchanged.
  For example, a two-block task with a fixed length of 96 could have just
  `kernel[2](x, out, 96)` in the host function. Derive these literals from the
  task contract when writing the code.
  Assignments, `x.numel()`, and `x.shape[0]` violate that contract even when
  they give the right number. In a general-purpose wrapper the host can
  normalize shapes, allocate outputs, and dispatch on dtypes and attributes.
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

For vector kernels, select the grid on the host after platform setup. Query
`asc.lib.runtime.device_info(DeviceModuleType.RT_MODULE_TYPE_VECTOR_CORE,
DeviceInfoType.INFO_TYPE_CORE_NUM)` using the enums from the same module.
On NPU, apply the active device and calling thread's AIV limits. The CANN 9
implementation is in `asctile-codegen-workflow/templates/kernel-template.py`
(`vector_blocks`). Bound each launch by its independent row groups or column
tiles, including any tail assigned to that launch. Skip empty work. A simulator
can report a different count from hardware even with the same requested target.

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
fastest. For a tile decision, identify the last tile’s owner and valid extent,
then estimate simultaneously live buffers. Separate estimates from compiler
measurements. Follow the [worked Maximum comparison](references/cann-performance-tuning.md#worked-example-maximum-on-three-vector-cores)
for a controlled change and a two-seed check.

### Choose a memory layout for split-half SwiGLU

When generating a complete operator on the archive-bundled public 950PR
build, implement the following dispatch in the generated module. Normalize
the split axis `d`, then derive `R = prod(shape[:d])` and
`H = (shape[d] // 2) * prod(shape[d+1:])`. A contiguous input is logically
`[R, 2*H]`. This describes the addresses for **three** physical copy patterns.
Choose the pattern before sizing its tile. A single 2D descriptor with
different tile sizes pads every short row and misses the dense-slab path.

| First applicable layout | Generated device operations | Tile and work units |
|---|---|---|
| BF16, `H * itemsize <= 32`, and an aligned multirow slab fits | One rank-one input copy of complete rows, two local `asctile.reindex` gathers, one rank-one output store. This includes H=1 and odd H. | Size T in **rows** using the [DMA and UB bounds](references/reindex-slabs.md). One work item owns up to T complete rows, including the final partial slab. |
| Several moderate half-rows fit in UB with useful work across the grid | Two strided rank-two input copies and a bounded rank-two store. Use padded physical width for allocation and logical H for DMA. | Choose rows and unroll from [row geometry](references/strided-row-tiles.md). On this build 3 KiB per half-row is a measured starting cutover to streaming. |
| Wide contiguous halves, a single long row, or too few rows to fill the grid with row tiles | Two rank-one contiguous chunk reads with source offsets H apart and a rank-one store. | T counts **elements**. There are `R * ceil(H/T)` live chunks. Keep each chunk inside its half-row and bound its tail as in the [flat template](templates/streamed-swiglu-flat.py). |

The archive-bundled CPython 3.12 evaluator uses CPU-backed Torch tensors
and the PyASC launcher transfers them for device execution. Keep the input
on its supplied device. Call `ensure_npu_platform()`, allocate
`out = input.new_empty(out_shape)`, and pass the input, output and any
cached integer indices directly to the public JIT. Calling `.to("npu")`
or importing `torch_npu` fails on this wheel. SwiGLU needs no host tensor
containing copied input halves. Launch syntax is **one integer**:
`kernel[vector_blocks(work_items)](input, out, ...)`. Compute work_items
inside the selected branch. Even a single wide row can contain many chunks.

For the row branch, the selected height T belongs in the copy itself.
Load one tile at `[r,0]` and one at `[r,H]`, each physical shape
`[T,W]` and real shape `[min(T,R-r),H]`. Write one tile to the same
valid output rectangle. A loop of T separate one-row copies loses batching.
Use `@asctile.jit(reuse_alloc=2)` with the compiled UB estimates below.
Keep `T`, `W` and `U` compile-time constants. A grid-stride
`asctile.range(block_idx, tile_count, block_count, unroll_factor=U)`
owns one tile per iteration, including the bounded final tile.

For short BF16 rows test slab eligibility first. If UB, useful row count
or DMA alignment declines the gather, try bounded rows. On other builds
check the public reindex lowering and compiled footprint. Use dispatch
from shape, dtype and runtime resources, including for unseen shapes.
For a mixed-shape task, trace at least one reachable launch of each selected
family. Check both one-element and multi-element halves in the slab branch:
a `[T,16]` copy with a one-column real shape still uses the padded row path.

### Batch short rows within the active UB budget

First check the physical layout. For contiguous, interleaved BF16 pairs
consumed together, use the [packed pair pattern](references/packed-bf16-pairs.md)
to choose a tile from active UB, give the tail one owner in the same grid,
and apply expensive unary operations to the consuming plane. On the qualified
public 950PR reindex build, dense pairs use the short-half slab gather. For native-enabled
950PR generation, select register deinterleaving and keep SwiGLU in FP32
registers through the final BF16 conversion.

For other layouts with many rows whose useful width is smaller than a 32-byte transfer, choose
the row count on the host after platform setup. On PyASC pins exposing
`asc.runtime.launcher.get_platform_info`, query
`get_platform_info(runtime.get_soc_version()).ub_size` in bytes. Use the
smaller capacity if the active device or runner imposes a lower UB limit.
Keep the row height a ConstExpr of each launched specialization.

1. Round each local operand's width to its required transfer alignment.
   Budget bytes per physical row across the simultaneously live inputs,
   promoted intermediates, outputs, scratch and buffering. Include fixed
   allocation overhead and reserve. Use the compiled allocation for the
   exact JIT options to check the estimate before accepting a height. A
   compiler can allocate more buffers than Python-level liveness predicts.
   On a UB overflow, retain the rejected height and consumed bytes. Use
   the observed per-row allocation for that kernel, dtype and physical width
   as a lower bound on the cost, then compile and validate the reduced height.
2. Compute the UB ceiling as `(ub_bytes - fixed_bytes) // bytes_per_row`.
   Bound it also by useful rows per available AIV so small inputs retain
   parallel work. For an unavailable capacity, use a separately qualified
   conservative path or report the missing platform query. If a minimum tile
   cannot fit, select a smaller qualified layout or report the budget error.
   Round down to a supported height (e.g. a power of two).
   Ensure full-tile output boundaries meet the kernel's store alignment.
   Handle a sub-tile input or final partial group with valid extents.
3. Use the chosen height for tile shape, row offsets, group count, tail
   and grid. Skip empty work. Check a short-row case at two modeled UB
   budgets and record the selected height, compiled UB and actual launches
   on hardware. Keep the tiling of wider rows independently qualified.
   For the padded CANNBench SwiGLU kernel, use the matching measured
   per-row cost in `asctile-cannbench-kernel/references/operator-patterns.md`.

AscTile dtype constraints observed by the CANNBench integration:

- Direct int8 arithmetic/cast-to-f32 is incomplete; the measured route is
  int8 -> f16 -> f32.
- A single specialization mixing active f16 and bf16 temporaries can generate
  conflicting scalar declarations. Convert each 16-bit input directly toward
  f32 and keep all arithmetic in f32.
- Compare/select destinations can overwrite to a vector-repeat boundary when
  their aligned byte size is not a multiple of 256. Prefer a 256-byte-safe
  destination shape or a counted operation such as `maximum`/`minimum` when it
  expresses the same operation. See `references/api-jit-options.md`.

## Broadcast work partition

For a leading singleton expanded to several copies of a contiguous row,
distribute **output rows** across blocks. A block can iterate
`asctile.range(asctile.block_idx(), copies, asctile.block_num())` and
process every tile of each assigned row. A tile starting at column `c`
reads source offset `c` and writes destination offset `row * row_length + c`.
If the row fits in one tile, each assigned row still needs a store. Starting
the inner source-tile loop at the block index as well skips entire rows.

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
alignment; preserve logical tails with `real_shape`. AscTile has explicit
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
2. Exact-runtime compile/lowering for every host dispatch and specialization.
3. Camodel numerics for ordinary and adversarial routes.
4. CANNBench real-NPU correctness and profiler performance.

Compile-only/QEMU evidence proves lowering and memory-budget checks, not
numerical correctness or performance.

## References

- [JIT options and measured hazards](references/api-jit-options.md)
- [Specialization and block scheduling for flat SwiGLU streams](references/streamed-halves.md)
- [Height and unroll for strided 2D row copies](references/strided-row-tiles.md)
- [DMA and UB sizing for row-slab reindex gathers](references/reindex-slabs.md)
- [Reduction tiling](references/reduction-tiling.md)
- [CANN performance techniques and evidence-driven experiments](references/cann-performance-tuning.md)
- `asctile-cannbench-kernel` for the benchmark contract and evidence labels
