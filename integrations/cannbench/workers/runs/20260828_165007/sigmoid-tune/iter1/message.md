You are optimizing the PERFORMANCE of a working CANN Bench operator kernel (pyasc asc2, Ascend 950PR NPU, 72 AIV cores). You have NO NPU access — the official harness measures your file on real hardware after you finish.

Correctness is currently 100% (20/20 cases pass) and MUST stay 100% — a single failed case loses more score than any speedup can win back. Only kernel speed can improve the score now: the performance sub-score (0-50) grows with measured kernel time relative to the per-case `baseline_us` (aclnn reference) and saturates near the analytical hardware limit `t_hw_us`. Speedup = baseline_us / elapsed_us; current average is 0.737x, leaders on this hardware reach 1.0-3.4x.

# Current module — operator sigmoid (harness score 68.65/100: compile 20.0/20, accuracy 30.0/30, performance 18.65/50)

```python
"""CANN Bench Sigmoid interface implemented as a pyasc asc2 kernel.

Kernel design (pyasc-api-patterns Pattern A, grid-stride variant):
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with ``real_shape`` loads/stores (no host padding).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision
    (the spec's precision standard expects f32 internal compute).
  - sigmoid(x) = 1 / (1 + e^(-x)); IEEE saturation gives the correct
    limits at extreme inputs (e^inf -> inf -> y=0, e^-inf -> 0 -> y=1).
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE = 2048
_MAX_CORES = 72


@asc2.jit
def _sigmoid_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                    size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        y = asc2.div(1.0, asc2.exp(-xf) + 1.0)
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Element-wise sigmoid of an NPU tensor via a pyasc asc2 kernel."""
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    num_tiles = asc.ceildiv(size, _TILE)
    cores = min(_MAX_CORES, num_tiles)
    _sigmoid_kernel[cores](x, out, size, num_tiles, _TILE)
    return out

```

# Measured per-case kernel timings (torch_npu profiler, kernel time only)

| case | shape | dtype | elapsed_us | baseline_us | t_hw_us | speedup |
|---|---|---|---|---|---|---|
| sigmoid_1 | [[1024, 1024]] | ['float16'] | 6.57 | 4.25 | 1.31 | 0.647x |
| sigmoid_2 | [[2048, 2048]] | ['float32'] | 18.09 | 15.23 | 10.49 | 0.842x |
| sigmoid_3 | [[4096, 4096]] | ['bfloat16'] | 62.39 | 29.635 | 20.97 | 0.475x |
| sigmoid_4 | [[8192, 8192]] | ['float16'] | 258.52 | 168.32 | 83.89 | 0.651x |
| sigmoid_5 | [[8192, 8192]] | ['float32'] | 365.7 | 387.715 | 167.77 | 1.06x |
| sigmoid_6 | [[1023, 1023]] | ['bfloat16'] | 6.64 | 4.51 | 1.31 | 0.679x |
| sigmoid_7 | [[1009, 1021]] | ['float16'] | 6.77 | 4.41 | 1.29 | 0.651x |
| sigmoid_8 | [[1537, 769]] | ['float32'] | 7.04 | 6.42 | 2.95 | 0.912x |
| sigmoid_9 | [[363, 367, 373]] | ['bfloat16'] | 187.61 | 113.67 | 62.11 | 0.606x |
| sigmoid_10 | [[2049, 513]] | ['float16'] | 7.03 | 4.41 | 1.31 | 0.627x |
| sigmoid_11 | [[3, 7, 13, 4001]] | ['float32'] | 6.71 | 6.16 | 2.73 | 0.918x |
| sigmoid_12 | [[1000003]] | ['bfloat16'] | 7.04 | 4.36 | 1.25 | 0.619x |
| sigmoid_13 | [[11, 13, 17, 67, 67]] | ['float32'] | 42.1 | 37.825 | 27.28 | 0.898x |
| sigmoid_14 | [[3, 7, 11, 13, 1009]] | ['float16'] | 13.94 | 8.32 | 3.79 | 0.597x |
| sigmoid_15 | [[512, 2049]] | ['float32'] | 6.86 | 6.04 | 2.62 | 0.88x |
| sigmoid_16 | [[255, 8193]] | ['bfloat16'] | 10.58 | 6.83 | 2.61 | 0.646x |
| sigmoid_17 | [[4097, 511]] | ['float16'] | 10.67 | 6.66 | 2.62 | 0.624x |
| sigmoid_18 | [[2, 511, 2049]] | ['float32'] | 10.47 | 8.97 | 5.24 | 0.857x |
| sigmoid_19 | [[4, 255, 2049]] | ['bfloat16'] | 10.72 | 6.79 | 2.61 | 0.633x |
| sigmoid_20 | [[2, 3, 17, 1024, 101]] | ['float32'] | 40.06 | 36.74 | 26.37 | 0.917x |

# Known optimization levers for this hardware (priority order, all previously measured)

1. **Wider tiles** amortize per-tile DMA/loop setup: TILE=2048 measured ~0.9x
   of hand-written AscendC for a simple elementwise op, vs ~0.2x at TILE=128.
   Constraint: UB budget — roughly `num_f32_temporaries * 4 * TILE * 2 <= 250000`
   bytes (the trailing *2 is unroll double-buffering).
2. **Fewer tile temporaries** (algebraic simplification / fusing constants)
   directly buys a wider tile. This is the main lever for long op chains.
3. **Small-case parallelism**: `cores = min(72, num_tiles)` means small shapes
   underuse the 72 cores when TILE is wide. Consider selecting between two
   compiled tile variants at the host by size (e.g. wide tile when
   `numel >= 72 * WIDE_TILE`, narrow otherwise). Look at the per-case table:
   cases with small `numel` and low speedup are parallelism-starved.
4. `unroll_factor=2` on the grid-stride loop is already present — keep it.
5. sigmoid's chain is short and TILE is already 2048; check whether
   TILE=4096 still fits UB (temps * 4 * TILE * 2 <= ~250000), and whether a
   narrow-tile variant for small cases (see table) recovers parallelism.

# pyasc asc2 kernel contract (follow EXACTLY — every rule below was learned from real failures on this hardware)

## Module shape

Your file becomes `cann_bench/<module>.py` inside the submission wheel. It must contain:

- imports at module top: `import torch`, `import asc`, `import asc2`,
  `from ._pyasc_runtime import ensure_npu_platform` (and `import math` if needed)
- one or more `@asc2.jit` kernel functions
- ONE public callable matching the operator schema exactly (name and signature)
- wrapper body: call `ensure_npu_platform()` first; make inputs contiguous if
  needed (`x = x.contiguous()` is allowed); allocate outputs with
  `torch.empty_like(x)` or `torch.empty(shape, dtype=..., device=x.device)`;
  launch `kernel[cores](tensor_args..., int_args..., float_args..., constexpr_args...)`;
  return contiguous NPU tensor(s)

## Kernel authoring rules

- Global memory views: `asc2.global_tensor(ptr, [size])` (1-D) or
  `asc2.global_tensor(ptr, [rows, cols])` (2-D). Ranks of global_tensor /
  copy_in / copy_out / offsets must ALL match — never mix 1-D and 2-D.
- Kernel params: pointers typed `asc.GlobalAddress`; sizes as plain `int`
  (runtime); tile sizes as `asc.ConstExpr[int]` (compile-time; REQUIRED for any
  value used inside a copy_in tile shape); scalars as `float`.
- Grid-stride tile loop (the proven pattern):

```python
for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):
    off = t * tile_size
    n = tile_size if off + tile_size <= size else size - off   # tail handling
    x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
    ...compute on tiles...
    asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])
```

- Launch: `kernel[cores](...)` with `cores = min(72, num_tiles)` (72 AIV cores
  on this 950PR box). No stream argument.
- Available tile ops: `+ - * /` (tile-tile and tile-scalar), `asc2.abs`,
  `asc2.exp`, `asc2.log`, `asc2.sqrt`, `asc2.tanh`, `asc2.erf`, `asc2.sin`,
  `asc2.cos`, `asc2.maximum`, `asc2.minimum`, comparisons
  (`x >= 0.0`, `asc2.less(a, b)`, ... — NO int64 operands),
  `asc2.where(cond, a, b)`, `asc2.reduce_sum(x)`, `asc2.reduce_max(x)`,
  `asc2.full([shape], scalar, dtype=...)`, `tile.to(asc.float32)` casts,
  unary `-x`.
- Scalars go on the RIGHT of tile arithmetic (Tile has no `__rmul__`):
  write `x * 0.5`, NEVER `0.5 * x`. Same for `+ - /`.
- f16/bf16 inputs: promote to f32 in-kernel (`xf = x.to(asc.float32)`),
  compute in f32, cast back on copy_out (`y.to(x.dtype)`).
- UB (unified buffer) budget: ~253952 bytes total under static allocation.
  Every distinct f32 tile temporary costs `4 * TILE` bytes and
  `unroll_factor=2` roughly doubles the total. Rule of thumb: TILE=2048 for
  short chains (< 8 temporaries), 1024 for medium, 512 for long (> 16).
  A launch failing with `RuntimeError: UB overflow: X bytes are available,
  Y bytes are used` means: halve TILE (do NOT drop cases).
- `asc2.where` / comparison destination tiles must be a multiple of 256 bytes
  (`TILE * 4 % 256 == 0` for f32 — any TILE >= 64 is safe).
- Loop-carried scalar accumulators: seed with
  `acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))`
  (a bare `acc = 0.0` fails codegen with "re-assigned to an object with
  different type"), and the accumulating loop must pass `gm_barrier=True`.
- Scalar reduction results must be widened before store:
  `asc2.copy_out(asc2.full([8], s, dtype=...), out_gm, [0], real_shape=[8])`
  style (min 32 bytes).
- Inside `@asc2.jit`: NO `print`, NO imports, NO `break`/`continue`/early
  `return`, NO exceptions, NO Python `range()` over runtime values (use
  `asc2.range`), NO `math.*` calls (precompute module-level constants).

## Numerical stability (MANDATORY — f32 cases use ranges like [-88, 88] and [-100, 100] under a ~1.2e-4 relative-error threshold)

- Never let `exp()` see a positive argument that can overflow; never subtract
  nearly-equal quantities (catastrophic cancellation); never rely on
  `log(1 + tiny)` (flushes to 0 below tiny < 6e-8).
- Proven cancellation-free building blocks (all verified on this harness):
  - `sigmoid(s) = exp(min(s, 0)) / (1 + exp(-|s|))`
  - `1 + tanh(u) = 2 * sigmoid(2u)`
  - `tanh(softplus(x))`: with `w = exp(-|x|)`, equals
    `(1 + 2w) / (1 + 2w + 2w^2)` for `x >= 0`, `(w^2 + 2w) / (w^2 + 2w + 2)`
    for `x < 0` (exact identities; blend with `asc2.where(xf >= 0.0, ...)`)
  - `erfc(z)` for `z >= 0`: Numerical Recipes fit `t * exp(-z*z + P(t))`,
    `t = 1/(1 + z/2)`, rel. err < 1.2e-7 (see the gelu reference module for
    the 9-coefficient Horner chain)
- IEEE special values (inf/nan scalars or extreme inputs) propagate correctly
  through the hardware ops — do NOT special-case them with host branches
  unless the golden does.

## Anti-cheat (violations zero the submission)

- ALL numerical work happens inside `@asc2.jit` kernels launched on the NPU.
- torch usage is allowed ONLY for: output allocation (`torch.empty`,
  `torch.empty_like`), metadata (`.shape`, `.numel()`, `.stride()`, `.dtype`,
  `.is_contiguous()`), contiguity (`.contiguous()`), and views (`.view`,
  `.reshape`, `.narrow`, indexing that returns a view).
- FORBIDDEN anywhere in the module: torch math/compute ops (`torch.mul`,
  `torch.norm`, `torch.nn.functional.*`, tensor arithmetic like `a + b`,
  `x.sigmoid()`, `.to(dtype)` casts of device data, `torch.cat`,
  `torch.clone`, `torch.sum`, ...). The harness hooks torch dispatch and
  rotates input data pointers between calls — caching outputs by `data_ptr`
  is detected and scored as cheating.
- Outputs must be contiguous NPU tensors with exactly the golden's
  shape/dtype. Do not return views of inputs.


# Deliverable

Write the improved COMPLETE module to `candidate.py` in the current working directory. Keep the public callable name and signature EXACTLY `sigmoid`. The numerically-stable forms in the current source exist because naive forms failed the harness — keep the math cancellation-free. Do not change behavior, only speed.
