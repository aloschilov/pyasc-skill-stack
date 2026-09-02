You are optimizing the PERFORMANCE of a working CANN Bench operator kernel (pyasc asc2, Ascend 950PR NPU, 72 AIV cores). You have NO NPU access — the official harness measures your file on real hardware after you finish.

Correctness is currently 100% (20/20 cases pass) and MUST stay 100% — a single failed case loses more score than any speedup can win back. Only kernel speed can improve the score now: the performance sub-score (0-50) grows with measured kernel time relative to the per-case `baseline_us` (aclnn reference) and saturates near the analytical hardware limit `t_hw_us`. Speedup = baseline_us / elapsed_us; current average is 0.399x, leaders on this hardware reach 1.0-3.4x.

# Current module — operator mish (harness score 61.13/100: compile 20.0/20, accuracy 30.0/30, performance 11.13/50)

```python
"""CANN Bench Mish interface implemented as a pyasc asc2 kernel.

mish(x) = x * tanh(softplus(x)). Computing softplus then tanh naively
loses all precision for x << 0: log(1 + e^x) rounds 1 + e^x to 1.0 once
x < -16 in f32, collapsing the output to 0 while the true value is a
small nonzero number (harness cases with value ranges like [-88, 88]
fail their f32 relative-error check).

Instead use the exact algebraic identity (w = e^(-|x|), no cancellation,
every addition operates on same-sign terms):

  tanh(softplus(x)) = (1 + 2w) / (1 + 2w + 2w^2)   for x >= 0
                    = (w^2 + 2w) / (w^2 + 2w + 2)  for x <  0

derived from tanh(log(z)) = (z^2 - 1) / (z^2 + 1) with z = 1 + e^x
(negative branch) and the same identity scaled by e^(-2x) (positive
branch). e^() never sees a positive argument, so no f32 overflow.

Kernel design follows pyasc-api-patterns Pattern A (1-D flatten,
grid-stride tile loop, ``real_shape`` tails, f32 internal compute).
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE = 1024
_MAX_CORES = 72


@asc2.jit
def _mish_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                 size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        w = asc2.exp(-asc2.abs(xf))
        w2 = w * w
        num_pos = w * 2.0 + 1.0            # 1 + 2w
        den_pos = num_pos + w2 * 2.0       # 1 + 2w + 2w^2
        num_neg = w2 + w * 2.0             # w^2 + 2w
        den_neg = num_neg + 2.0            # w^2 + 2w + 2
        tanh_sp = asc2.where(xf >= 0.0, num_pos / den_pos, num_neg / den_neg)
        y = xf * tanh_sp
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def mish(x: torch.Tensor) -> torch.Tensor:
    """Element-wise Mish activation of an NPU tensor via a pyasc asc2 kernel."""
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    num_tiles = asc.ceildiv(size, _TILE)
    cores = min(_MAX_CORES, num_tiles)
    _mish_kernel[cores](x, out, size, num_tiles, _TILE)
    return out

```

# Measured per-case kernel timings (torch_npu profiler, kernel time only)

| case | shape | dtype | elapsed_us | baseline_us | t_hw_us | speedup |
|---|---|---|---|---|---|---|
| mish_1 | [[1024, 1024]] | ['float16'] | 14.36 | 5.88 | 1.31 | 0.409x |
| mish_2 | [[2048, 2048]] | ['float32'] | 45.63 | 16.4 | 10.49 | 0.359x |
| mish_3 | [[4096, 4096]] | ['bfloat16'] | 172.28 | 52.37 | 20.97 | 0.304x |
| mish_4 | [[8192, 8192]] | ['float16'] | 689.77 | 206.72 | 83.89 | 0.3x |
| mish_5 | [[8192, 8192]] | ['float32'] | 705.93 | 388.81 | 167.77 | 0.551x |
| mish_6 | [[1023, 1023]] | ['bfloat16'] | 14.34 | 6.03 | 1.31 | 0.421x |
| mish_7 | [[1009, 1021]] | ['float16'] | 13.37 | 5.645 | 1.29 | 0.422x |
| mish_8 | [[1537, 769]] | ['float32'] | 15.12 | 7.58 | 2.96 | 0.501x |
| mish_9 | [[363, 367, 373]] | ['bfloat16'] | 508.7 | 148.8 | 62.11 | 0.293x |
| mish_10 | [[2049, 513]] | ['float16'] | 14.31 | 5.88 | 1.31 | 0.411x |
| mish_11 | [[3, 7, 13, 4001]] | ['float32'] | 14.56 | 7.19 | 2.73 | 0.494x |
| mish_12 | [[1000003]] | ['bfloat16'] | 13.61 | 5.81 | 1.25 | 0.427x |
| mish_13 | [[11, 13, 17, 67, 67]] | ['float32'] | 112.55 | 39.905 | 27.28 | 0.355x |
| mish_14 | [[3, 7, 11, 13, 1009]] | ['float16'] | 34.04 | 11.95 | 3.79 | 0.351x |
| mish_15 | [[512, 2049]] | ['float32'] | 14.6 | 7.09 | 2.62 | 0.486x |
| mish_16 | [[255, 8193]] | ['bfloat16'] | 24.81 | 9.37 | 2.61 | 0.378x |
| mish_17 | [[4097, 511]] | ['float16'] | 24.96 | 9.3 | 2.62 | 0.373x |
| mish_18 | [[2, 511, 2049]] | ['float32'] | 24.81 | 10.59 | 5.24 | 0.427x |
| mish_19 | [[4, 255, 2049]] | ['bfloat16'] | 24.92 | 9.5 | 2.61 | 0.381x |
| mish_20 | [[2, 3, 17, 1024, 101]] | ['float32'] | 109.17 | 37.59 | 26.37 | 0.344x |

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
5. mish currently runs TILE=1024 with ~14 temporaries. Reusing repeated
   subexpressions (w * 2.0 appears twice) and dropping temps may admit
   TILE=2048. The where-blend needs both branch quotients — consider a
   formulation computing numerator/denominator via one where each.

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
  Every distinct f32 tile value costs `4 * TILE` bytes, `unroll_factor=2`
  doubles the total, and the compiler adds hidden temporaries — MEASURED
  calibration: the sigmoid chain (f16 load, f32 cast, `-x`, `exp`, `+1`,
  `div`, f16 store ≈ 6 visible values) uses 155648 bytes at TILE=2048 and
  311296 (OVERFLOW) at TILE=4096, i.e. real usage ≈ 1.6x the naive
  `visible_values * 4 * TILE * 2` estimate. Budget with that 1.6x factor.
  Rule of thumb: TILE=2048 for short chains (< 8 values), 1024 for medium,
  512 for long (> 16). A launch failing with `RuntimeError: UB overflow: X
  bytes are available, Y bytes are used` means: halve TILE (do NOT drop
  cases).
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

Write the improved COMPLETE module to `candidate.py` in the current working directory. Keep the public callable name and signature EXACTLY `mish`. The numerically-stable forms in the current source exist because naive forms failed the harness — keep the math cancellation-free. Do not change behavior, only speed.

IMPORTANT — no local execution: pyasc/asc/asc2/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.
