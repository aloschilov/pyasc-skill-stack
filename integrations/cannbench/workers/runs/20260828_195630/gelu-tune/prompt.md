You are optimizing the PERFORMANCE of a working CANN Bench operator kernel (pyasc asc2, Ascend 950PR NPU, 72 AIV cores). You have NO NPU access — the official harness measures your file on real hardware after you finish.

Correctness is currently 100% (20/20 cases pass) and MUST stay 100% — a single failed case loses more score than any speedup can win back. Only kernel speed can improve the score now: the performance sub-score (0-50) grows with measured kernel time relative to the per-case `baseline_us` (aclnn reference) and saturates near the analytical hardware limit `t_hw_us`. Speedup = baseline_us / elapsed_us; current average is 0.272x, leaders on this hardware reach 1.0-3.4x.

# Current module — operator gelu (harness score 57.19/100: compile 20.0/20, accuracy 30.0/30, performance 7.19/50)

```python
"""CANN Bench Gelu interface implemented as pyasc asc2 kernels.

Two modes per the spec (proto.yaml attr ``approximate``):
  - "none": y = x * 0.5 * (1 + erf(x / sqrt(2)))
  - "tanh": y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))

Both naive forms cancel catastrophically for x << 0: ``1 + erf(v)`` and
``1 + tanh(u)`` round to 0 in f32 once x < ~-5.5, while the true output
is small but nonzero, failing the harness f32 relative-error check on
wide value ranges (e.g. [-88, 88]).

Stable reformulations used here:
  - erf mode: 1 + erf(v) = 2 - erfc(|v|) for v >= 0 and erfc(|v|) for
    v < 0, with erfc computed by the Numerical Recipes rational-
    Chebyshev fit  erfc(z) = t * exp(-z^2 + P(t)), t = 1/(1 + z/2)
    (fractional error < 1.2e-7 for all z >= 0). No cancellation in
    either branch.
  - tanh mode: 1 + tanh(u) = 2*sigmoid(2u), and sigmoid(s) is computed
    in the branch-free stable form e^min(s,0) / (1 + e^-|s|), so e^()
    never sees a positive argument.

One kernel per mode (the mode is a compile-time choice, not a runtime
value). Kernel design follows pyasc-api-patterns Pattern A (1-D flatten,
grid-stride tile loop, ``real_shape`` tails, f32 internal compute).
Tiles are always the LEFT operand of arithmetic (asc2 Tile has no
``__rmul__``).
"""

import math

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE_TANH = 1024
# The erf kernel's Horner chain allocates ~2x more UB temporaries under
# static allocation; tile 1024 overflows (409984 > 253952 bytes), 512 fits.
_TILE_ERF = 512
_MAX_CORES = 72

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_TWO_SQRT_2_OVER_PI = 2.0 * math.sqrt(2.0 / math.pi)
_GELU_C = 0.044715


@asc2.jit
def _gelu_erf_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                     size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        v = xf * _INV_SQRT2
        z = asc2.abs(v)
        den = z * 0.5 + 1.0
        tt = asc2.full([tile_size], 1.0, dtype=asc.float32) / den
        # Numerical Recipes erfc Chebyshev fit, Horner form in tt.
        p = tt * 0.17087277 - 0.82215223
        p = p * tt + 1.48851587
        p = p * tt - 1.13520398
        p = p * tt + 0.27886807
        p = p * tt - 0.18628806
        p = p * tt + 0.09678418
        p = p * tt + 0.37409196
        p = p * tt + 1.00002368
        p = p * tt - 1.26551223
        erfc_z = tt * asc2.exp(p - z * z)          # erfc(|v|), rel err ~1e-7
        one_plus_erf = asc2.where(xf >= 0.0, erfc_z * -1.0 + 2.0, erfc_z)
        y = xf * one_plus_erf * 0.5
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


@asc2.jit
def _gelu_tanh_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        s = (xf * xf * xf * _GELU_C + xf) * _TWO_SQRT_2_OVER_PI   # 2u
        # y = x * sigmoid(s), stable: e^min(s,0) / (1 + e^-|s|)
        sig = asc2.exp(asc2.minimum(s, 0.0)) / (asc2.exp(-asc2.abs(s)) + 1.0)
        y = xf * sig
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Element-wise GELU of an NPU tensor via pyasc asc2 kernels."""
    ensure_npu_platform()
    if approximate not in ("none", "tanh"):
        raise ValueError(
            f"approximate must be 'none' or 'tanh', got {approximate!r}")
    if approximate == "none":
        kernel, tile = _gelu_erf_kernel, _TILE_ERF
    else:
        kernel, tile = _gelu_tanh_kernel, _TILE_TANH
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    num_tiles = asc.ceildiv(size, tile)
    cores = min(_MAX_CORES, num_tiles)
    kernel[cores](x, out, size, num_tiles, tile)
    return out

```

# Measured per-case kernel timings (torch_npu profiler, kernel time only)

| case | shape | dtype | elapsed_us | baseline_us | t_hw_us | speedup |
|---|---|---|---|---|---|---|
| gelu_1 | [[1024, 1024]] | ['float16'] | 27.45 | 4.49 | 1.31 | 0.164x |
| gelu_2 | [[2048, 2048]] | ['float32'] | 102.6 | 15.37 | 10.49 | 0.15x |
| gelu_3 | [[4096, 4096]] | ['bfloat16'] | 380.81 | 30.14 | 20.97 | 0.079x |
| gelu_4 | [[8192, 8192]] | ['float16'] | 522.41 | 172.93 | 83.89 | 0.331x |
| gelu_5 | [[8192, 8192]] | ['float32'] | 549.39 | 387.265 | 167.77 | 0.705x |
| gelu_6 | [[1023, 1023]] | ['bfloat16'] | 11.0 | 4.46 | 1.31 | 0.405x |
| gelu_7 | [[1009, 1021]] | ['float16'] | 26.57 | 4.45 | 1.29 | 0.167x |
| gelu_8 | [[1537, 769]] | ['float32'] | 11.43 | 6.11 | 2.95 | 0.535x |
| gelu_9 | [[363, 367, 373]] | ['bfloat16'] | 1133.23 | 117.53 | 62.11 | 0.104x |
| gelu_10 | [[2049, 513]] | ['float16'] | 10.89 | 4.56 | 1.31 | 0.419x |
| gelu_11 | [[3, 7, 13, 4001]] | ['float32'] | 29.6 | 6.05 | 2.73 | 0.204x |
| gelu_12 | [[1000003]] | ['bfloat16'] | 10.37 | 4.38 | 1.25 | 0.422x |
| gelu_13 | [[11, 13, 17, 67, 67]] | ['float32'] | 260.75 | 37.455 | 27.28 | 0.144x |
| gelu_14 | [[3, 7, 11, 13, 1009]] | ['float16'] | 25.55 | 7.78 | 3.79 | 0.305x |
| gelu_15 | [[512, 2049]] | ['float32'] | 28.62 | 5.98 | 2.62 | 0.209x |
| gelu_16 | [[255, 8193]] | ['bfloat16'] | 51.32 | 6.23 | 2.61 | 0.121x |
| gelu_17 | [[4097, 511]] | ['float16'] | 18.71 | 6.31 | 2.62 | 0.337x |
| gelu_18 | [[2, 511, 2049]] | ['float32'] | 53.04 | 8.86 | 5.24 | 0.167x |
| gelu_19 | [[4, 255, 2049]] | ['bfloat16'] | 18.61 | 6.26 | 2.61 | 0.336x |
| gelu_20 | [[2, 3, 17, 1024, 101]] | ['float32'] | 253.37 | 36.195 | 26.37 | 0.143x |

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
5. The erf-mode kernel's 9-step Horner chain forces TILE=512 (each Horner
   step is a fresh tile temporary under static allocation). Reducing
   temporaries (e.g. fewer polynomial steps that still meet ~1e-5 relative
   accuracy, or restructuring so p is reused in place) buys TILE=1024+.
   The tanh-mode kernel is short and could run TILE=2048 independently —
   the two kernels need not share a tile size.

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

Write the improved COMPLETE module to `candidate.py` in the current working directory. Keep the public callable name and signature EXACTLY `gelu`. The numerically-stable forms in the current source exist because naive forms failed the harness — keep the math cancellation-free. Do not change behavior, only speed.

IMPORTANT — no local execution: pyasc/asc/asc2/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.
