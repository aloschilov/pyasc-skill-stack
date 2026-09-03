# GELU — Design Document

## 1. Overview

Single operator `gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor`.
Elementwise activation. Output shape == input shape, output dtype == input dtype.

Two mutually exclusive modes dispatched at the host level:

| Mode | Formula |
|------|---------|
| `"none"` | `y = x * 0.5 * (1 + erf(x / sqrt(2)))` |
| `"tanh"` | `y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` |

Runtime pin: **pyasc v2 commit `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`**, module name **`asctile`** (not `asc2`).

---

## 2. Module Shape

```
imports:        torch, asc, asctile, math, _pyasc_runtime.ensure_npu_platform
constants:      _INV_SQRT2, _TWO_SQRT_2_OVER_PI, _GELU_C, _GC_T
kernels:        _gelu_erf_kernel   (unroll_factor=2)
                _gelu_erf_kernel_u1 (unroll_factor=1)
                _gelu_tanh_kernel  (unroll_factor=2)
public callable: gelu(x, approximate="none")
```

All constants precomputed at module level with `math.*` (forbidden inside `@asctile.jit`).

| Constant | Value |
|----------|-------|
| `_INV_SQRT2` | `1.0 / math.sqrt(2.0)` = `0.7071067811865476` |
| `_TWO_SQRT_2_OVER_PI` | `2.0 * math.sqrt(2.0 / math.pi)` = `1.5957691216057308` |
| `_GELU_C` | `0.044715` |
| `_GC_T` | `_GELU_C * _TWO_SQRT_2_OVER_PI` = `0.07135278842249998` |
| `_MAX_CORES` | `72` |

---

## 3. Case Dispatch Matrix

All 20 evaluation cases grouped by mode and dtype:

| case | shape | dtype | value range | mode | numel |
|------|-------|-------|-------------|------|-------|
| 1 | [1024,1024] | f16 | [-1,1] | none | 1048576 |
| 2 | [2048,2048] | f32 | [-2,2] | none | 4194304 |
| 3 | [4096,4096] | bf16 | [-3,3] | none | 16777216 |
| 4 | [8192,8192] | f16 | [-10,10] | tanh | 67108864 |
| 5 | [8192,8192] | f32 | [-100,100] | tanh | 67108864 |
| 6 | [1023,1023] | bf16 | [-0.1,0.1] | tanh | 1046529 |
| 7 | [1009,1021] | f16 | [-1,2] | none | 1030189 |
| 8 | [1537,769] | f32 | [-5,10] | tanh | 1181953 |
| 9 | [363,367,373] | bf16 | [-50,100] | none | 49792167 |
| 10 | [2049,513] | f16 | [-65504,65504] | tanh | 1051137 |
| 11 | [3,7,13,4001] | f32 | [-88,88] | none | 1092273 |
| 12 | [1000003] | bf16 | [-inf,inf] | tanh | 1000003 |
| 13 | [11,13,17,67,67] | f32 | [nan,nan] | none | 1619573 |
| 14 | [3,7,11,13,1009] | f16 | [0,0] | tanh | 3060307 |
| 15 | [512,2049] | f32 | [-0.5,0.5] | none | 1049088 |
| 16 | [255,8193] | bf16 | [-1,3] | none | 2089215 |
| 17 | [4097,511] | f16 | [-1000,1000] | tanh | 2093567 |
| 18 | [2,511,2049] | f32 | [-0.2,0.2] | none | 2093058 |
| 19 | [4,255,2049] | bf16 | [-3,6] | tanh | 2090940 |
| 20 | [2,3,17,1024,101] | f32 | [-20,40] | none | 10747904 |

Mode split: 12 "none" cases, 8 "tanh" cases.

---

## 4. erf Mode Kernel (`approximate == "none"`)

### 4.1 Why NOT `asctile.erf` directly

Naive form: `y = x * 0.5 * (1 + erf(x/sqrt(2)))`

Catastrophic cancellation when `x < 0` and `|x| > ~5.66`:
- `erf(x/sqrt(2))` saturates toward `-1`
- `1 + erf(...)` flushes to 0 in f32 while the true result is `erfc(|x|/sqrt(2))`, a tiny but nonzero value
- Golden (`torch.nn.functional.gelu`) computes `x * 0.5 * erfc(|x|/sqrt(2))` which is nonzero
- Relative error blows up, failing the f32 MERE/MARE threshold (2^-13 ≈ 1.22e-4)

Affected test cases:
- Case 11: [-88, 88] f32 — values near x ≈ -6 produce the worst cancellation
- Case 20: [-20, 40] f32 — values near x ≈ -6 produce cancellation
- Case 2: [-2, 2] f32 — safe (|x|/sqrt(2) ≤ 1.41, no saturation)
- Case 13: [nan, nan] f32 — nan propagates through erf, no cancellation issue
- Case 15, 18: small magnitude — safe

### 4.2 Cancellation-free reformulation (Numerical Recipes erfc)

For `z = |x| / sqrt(2) >= 0`, compute `erfc(z)` via the 9-coefficient Horner rational fit:

```
t = 1 / (1 + z/2)
p = Horner(t) with 9 coefficients
erfc(z) = t * exp(p - z^2)
```

Relative error of this fit < 1.2e-7 for all `z >= 0`.

**Horner coefficients** (in evaluation order from innermost to outermost):

| Coefficient | Index |
|-------------|-------|
| `0.17087277` | first multiply |
| `-0.82215223` | add |
| `+1.48851587` | |
| `-1.13520398` | |
| `+0.27886807` | |
| `-0.18628806` | |
| `+0.09678418` | |
| `+0.37409196` | |
| `+1.00002368` | |
| `-1.26551223` | final add |

Evaluation (all tile operations, scalars on RIGHT):

```python
z = asctile.abs(xf) * _INV_SQRT2
den = z * 0.5 + 1.0
one_tile = asctile.full([tile_size], 1.0, dtype=asc.float32)
tt = one_tile / den
p = tt * 0.17087277 - 0.82215223
p = p * tt + 1.48851587
p = p * tt - 1.13520398
p = p * tt + 0.27886807
p = p * tt - 0.18628806
p = p * tt + 0.09678418
p = p * tt + 0.37409196
p = p * tt + 1.00002368
p = p * tt - 1.26551223
erfc_z = tt * asctile.exp(p - z * z)
```

**Critical**: `1.0 / den` fails because Python `float.__truediv__(Tile)` returns NotImplemented. Must use `asctile.full` to create a tile of 1.0 first, yielding tile/tile.

### 4.3 Cancellation-free output assembly

```python
half_erfc = erfc_z * 0.5
y_neg = xf * half_erfc           # x < 0 branch: y = x * 0.5 * erfc(|x|/sqrt(2))
y_pos = xf - y_neg               # x >= 0 branch: y = x - x * 0.5 * erfc  (safe: 0.5*erfc in [0, 0.5])
y = asctile.where(xf >= 0.0, y_pos, y_neg)
```

Why this is cancellation-free:
- **x >= 0**: `y = x * 0.5 * (2 - erfc) = x - x*0.5*erfc`. Since `0.5*erfc ∈ [0, 0.5]`, the subtraction `x - small_fraction_of_x` stays in `[0.5x, x]`. No cancellation.
- **x < 0**: `y = x * 0.5 * erfc`. Product of two finite values. No cancellation.

`asctile.where` constraint: both data branches must be LocalTensor (not Python scalar). `y_pos` and `y_neg` are both tiles.

### 4.4 UB budget — erf mode

The Horner chain produces ~12 SSA f32 tile values that remain live during static allocation. Measured on the pinned v2 compiler:

| tile_size | unroll_factor | UB used | Limit | Fits? |
|-----------|---------------|---------|-------|-------|
| 512 | 2 | ~95-105 KB | 253952 B (~248 KB) | YES (verified) |
| 1024 | 1 | ~170-190 KB | 253952 B | YES (verified by canonical) |
| 1024 | 2 | ~796-820 KB | 253952 B | **NO** — rejected |
| 2048 | any | overflow | 253952 B | **NO** |

**Strategy**: Two compiled erf kernels:
- `_gelu_erf_kernel`: `unroll_factor=2`, `tile_size=512` — safe fallback, always fits
- `_gelu_erf_kernel_u1`: `unroll_factor=1`, `tile_size=1024` — wider tile that halving UB doubles enables larger tile

Host tries wide variant first (TILE=1024, unroll=1), catches `RuntimeError` (UB overflow), falls back to safe (TILE=512, unroll=2). Caches result in module-level flag `_erf_wide_ok`.

---

## 5. tanh Mode Kernel (`approximate == "tanh"`)

### 5.1 Why NOT `1 + tanh(u)` directly

Naive form: `y = 0.5 * x * (1 + tanh(u))`, `u = sqrt(2/pi) * (x + 0.044715 * x^3)`

Same catastrophic cancellation as erf mode. When `u` is moderately large negative (e.g., `x = -4` → `u ≈ -5.47`):
- `tanh(u) ≈ -0.99996` in f32
- `1 + tanh(u)` flushes near 0, but golden is nonzero
- Relative error exceeds f32 threshold 2^-13 ≈ 1.22e-4

Affected test cases:
- Case 5: [-100, 100] f32 — x ≈ -4 to -7 range problematic
- Case 10: [-65504, 65504] f16 — f16 threshold 2^-10 ≈ 9.77e-4, less tight but still risky
- Case 12: [-inf, inf] bf16 — bf16 threshold 2^-7 ≈ 7.81e-3, lenient
- Case 17: [-1000, 1000] f16 — wide range

### 5.2 Cancellation-free reformulation (stable sigmoid)

Identity: `1 + tanh(u) = 2 * sigmoid(2u)`

Let `s = 2u = 2 * sqrt(2/pi) * (x + 0.044715 * x^3)`:

```
sigmoid(s) = exp(min(s, 0)) / (1 + exp(-|s|))
y = x * sigmoid(s)
```

This is cancellation-free because:
- `min(s, 0)` ensures `exp()` never sees a positive argument (no overflow)
- `exp(-|s|) ∈ [0, 1]`, so denominator ∈ [1, 2] (no underflow, no cancellation)
- `numerator ∈ [0, 1]` (bounded)
- The subtraction in `1 + tanh(u) ≈ 0` is replaced by `exp(large_negative) / ~1 ≈ 0`, which is exact in f32

**Factored cubic** (saves one tile-scalar multiply):

```python
x2 = xf * xf
s = (x2 * _GC_T + _TWO_SQRT_2_OVER_PI) * xf
```

Where `_GC_T = _GELU_C * _TWO_SQRT_2_OVER_PI`, so:
```
s = (_GELU_C * _TWO_SQRT_2_OVER_PI * x^2 + _TWO_SQRT_2_OVER_PI) * x
  = _TWO_SQRT_2_OVER_PI * (x + _GELU_C * x^3)
  = 2u
```

The tanh kernel code:

```python
xf = x.to(asc.float32)
x2 = xf * xf
s = (x2 * _GC_T + _TWO_SQRT_2_OVER_PI) * xf
sig = asctile.exp(asctile.minimum(s, 0.0)) / (asctile.exp(-asctile.abs(s)) + 1.0)
y = xf * sig
asctile.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])
```

`asctile.minimum(s, 0.0)` — tile-scalar minimum, supported.
`asctile.abs(s)` — tile unary abs, supported for f32.
`-asctile.abs(s)` — tile unary negation, supported.

### 5.3 UB budget — tanh mode

Visible tile values:
1. `xf` (promoted input, needed for final multiply)
2. `x2 = xf * xf`
3. `s = (x2 * _GC_T + k) * xf` — `x2` dies after
4. `abs_s = abs(s)` — transient
5. `neg_abs_s = -abs(s)` — transient
6. `exp_neg_abs = exp(neg_abs_s)` — needed for denominator
7. `den = exp_neg_abs + 1.0` — needed for division
8. `min_s = minimum(s, 0.0)` — `s` may die after
9. `exp_min = exp(min_s)` — numerator
10. `sig = exp_min / den` — needed for y
11. `y = xf * sig`

Peak concurrent tiles ~5-6 f32 tiles + hidden compiler temporaries.

| tile_size | unroll_factor | UB used | Limit | Fits? |
|-----------|---------------|---------|-------|-------|
| 1024 | 2 | ~172-184 KB | 253952 B | YES (verified) |
| 1536 | 2 | ~256-276 KB | 253952 B | marginal, try first |
| 2048 | 2 | ~344-368 KB | 253952 B | **NO** — overflow |

**Strategy**: Try `tile_size=1536` first, fall back to `tile_size=1024` on UB overflow. Cache in `_tanh_tile`.

---

## 6. Tile Size Selection (Summary)

| Mode | Kernel | unroll | tile_size | Role |
|------|--------|--------|-----------|------|
| none | `_gelu_erf_kernel_u1` | 1 | 1024 | wide — tried first |
| none | `_gelu_erf_kernel` | 2 | 512 | safe — fallback |
| tanh | `_gelu_tanh_kernel` | 2 | 1536 | wide — tried first |
| tanh | `_gelu_tanh_kernel` | 2 | 1024 | safe — fallback |

---

## 7. Grid-Stride Tile Loop Pattern

All three kernels use the identical grid-stride loop:

```python
x_gm = asctile.global_tensor(x_ptr, [size])
out_gm = asctile.global_tensor(out_ptr, [size])
for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                       unroll_factor=U):
    off = t * tile_size
    n = tile_size if off + tile_size <= size else size - off
    x_t = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
    # ... compute on x_t ...
    asctile.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])
```

1-D flatten: any rank input → `size = x.numel()`, 1-D global tensor views.
`real_shape=[n]` handles tails — only `n` elements participate in DMA and vector arithmetic.

---

## 8. Tail Handling

Non-aligned tails (case 7: 1030189 elements, case 8: 1181953, etc.):
- `n = size - off` for the last tile
- `real_shape=[n]` limits both load/stores
- Padded lanes (indices `n..tile_size-1`) are NOT copied back by `copy_out` with `real_shape`, but they DO participate in vector arithmetic during kernel execution
- Default `pad_value=0` for `copy_in` is safe here: erf mode uses `abs(z)` → `z=0` gives `t=1`, `erfc(0)=1`, output `y=0*0.5*1=0` (neutral). tanh mode: `s` computed on padding values produces a finite `sig`, output 0. No division-by-zero or log-of-zero hazards.

---

## 9. Dtype Handling

All compute happens in f32 inside the kernel:

```python
xf = x.to(asc.float32)       # promote f16/bf16 → f32 (f32 is a no-op cast)
# ... all arithmetic on xf, f32 tiles ...
y.to(x.dtype)                # cast back to original dtype
```

| Input dtype | Promotion | Compute | Cast back | Output dtype |
|-------------|-----------|---------|-----------|--------------|
| float16 | f16 → f32 | f32 | f32 → f16 | float16 |
| bfloat16 | bf16 → f32 | f32 | f32 → bf16 | bfloat16 |
| float32 | f32 (identity) | f32 | f32 (identity) | float32 |

Precision thresholds per dtype:

| dtype | MERE threshold | MARE threshold |
|-------|---------------|----------------|
| float16 | 2^-10 ≈ 9.77e-4 | 9.77e-3 |
| bfloat16 | 2^-7 ≈ 7.81e-3 | 7.81e-2 |
| float32 | 2^-13 ≈ 1.22e-4 | 1.22e-3 |

f32 internal compute gives ~7 decimal digits of precision, well within all thresholds.

---

## 10. Numerical Edge Cases

### NaN (case 13: [nan, nan] f32, mode "none")

- `z = abs(nan) * _INV_SQRT2` → `nan`
- `den = nan * 0.5 + 1.0` → `nan`
- `tt = 1 / nan` → `nan`
- Horner chain on `nan` → `nan`
- `exp(nan - nan*nan)` → `exp(nan)` → `nan`
- `erfc_z = nan * nan` → `nan`
- `half_erfc` → `nan`, `y_neg` → `nan`, `y_pos` → `nan`
- `where(nan >= 0, nan, nan)` → `nan` (NaN >= 0 is false → selects `y_neg = nan`)
- Output: all NaN. Golden: `torch.nn.functional.gelu(nan, approximate='none')` → NaN. **MATCH**.

### Inf (case 12: [-inf, inf] bf16, mode "tanh")

After f32 promotion:

**x = +inf**:
- `x2 = inf * inf` → `inf`
- `s = (inf * _GC_T + k) * inf` → `inf * inf` → `inf` (or just `inf`)
- `min(inf, 0) = 0` → `exp(0) = 1`
- `abs(inf) = inf` → `-inf` → `exp(-inf) = 0`
- `sig = 1 / (0 + 1) = 1`
- `y = inf * 1 = inf`. Golden: `inf`. **MATCH**.

**x = -inf**:
- `x2 = inf`
- `s = (inf * _GC_T + k) * (-inf)` → `-inf` (or `nan` depending on intermediate signs — both acceptable)
- If `s = -inf` or `s = nan`:
  - `min(s, 0) = s` → `exp(-inf) = 0` or `exp(nan) = nan`
  - `abs(s) = inf` or `nan` → `-inf` or `nan` → `exp(-inf) = 0` or `exp(nan) = nan`
  - `sig = 0 / 1 = 0` or `nan / inf = nan`
  - `y = -inf * 0 = nan` (IEEE: inf*0 = nan) or `-inf * nan = nan`
- Golden: `torch.nn.functional.gelu(-inf, approximate='tanh')`:
  - `u = sqrt(2/pi) * (-inf + 0.044715*(-inf)^3) = sqrt(2/pi) * (-inf + (-inf)) = -inf`
  - `tanh(-inf) = -1`
  - `1 + (-1) = 0`
  - `0.5 * (-inf) * 0` → nan (IEEE: inf*0 = nan)
- **MATCH**: both produce nan.

### All zeros (case 14: [0, 0] f16, mode "tanh")

- `xf = 0`
- `x2 = 0`, `s = (0 + k) * 0 = 0`
- `min(0, 0) = 0`, `exp(0) = 1`
- `abs(0) = 0`, `exp(0) = 1`, `den = 1 + 1 = 2`
- `sig = 1 / 2 = 0.5`
- `y = 0 * 0.5 = 0`
- Golden: `0`. **MATCH**.

### Extreme f16 (case 10: [-65504, 65504] f16, mode "tanh")

After f32 promotion, values are representable:
- For `x = 65504`: `x^3 ≈ 2.81e14`, `s` very large positive → `sig → 1` → `y ≈ 65504`. Cast to f16: `65504` (exactly f16 max). **OK**.
- For `x = -65504`: same magnitude, `s` very negative → `sig → 0` → `y → 0`. **OK**.

### Very large f32 (case 5: [-100, 100] f32, mode "tanh")

- For `x = 100`: `x^3 = 1e6`, `0.044715 * 1e6 = 44715`, `s ≈ 1.596 * 44815 ≈ 71532`
  - `min(71532, 0) = 0`, `exp(0) = 1`
  - `exp(-71532) = 0` (f32 underflow)
  - `sig = 1/1 = 1`, `y = 100`. **OK**.
- For `x = -100`: `s ≈ -71532`
  - `min(-71532, 0) = -71532`, `exp(-71532) = 0`
  - `abs(-71532) = 71532`, `exp(-71532) = 0`
  - `den = 0 + 1 = 1`, `sig = 0/1 = 0`, `y = -100 * 0 = 0`. **OK**.

### Moderate negative values (the cancellation-prone zone)

- Case 5, `x = -4` f32: `x^3 = -64`, `s ≈ 1.596 * (-4 + 0.044715*(-64)) = 1.596 * (-6.862) ≈ -10.948`
  - Stable form: `exp(-10.948) / (1 + exp(-10.948)) ≈ 1.76e-5 / 1.0000176` → exact in f32
  - Golden: same sigmoid value → tiny relative error from exp hardware precision (~1 ULP). **OK**.

### Large values in erf mode (case 11: [-88, 88] f32)

- For `x = 88`: `z = 88 * 0.707 ≈ 62.2`
  - `t = 1/(1 + 31.1) ≈ 0.0312`
  - Horner polynomial converges, `p` finite
  - `erfc ≈ t * exp(p - 3867)` → `t * 0` = 0 (exp underflow in f32)
  - `half_erfc = 0`, `y_neg = 0`, `y_pos = 88 - 0 = 88`
  - `y = 88`. Golden: `88`. **MATCH**.
- For `x = -88`: `z = 62.2` (same as above)
  - `erfc = 0`
  - `half_erfc = 0`, `y_neg = -88 * 0 = 0`
  - `y_neg = 0`. Golden: `0`. **MATCH**.
- For `x = -6`: `z = 4.24`
  - Horner fit gives `erfc(4.24) ≈ 4.25e-8` (exact, rel err < 1.2e-7)
  - `half_erfc ≈ 2.125e-8`
  - `y_neg = -6 * 2.125e-8 ≈ -1.275e-7`
  - Golden also computes `-6 * 0.5 * erfc(4.24)`. Same formula, same precision.
  - Relative error: dominated by exp hardware precision, well within 2^-13. **MATCH**.

---

## 11. Host Dispatch Logic

```
gelu(x, approximate="none"):
    1. ensure_npu_platform()
    2. x contiguous if needed
    3. out = torch.empty_like(x)
    4. size = x.numel(); early return if size == 0
    5. If approximate == "none":
         Try _gelu_erf_kernel_u1[cores](..., tile=1024)
         On RuntimeError (UB overflow):
           Fall back to _gelu_erf_kernel[cores](..., tile=512)
         Cache result in _erf_wide_ok
       Else (tanh):
         Try _gelu_tanh_kernel[cores](..., tile=1536)
         On RuntimeError:
           Fall back to _gelu_tanh_kernel[cores](..., tile=1024)
         Cache result in _tanh_tile
    6. cores = min(72, ceildiv(size, tile))
    7. Return out
```

Tile count: `num_tiles = asc.ceildiv(size, tile_size)`
Cores: `cores = min(_MAX_CORES, num_tiles)`

No stream argument. Launch: `kernel[cores](x, out, size, num_tiles, tile_size)`.

Pass tensors directly (not `data_ptr()`). The pinned JIT derives pointer dtype specialization from the tensor.

---

## 12. Anti-Cheat Compliance

| Rule | Compliance |
|------|-----------|
| All numerical work in `@asctile.jit` kernels | YES — erf/tanh chains fully in kernel |
| torch used only for alloc/metadata/contiguous/view | YES — `torch.empty_like`, `.numel()`, `.is_contiguous()`, `.contiguous()` only |
| No torch math ops | YES — no `torch.mul`, `torch.nn.functional.*`, no tensor arithmetic, no `.to(dtype)` casts of device data |
| No data_ptr caching | YES — tensors passed directly to kernel |
| Output is contiguous NPU tensor | YES — `torch.empty_like` produces contiguous, `copy_out` fills it |
| No views of inputs | YES — output allocated separately |
| NaN/Inf propagated through hardware ops | YES — no host special-casing |

---

## 13. `asctile.where` Scalar Branch Rule

The pinned v2 compiler requires both data branches of `asctile.where(condition, a, b)` to be `LocalTensor` values (not Python floats). The erf kernel satisfies this: both `y_pos` and `y_neg` are computed as tile-tile or tile-scalar products, always producing tiles.

`asctile.full([tile_size], 1.0, dtype=asc.float32)` is used to materialize the scalar `1.0` in a tile for the `tt = full / den` division (avoids `1.0 / den` which would fail as scalar/tile).

---

## 14. Scalar Operand Rule

All tile arithmetic: scalars on the RIGHT. `asctile` Tile class has `__mul__`, `__add__`, `__sub__`, `__truediv__` but NOT `__rmul__`, `__radd__`, etc.

Violations that would fail:
- `0.5 * z` → `float.__mul__(tile)` → NotImplemented → no `__rmul__` → **TypeError**
- `1.0 + ea` → `float.__add__(tile)` → NotImplemented → **TypeError**
- `1.0 / den` → `float.__truediv__(tile)` → NotImplemented → **TypeError**

All expressions in this design place scalars on the right: `z * 0.5`, `den + 1.0`, `asctile.full(...)/den`.

---

## 15. `real_shape` Padding Safety

For the erf chain with neutral (zero) padding in inactive lanes:
- `z = abs(0) * _INV_SQRT2 = 0` → safe
- `den = 0 * 0.5 + 1.0 = 1.0` → safe (no division by zero)
- `tt = 1.0 / 1.0 = 1.0` → safe
- Horner evaluates to constant `P(1)` — finite
- `exp(P(1) - 0) = exp(finite)` — safe (no overflow for this particular value)
- `erfc_z = 1.0 * exp(finite)` — finite result
- `half_erfc = finite * 0.5` — finite
- `y_neg = 0 * half_erfc = 0` → neutral output
- `y_pos = 0 - 0 = 0` → neutral
- `where(0 >= 0, 0, 0) = 0` → output is 0. `copy_out` with `real_shape` ignores pad values.

For the tanh chain with zero padding:
- `xf = 0`, `x2 = 0`, `s = (0 + k) * 0 = 0`
- `min(0, 0) = 0`, `exp(0) = 1`
- `abs(0) = 0`, `exp(0) = 1`
- `den = 1 + 1 = 2`, `sig = 1/2 = 0.5`
- `y = 0 * 0.5 = 0` → neutral output

Both chains are safe with default `pad_value=0`.

---

## 16. Performance Considerations

- **Grid-stride loop** ensures all 72 AIV cores are utilized for large inputs, with `cores = min(72, num_tiles)`.
- **Tile size** is the primary throughput knob: larger tiles amortize loop overhead but consume more UB.
- **Two-tier tile selection** (try wide, fall back to safe) adapts to the specific compiler build without hardcoded assumptions.
- **Factored cubic** in tanh mode saves one tile-scalar multiply per element: `(x2 * _GC_T + k) * xf` vs `xf + x3 * _GELU_C` then `* _TWO_SQRT_2_OVER_PI`.
- **`unroll_factor=1`** for the wide erf kernel halves UB allocation, enabling TILE=1024.

---

DESIGN_DONE.
