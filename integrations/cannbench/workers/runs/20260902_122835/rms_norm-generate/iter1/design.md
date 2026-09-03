# Kernel Design: RmsNorm (CANN Bench candidate.py)

## 1. Operation description

**Mathematical definition**:

  y_i = x_i * gamma_i * rsqrt( (1/D) * sum_j(x_j^2) + epsilon )

RMSNorm normalizes each row independently along the last dimension D.
No mean subtraction (unlike LayerNorm); no bias term. The reduction
axis is -1 for every row.

**Input tensors**:
- `x`: (..., D) — any rank 2-8, D is the last dim. dtypes: f16/f32/bf16.
- `gamma`: (D,) — 1-D scaling vector, same dtype as x.

**Output tensors**:
- `y`: same shape and dtype as x.

**Test cases** (20 total): D ranges 2-8192; S (rows) ranges 231-1000003;
dtypes f16/f32/bf16; epsilon 1e-12 to 1e-3; value ranges include
[-65504, 65504] (fp16 boundary), all-zeros [0,0], and unaligned prime
tails (D=1021, 1023, 2049, 4097, 4099).

## 2. Algorithm

Flatten x to (S, D) where S = product of leading dims. Each row is
normalized independently:

1. **Reduction**: sum_sq = sum(x_j^2) for j in [0, D). Computed in
   float32 (x promoted via `.to(asc.float32)` before squaring) to
   prevent f16/bf16 overflow (case 19: x up to +-65504, x^2 = 4.29e9
   overflows f16 max 65504 but is well within f32 range).
2. **Inverse RMS**: inv_rms = 1.0 / sqrt(sum_sq / D + epsilon).
   `asc2.rsqrt` accepts tiles only, not scalars, so use
   `1.0 / asc2.sqrt(scalar)` (verified in golden rms_norm_f32.py:108).
3. **Scale**: y_i = x_i * gamma_i * inv_rms. Scalars go on the RIGHT
   of tile arithmetic (task.md rule), so `xf * gf * inv_rms` is valid
   (tile * tile * scalar).
4. **Cast back**: `y.to(x.dtype)` before `asc2.copy_out`.

## 3. pyasc API selection

| Purpose | API | Notes |
|---------|-----|-------|
| Kernel decorator | `@asc2.jit` | Bare, matching sigmoid.py reference |
| Global memory views | `asc2.global_tensor(ptr, [rows, cols])` | 2-D for x/out; `asc2.global_tensor(gamma_ptr, [1, cols])` for gamma |
| Load tile from GM | `asc2.copy_in(gm, [row, col], [1, tile], real_shape=[1, n])` | `real_shape` for tail handling |
| Store tile to GM | `asc2.copy_out(tile, gm, [row, col], real_shape=[1, n])` | Cast back to input dtype first |
| Grid-stride row loop | `asc2.range(block_idx(), rows, block_num(), unroll_factor=2)` | Distributes rows across cores |
| Inner tile loop | `asc2.range(num_tiles, unroll_factor=2)` | Sequential (loop-carried acc in pass 1) |
| Full-tile reduction | `asc2.reduce_sum(tile)` | Returns PlainValue (scalar) |
| Scalar sqrt | `asc2.sqrt(scalar)` | Supports scalars (unlike `asc2.rsqrt`) |
| Scalar arithmetic | `sum_sq / cols + epsilon`, `1.0 / ...` | PlainValue operator overloads |
| Zero-seed accumulator | `asc2.full([1, tile], 0.0, dtype=asc.float32)` | Seeds loop-carried scalar |
| Tile arithmetic | `xf * xf`, `xf * gf * inv_rms` | Tile*tile=tile, tile*scalar=tile |
| Cast | `x.to(asc.float32)`, `y.to(x.dtype)` | f16/bf16 promoted to f32 for compute |
| Kernel params | `asc.GlobalAddress`, `asc.ConstExpr[int]`, `int`, `float` | Pointers, compile-time tile sizes, runtime ints/floats |
| Host tiling math | `asc.ceildiv(cols, tile)` | Host-side tile count |

**Key API constraints (from task.md kernel contract)**:
- `asc2.range` on this build accepts ONLY `unroll_factor` and `parallel` —
  NO `gm_barrier` kwarg (raises TypeError). Loop-carried accumulators
  work without `gm_barrier` (verified in existing submission's fallback).
- `asc2.rsqrt` does NOT accept scalars — use `1.0 / asc2.sqrt(scalar)`.
- Ranks must be consistent per-tensor (x/out 2-D, gamma 2-D `[1, cols]`).
  Never mix 1-D and 2-D offsets for the same global_tensor.
- Scalars on the RIGHT of tile arithmetic: `xf * gf * inv_rms`, not
  `inv_rms * xf * gf`.

## 4. Tiling strategy

Two kernel functions, selected by a host dispatcher:

### 4.1 Full-row kernel (`_rms_norm_full_row_kernel`)

**When**: D <= 2048 AND D is 32-byte aligned
  (D * element_size % 32 == 0: D % 8 == 0 for f32, D % 16 == 0 for f16/bf16).

**Params**: `x_ptr, gamma_ptr, out_ptr` (GlobalAddress), `rows: int`,
  `cols: asc.ConstExpr[int]`, `epsilon: float`.

**Per row** (grid-stride across `min(72, rows)` cores, unroll_factor=2):
1. Load full row: `x_row = asc2.copy_in(x_gm, [row, 0], [1, cols])`.
2. `xf = x_row.to(asc.float32)`.
3. `sum_sq = asc2.reduce_sum(xf * xf)` — scalar.
4. `inv_rms = 1.0 / asc2.sqrt(sum_sq / cols + epsilon)`.
5. `gamma_row = asc2.copy_in(gamma_gm, [0, 0], [1, cols])` (shared across rows, loaded per row like the golden).
6. `gf = gamma_row.to(asc.float32)`.
7. `y = xf * gf * inv_rms`.
8. `asc2.copy_out(y.to(x_row.dtype), out_gm, [row, 0])`.

**Advantage**: single-pass — x loaded once, reused for both reduction
and output. No re-read of x from GM.

**ConstExpr `cols`**: kernel is JIT-compiled once per distinct D value
and cached. Test cases with aligned D <= 2048: D=128, 768, 1024, 2048
(4 compilations).

### 4.2 Split-D kernel (`_rms_norm_split_d_kernel`)

**When**: D > 2048 OR D not 32-byte aligned.

**Params**: `x_ptr, gamma_ptr, out_ptr` (GlobalAddress), `rows: int`,
  `cols: int` (runtime), `num_tiles: int`, `tile_size: asc.ConstExpr[int]`,
  `epsilon: float`.

**Two tile sizes** (ConstExpr, 2 compilations total):
- `_TILE_LARGE = 2048` — for D > 256. Aligned (2048*4=8192 % 32 = 0).
- `_TILE_SMALL = 64` — for D <= 256 (covers D=2, 67). Aligned (64*4=256 % 32 = 0).

**Per row** (grid-stride, unroll_factor=2):

*Pass 1 (reduction)*:
1. Seed: `acc = asc2.reduce_sum(asc2.full([1, tile_size], 0.0, dtype=asc.float32))`.
2. Loop `for tile_id in asc2.range(num_tiles, unroll_factor=2)`:
   - `col = tile_id * tile_size`
   - `n = tile_size if col + tile_size <= cols else cols - col`
   - `x = asc2.copy_in(x_gm, [row, col], [1, tile_size], real_shape=[1, n])`
   - `xf = x.to(asc.float32)`
   - `acc = acc + asc2.reduce_sum(xf * xf)`

*Compute*: `inv_rms = 1.0 / asc2.sqrt(acc / cols + epsilon)`

*Pass 2 (output)*:
3. Loop `for tile_id in asc2.range(num_tiles, unroll_factor=2)`:
   - `col = tile_id * tile_size`
   - `n = tile_size if col + tile_size <= cols else cols - col`
   - `x = asc2.copy_in(x_gm, [row, col], [1, tile_size], real_shape=[1, n])`
   - `gamma = asc2.copy_in(gamma_gm, [0, col], [1, tile_size], real_shape=[1, n])`
   - `y = x.to(asc.float32) * gamma.to(asc.float32) * inv_rms`
   - `asc2.copy_out(y.to(x.dtype), out_gm, [row, col], real_shape=[1, n])`

**Two-pass rationale**: x must be re-read from GM in pass 2 because the
full row doesn't fit in UB for large D. For D <= 2048 (non-aligned),
the two-pass overhead is acceptable (only 1 tile per row, the waste is
in padding not in extra GM reads).

### 4.3 Host dispatcher

```
cols = x.shape[-1]
rows = x.numel() // cols
element_size = x.element_size()        # 4 f32, 2 f16/bf16
align = 32 // element_size              # 8 f32, 16 f16/bf16

if cols <= 2048 and cols % align == 0:
    _rms_norm_full_row_kernel[min(72, rows)](
        x, gamma, out, rows, cols, float(epsilon))
elif cols <= 256:
    tile = 64
    num_tiles = asc.ceildiv(cols, tile)
    _rms_norm_split_d_kernel[min(72, rows)](
        x, gamma, out, rows, cols, num_tiles, tile, float(epsilon))
else:
    tile = 2048
    num_tiles = asc.ceildiv(cols, tile)
    _rms_norm_split_d_kernel[min(72, rows)](
        x, gamma, out, rows, cols, num_tiles, tile, float(epsilon))
```

## 5. Tail handling

**Method**: `real_shape` parameter on `asc2.copy_in`/`asc2.copy_out`
(no host zero-padding, matching the sigmoid.py reference).

For the last tile in a row where `col + tile_size > cols`:
- `n = cols - col` (runtime conditional, supported in @asc2.jit per
  existing submission's fallback kernel).
- `copy_in(..., real_shape=[1, n])` — loads only n real elements,
  pads the rest with 0 (default `pad_value=0`).
- `copy_out(..., real_shape=[1, n])` — writes only n elements.

**Padding zeros do not bias the reduction**: zero-padded elements
contribute 0 to `reduce_sum(xf * xf)`, so `acc` is the correct sum of
real x^2 values. The division `acc / cols` uses the REAL `cols`
(runtime int), not `tile_size * num_tiles`, so the mean is correct.

**Full-row kernel**: no tail handling needed (D is aligned and exactly
`cols` elements are loaded/stored).

## 6. UB budget

Total UB on 950PR: ~253,952 bytes. The 1.6x safety factor from
task.md accounts for hidden compiler temporaries. All estimates
include `unroll_factor=2` (doubles live values).

### 6.1 Full-row kernel (D=2048, f32 — worst case)

| Value | dtype | bytes | x unroll=2 |
|-------|-------|-------|------------|
| x_row | f32 | 8,192 | 16,384 |
| xf (f32 cast) | f32 | 8,192 | 16,384 |
| x_sq = xf*xf | f32 | 8,192 | 16,384 |
| gamma_row | f32 | 8,192 | 16,384 |
| gf (f32 cast) | f32 | 8,192 | 16,384 |
| out_f32 | f32 | 8,192 | 16,384 |
| out_cast | f32 | 8,192 | 16,384 |
| **Total visible** | | 57,344 | **114,688** |
| **x 1.6 safety** | | | **183,501** |

183,501 < 253,952. **SAFE** for D=2048 f32.

For f16 D=2048: visible = 22*D*2 = 90,112 bytes; x 1.6 = 144,179. SAFE.

### 6.2 Split-D kernel, pass 2 (heaviest phase, tile=2048, f32)

| Value | dtype | bytes | x unroll=2 |
|-------|-------|-------|------------|
| x | f32 | 8,192 | 16,384 |
| xf | f32 | 8,192 | 16,384 |
| gamma | f32 | 8,192 | 16,384 |
| gf | f32 | 8,192 | 16,384 |
| y = xf*gf*inv_rms | f32 | 8,192 | 16,384 |
| y_cast | f32 | 8,192 | 16,384 |
| **Total visible** | | 49,152 | **98,304** |
| **x 1.6 safety** | | | **157,286** |

157,286 < 253,952. **SAFE**.

Pass 1 (reduction): 3 visible values (x, xf, x_sq) = 24,576 x 2 x 1.6
= 78,643 bytes. SAFE.

### 6.3 Split-D with tile=64

All values scale down 32x from tile=2048. Pass 2: ~4,915 bytes. Trivially safe.

### 6.4 Full-row for D=4096 (NOT used — documented as risk)

If we tried full_row with D=4096 f32 unroll=2:
visible = 57,344*2 = 114,688 x 2 (unroll) x 1.6 = 367,002 > 253,952.
**OVERFLOW**. This is why D > 2048 routes to split_d.

## 7. Numerical risks

### 7.1 f16/bf16 overflow in x^2 (CRITICAL — case 19)

Case 19: f16, [-65504, 65504], D=4096. x^2 can be 65504^2 = 4.29e9,
which overflows f16 (max 65504). **Mitigation**: promote to f32 via
`x.to(asc.float32)` BEFORE squaring. f32 max is ~3.4e38, so 4.29e9 is
safe. Sum over D=4096: max 1.76e13, also safe in f32.

### 7.2 Sum overflow

Worst case: D=8192, x=+-100 (f32), x^2=1e4, sum=8192*1e4=8.19e7.
Well within f32 range. Even case 19 (f16->f32, x=+-65504):
sum = 4096 * 4.29e9 = 1.76e13, safe in f32.

### 7.3 Underflow for very small x

Case 7: f16, [-0.1, 0.1], D=1023. x^2 <= 0.01. Sum <= 10.23.
In f32, no underflow (f32 min normal ~1.18e-38). Fine.

### 7.4 Division by zero

`sum_sq / D + epsilon`: epsilon >= 1e-12 > 0, so the denominator is
always positive. `asc2.sqrt(positive)` is finite. `1.0 / finite` is
finite. No division by zero possible.

### 7.5 All-zero rows (case 17)

x=0, sum_sq=0, mean=0, sqrt(0 + 1e-4)=0.01, inv_rms=100.
y = 0 * gamma * 100 = 0. Matches golden (torch.nn.functional.rms_norm
gives the same: 0 normalized = 0).

### 7.6 inf/nan propagation

If x_i = inf: x^2 = inf, sum = inf, sqrt(inf+eps) = inf,
inv_rms = 1/inf = 0.0, y = inf * gamma * 0.0 = nan.
This matches `torch.nn.functional.rms_norm` (same f32 computation
path). **Do NOT clamp or special-case** (task.md: "Do not clamp or
replace IEEE values; match the golden propagation").

### 7.7 Cancellation

No subtraction of nearly-equal quantities. The formula is
multiplicative: x * gamma * inv_rms. No catastrophic cancellation risk.

### 7.8 epsilon as runtime float (not ConstExpr)

`epsilon: float` avoids per-epsilon JIT recompilation (8 distinct
epsilon values in test cases). `asc2.sqrt(scalar + float)` works via
PlainValue.__add__. The existing submission's fallback kernel uses
this approach successfully.

## 8. Anti-cheat constraints

| Rule | Compliance |
|------|------------|
| All numerical work inside @asc2.jit | sum_sq, sqrt, multiply — all in kernel |
| torch only for allocation/metadata | `ensure_npu_platform()`, `.contiguous()`, `.shape`, `.numel()`, `.element_size()`, `torch.empty_like()` |
| No torch math/compute ops | No `torch.mul`, `torch.norm`, `torch.nn.functional.*`, tensor arithmetic, `.to(dtype)` on device data |
| No caching by data_ptr | No data_ptr inspection |
| Output contiguous NPU tensor | `torch.empty_like(x)` — contiguous, same shape/dtype |
| No views of inputs returned | Output is a fresh allocation |
| Public callable signature | `rms_norm(x, gamma, epsilon=1e-6)` |
| Module imports | `import torch`, `import asc`, `import asc2`, `from ._pyasc_runtime import ensure_npu_platform` |
| No math.* in @asc2.jit | `asc2.sqrt` used, not `math.sqrt`; no module-level math constants needed |

## 9. Static verification plan

No NPU/pyasc available locally. Verification is static only:

1. **Syntax check**: `python3 -m py_compile candidate.py` — valid Python.
2. **AST checks** (via `ast` module or manual inspection):
   - `@asc2.jit` decorator present on kernel functions.
   - No `print`, `break`, `continue`, `try/except`, `import`, `lambda`
     inside `@asc2.jit` functions.
   - No `math.*` calls inside `@asc2.jit`.
   - No `range()` (Python built-in) over runtime values inside `@asc2.jit`
     — use `asc2.range` only.
   - No early `return` inside kernel.
3. **API usage checks**:
   - `asc2.global_tensor` used (not `asc2.tensor`).
   - `asc2.copy_in`/`asc2.copy_out` used (not `asc2.load`/`asc2.store`).
   - `asc2.reduce_sum` used for sum-of-squares.
   - `1.0 / asc2.sqrt(...)` used for inv_rms (not `asc2.rsqrt` on scalar).
   - `asc2.range` has no `gm_barrier` kwarg.
   - Ranks consistent per-tensor (x/out 2-D, gamma 2-D `[1, cols]`).
   - `real_shape` used for tail handling (no host padding).
   - Scalars on RIGHT of tile arithmetic.
4. **Host code checks**:
   - `ensure_npu_platform()` called first.
   - `x.contiguous()` called if needed.
   - Output allocated with `torch.empty_like(x)`.
   - `cores = min(72, rows)` used for launch.
   - No torch math ops (only `empty_like`, `contiguous`, shape/dtype queries).
5. **Signature check**: public function is `rms_norm(x, gamma, epsilon=1e-6)`.
6. **Case coverage** (manual): all 20 cases map to a valid kernel path
   (full_row for aligned D<=2048; split_d tile=64 for D<=256;
   split_d tile=2048 for D>256).

## 10. Case-to-kernel mapping

| Case | D | dtype | Aligned? | Kernel | Tile | Tiles/row |
|------|---|-------|----------|--------|------|-----------|
| 1 | 768 | f16 | yes (768%16=0) | full_row | — | 1 |
| 2 | 1024 | f32 | yes | full_row | — | 1 |
| 3 | 2048 | bf16 | yes | full_row | — | 1 |
| 4 | 4096 | f16 | yes but >2048 | split_d | 2048 | 2 |
| 5 | 8192 | f32 | yes but >2048 | split_d | 2048 | 4 |
| 6 | 4097 | bf16 | no | split_d | 2048 | 3 |
| 7 | 1023 | f16 | no | split_d | 2048 | 1 |
| 8 | 2049 | f32 | no | split_d | 2048 | 2 |
| 9 | 4099 | bf16 | no | split_d | 2048 | 3 |
| 10 | 769 | f16 | no | split_d | 2048 | 1 |
| 11 | 2049 | f32 | no | split_d | 2048 | 2 |
| 12 | 4097 | bf16 | no | split_d | 2048 | 3 |
| 13 | 1021 | f16 | no | split_d | 2048 | 1 |
| 14 | 373 | f32 | no | split_d | 2048 | 1 |
| 15 | 2 | bf16 | no, D<=256 | split_d | 64 | 1 |
| 16 | 67 | f16 | no, D<=256 | split_d | 64 | 2 |
| 17 | 4096 | f32 | yes but >2048 | split_d | 2048 | 2 |
| 18 | 8192 | bf16 | yes but >2048 | split_d | 2048 | 4 |
| 19 | 4096 | f16 | yes but >2048 | split_d | 2048 | 2 |
| 20 | 128 | f32 | yes | full_row | — | 1 |

**JIT compilations**: 4 (full_row: D=128,768,1024,2048) + 2 (split_d:
tile=64, tile=2048) = **6 total**. All cached after first call.

## 11. Syntax compliance check

- [x] All constructs inside `@asc2.jit` are in the supported set
  (for/if-else/asc2.range/asc2.copy_in/asc2.copy_out/asc2.reduce_sum/
  asc2.sqrt/asc2.full/tile arithmetic/.to() casts)
- [x] No unsupported syntax (no print, break, continue, lambda, import,
  try/except, yield, global, nonlocal, nested functions inside JIT)
- [x] No `math.*` calls inside `@asc2.jit`
- [x] No Python `range()` inside `@asc2.jit` (only `asc2.range`)
- [x] Kernel does not return a value (void — output via `asc2.copy_out`)
- [x] No `gm_barrier` kwarg on `asc2.range`
- [x] No `asc2.tensor`/`asc2.load`/`asc2.store` (v1 API) — uses `asc2.global_tensor`/`asc2.copy_in`/`asc2.copy_out`
- [x] Scalars on RIGHT of tile arithmetic (`xf * gf * inv_rms`)
- [x] `asc.ConstExpr[int]` for tile sizes used in `copy_in` shape
- [x] `asc.GlobalAddress` for pointer params
- [x] f16/bf16 promoted to f32 before compute
- [x] Loop-carried accumulator seeded with `asc2.reduce_sum(asc2.full(...))`
- [x] `real_shape` used for tail handling (no host padding)
- [x] `ensure_npu_platform()` called first in host function
- [x] No torch math/compute ops in host code
- [x] Output allocated with `torch.empty_like(x)`

## 12. Design Score

**Rating: 9.0/10**

**Strengths**:
- Proven two-kernel pattern (full_row + split_d) directly from golden
  rms_norm_f32.py/rms_norm_f16.py, adapted for 950PR (72 cores, real_shape
  tails, no host padding, runtime epsilon).
- UB budget verified with 1.6x safety factor for all D/dtype combinations.
- Numerical stability: f32 accumulation prevents f16 x^2 overflow (case 19);
  no catastrophic cancellation; IEEE inf/nan propagation matches golden.
- Minimal JIT compilations (6 total) via two fixed ConstExpr tile sizes.
- Small-D optimization (tile=64) reduces padding waste for cases 15-16.

**Risks**:
- Non-aligned D in [257, 2048] range uses split_d with tile=2048, causing
  padding waste (e.g. D=373 processes 2048 elements per tile). Performance
  impact on cases 7, 10, 13, 14. Could add tile=512 for medium D if needed.
- D=2 (case 15) with tile=64 still has 32x padding waste. Unavoidable
  without a D-specialized kernel (would add compilations).
- `gm_barrier` not available on this build — loop-carried accumulator
  relies on default `parallel=False` semantics. Verified in existing
  submission's fallback kernel but not independently tested.
- No runtime verification possible (no NPU/pyasc locally). All checks
  are static/AST-based.
