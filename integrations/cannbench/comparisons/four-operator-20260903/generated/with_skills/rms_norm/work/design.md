# RmsNorm — Design Document

## 0. Runtime Pin

pyasc v2 commit `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`.  
Package exports `asctile` (not `asc2`). All kernel code uses `import asctile`, `@asctile.jit`, `asctile.*`.

## 1. Specification Summary

**Formula**: `y = x / sqrt(mean(x^2) + epsilon) * gamma`

| Property | Domain |
|----------|--------|
| x rank | 2–5 (measured minimum 2, maximum 5 in 20 cases) |
| D (last dim) | 2, 67, 128, 373, 768, 769, 1021, 1023, 1024, 2048, 2049, 4096, 4097, 4099, 8192 |
| S (leading product) | 102 – 1000003 |
| x dtype | float16, float32, bfloat16 |
| gamma dtype | same as x |
| epsilon | 1e-12, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3 |
| value ranges | [-65504, 65504] (fp16), [-100, 100] (fp32), [-5, 5] (bf16), [0, 0] (all zeros) |
| gamma shape | (D,) — always 1-D |
| output y | same shape and dtype as x |

**Precision thresholds**:

| dtype | MERE | MARE |
|-------|------|------|
| fp16 | < 2^-10 ≈ 9.77e-4 | < 9.77e-3 |
| bf16 | < 2^-7 ≈ 7.81e-3 | < 7.81e-2 |
| fp32 | < 2^-13 ≈ 1.22e-4 | < 1.22e-3 |

## 2. Case Matrix

| case | x shape | D | S | dtype | epsilon | value_range | dispatch path |
|------|---------|------|------|-------|---------|-------------|---------------|
| 1 | [32, 128, 768] | 768 | 4096 | fp16 | 1e-6 | [-1, 1] | builtin |
| 2 | [32, 128, 1024] | 1024 | 4096 | fp32 | 1e-6 | [-2, 2] | builtin |
| 3 | [32, 128, 2048] | 2048 | 4096 | bf16 | 1e-6 | [-3, 3] | builtin (f16 hop) |
| 4 | [16, 256, 4096] | 4096 | 4096 | fp16 | 1e-6 | [-10, 10] | builtin |
| 5 | [8, 512, 8192] | 8192 | 4096 | fp32 | 1e-6 | [-100, 100] | manual (D>4096) |
| 6 | [4, 1023, 4097] | 4097 | 4092 | bf16 | 1e-5 | [-5, 5] | manual (unaligned) |
| 7 | [63, 67, 1023] | 1023 | 4221 | fp16 | 1e-8 | [-0.1, 0.1] | manual (unaligned) |
| 8 | [16, 511, 2049] | 2049 | 8176 | fp32 | 1e-4 | [-1, 1] | manual (unaligned) |
| 9 | [8, 1021, 4099] | 4099 | 8168 | bf16 | 1e-12 | [-0.5, 0.5] | manual (unaligned, D>4096) |
| 10 | [33, 127, 769] | 769 | 4191 | fp16 | 1e-6 | [-1, 2] | manual (unaligned) |
| 11 | [31, 129, 2049] | 2049 | 4019 | fp32 | 1e-6 | [-50, 100] | manual (unaligned) |
| 12 | [17, 255, 4097] | 4097 | 4335 | bf16 | 1e-6 | [-3, 6] | manual (unaligned) |
| 13 | [7, 1009, 1021] | 1021 | 7063 | fp16 | 1e-7 | [-1, 1] | manual (unaligned) |
| 14 | [11, 367, 373] | 373 | 4037 | fp32 | 1e-5 | [-10, 10] | manual (unaligned) |
| 15 | [1000003, 2] | 2 | 1000003 | bf16 | 1e-6 | [None, None] | manual (unaligned) |
| 16 | [11, 13, 17, 67] | 67 | 2431 | fp16 | 1e-8 | [None, None] | manual (unaligned) |
| 17 | [3, 7, 11, 4096] | 4096 | 231 | fp32 | 1e-4 | [0, 0] | builtin |
| 18 | [2, 511, 8192] | 8192 | 1022 | bf16 | 1e-6 | [-0.2, 0.2] | manual (D>4096) |
| 19 | [4, 255, 4096] | 4096 | 1020 | fp16 | 1e-3 | [-65504, 65504] | builtin |
| 20 | [2, 3, 17, 1024, 128] | 128 | 102 | fp32 | 1e-6 | [-20, 40] | builtin |

## 3. Overall Architecture

Two kernel paths gated by host dispatch:

```
rms_norm(x, gamma, epsilon)
    ├─ builtin path: asctile.rms_norm on full-row tiles (2-D)
    │   • fp16 and fp32 directly
    │   • bf16 via f16 hop (cast→builtin→cast back)
    │   • Gate: (D * element_size) % 32 == 0 AND D <= D_BUILTIN_MAX
    │
    └─ manual path: two-pass streaming per row
        • Pass 1: accumulate sum(x^2) in f32 scalar via reduce_sum
        • Pass 2: y = x * inv_rms * gamma, streaming D in TILE_D chunks
        • Handles: unaligned D, D > builtin limit, all bf16 (if builtin rejected)
```

### 3.1 Host Dispatch Logic

```python
def rms_norm(x, gamma, epsilon=1e-6):
    ensure_npu_platform()
    # contiguity
    x = x.contiguous() if not x.is_contiguous() else x
    gamma = gamma.contiguous() if not gamma.is_contiguous() else gamma
    # metadata
    D = x.shape[-1]
    S = x.numel() // D
    out = torch.empty_like(x)
    if S == 0 or D == 0: return out
    
    # dispatch
    copy_aligned = (D * x.element_size()) % 32 == 0
    fp32_overflow = (x.dtype == torch.float32 and D > MAX_TILE_D)
    use_builtin = copy_aligned and not fp32_overflow and D <= D_BUILTIN_MAX
    
    if use_builtin:
        # launch builtin kernel
    else:
        # launch manual kernel
    return out
```

## 4. Builtin Path

### 4.1 Kernel: `_rms_norm_builtin_kernel`

**Tiling**: 2-D tiles `[rows_per_tile, D]` where `rows_per_tile = max(1, MAX_TILE_ELEMENTS // D)`.  
Grid-stride over row groups.

**For fp16/fp32**:
```python
@asctile.jit(reuse_alloc=1)
def _rms_norm_builtin_kernel(x_ptr, gamma_ptr, out_ptr,
                              rows, cols: ConstExpr, total_tiles,
                              rows_per_tile: ConstExpr, epsilon: float):
    x_gm = asctile.global_tensor(x_ptr, [rows, cols])
    g_gm = asctile.global_tensor(gamma_ptr, [cols])
    o_gm = asctile.global_tensor(out_ptr, [rows, cols])
    gamma = asctile.copy_in(g_gm, [0], [cols])
    for tile_id in asctile.range(block_idx(), total_tiles, block_num(), unroll_factor=2):
        row = tile_id * rows_per_tile
        active_rows = rows_per_tile if row + rows_per_tile <= rows else rows - row
        x = asctile.copy_in(x_gm, [row, 0], [rows_per_tile, cols],
                            real_shape=[active_rows, cols])
        y = asctile.rms_norm(x, gamma, epsilon)
        asctile.copy_out(y, o_gm, [row, 0], real_shape=[active_rows, cols])
```

**For bf16 (f16 hop)**:
```python
        if x.dtype == asc.bfloat16:
            y = asctile.rms_norm(
                x.to(asc.float16), gamma.to(asc.float16), epsilon
            ).to(asc.bfloat16)
        else:
            y = asctile.rms_norm(x, gamma, epsilon)
```

**Rationale for f16 hop**: The builtin `asctile.rms_norm` only accepts fp16 and fp32 (check_dtype enforces this). Bf16 tolerance (2^-7) is loose enough that fp16 internal compute produces outputs within spec.

**UB budget estimation** (builtin):
- The builtin is an opaque hardware op — it manages its own UB internally.
- Live values outside: gamma tile (once, reused), x tile, y tile.
- For fp32 D=4096: x tile = 4 * rows_per_tile * 4096 bytes. With rows_per_tile=1: 16384 bytes. Manageable.
- For fp16 D=4096: x tile = 2 * rows_per_tile * 4096. With rows_per_tile=1: 8192 bytes.
- D_BUILTIN_MAX = 4096 (for fp32, reject D>4096 because builtin internal UB grows).

### 4.2 Builtin Constraints

- `asctile.rms_norm` input must be ≤ 2-D (LocalTensor). Satisfied by our 2-D `[rows_per_tile, D]` copy_in.
- gamma is 1-D, same length as last dim of input.
- epsilon is a RuntimeFloat — passed as kernel `float` parameter.
- Copy alignment: `(D * element_size) % 32 == 0` ensures DMA alignment.

## 5. Manual Fallback Path

### 5.1 Kernel: `_rms_norm_manual_kernel`

Two-pass streaming design. Single tile size TILE_D, no `unroll_factor` on D-tile loops (to minimize UB).

```
@asctile.jit
def _rms_norm_manual_kernel(x_ptr, gamma_ptr, out_ptr,
                             S, D, num_d_tiles,
                             epsilon: float, inv_D: float,
                             tile_d: ConstExpr):
    x_gm = asctile.global_tensor(x_ptr, [S * D])   # 1-D flat
    g_gm = asctile.global_tensor(gamma_ptr, [D])
    o_gm = asctile.global_tensor(out_ptr, [S * D])

    for r in asctile.range(block_idx(), S, block_num()):
        row_off = r * D

        # ---- Pass 1: accumulate sum(x^2) in f32 ----
        acc = asctile.reduce_sum(asctile.full([1, 64], 0.0, dtype=asc.float32))
        for dt in asctile.range(num_d_tiles):
            od = dt * tile_d
            n = tile_d if od + tile_d <= D else D - od
            x = asctile.copy_in(x_gm, [row_off + od], [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            acc = acc + asctile.reduce_sum(xf * xf)

        # ---- Scalar rstd ----
        # acc is PlainValue; asctile.rsqrt rejects PlainValue directly.
        # Broadcast to a tile to use tensor-only rsqrt.
        inv_rms_tile = asctile.rsqrt(
            asctile.full([tile_d], acc * inv_D + epsilon, dtype=asc.float32)
        )

        # ---- Pass 2: normalize ----
        for dt in asctile.range(num_d_tiles):
            od = dt * tile_d
            n = tile_d if od + tile_d <= D else D - od
            x = asctile.copy_in(x_gm, [row_off + od], [tile_d], real_shape=[n])
            g = asctile.copy_in(g_gm, [od], [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            gf = g.to(asc.float32)
            y = xf * inv_rms_tile * gf
            asctile.copy_out(y.to(x.dtype), o_gm, [row_off + od], real_shape=[n])
```

### 5.2 TILE_D Selection

TILE_D = 1024. Justification:

**UB budget analysis** (pass 1 inner loop, live values at any point):
- `x` (f16/bf16/f32 tile): 2*1024 or 4*1024 bytes
- `xf` (f32 tile): 4*1024 = 4096 bytes
- `xf * xf` (f32 tile): 4096 bytes
- `reduce_sum(xf*xf)` produces scalar; the intermediate tile is freed
- `acc` is PlainValue scalar: negligible

**UB budget analysis** (pass 2 inner loop, live values at any point):
- `x` (input type tile): 2*1024 or 4*1024 bytes
- `g` (gamma tile): 2*1024 or 4*1024 bytes
- `xf` (f32): 4096, `gf` (f32): 4096
- `inv_rms_tile` (f32): 4096 (persists across all D-tile iterations in pass 2)
- `y` (f32): 4096

**Total pass 2 visible**: x + g + xf + gf + inv_rms_tile + y = 2*4096 + 4*4096 = 24576 bytes (worst case, input is f32 and gamma is f32).

With the 1.6x overhead factor: ~39322 bytes. Well within 253952. No unroll doubling since D-tile loops use no `unroll_factor`.

**Total pass 1 visible**: x + xf + xf*xf + acc ≈ 3*4096 + small = 12288 bytes × 1.6 ≈ 19661 bytes. Safe.

**Between passes**: `inv_rms_tile` is the only live value carried from pass 1 to pass 2 = 4096 bytes × 1.6 ≈ 6554 bytes.

**Conclusion**: TILE_D=1024 with no unroll is well within UB budget for all dtypes.

### 5.3 Tile Size vs D

For very small D (D=2, case 15), TILE_D=1024 wastes space via padded lanes.  
However: `real_shape=[n]` with n=2 ensures only 2 elements are DMA'd and stored. The padded 1022 lanes execute arithmetic but their results are discarded. This is safe because:
- Padded `x` lanes in pass 1: copy_in fills with 0 (default pad_value), so `xf * xf = 0`, contributing nothing to `acc`. Operation-neutral for additive reduction.
- Padded `inv_rms_tile` lanes: broadcast from the same scalar, so all lanes have the same value. Harmless multiplication.
- Padded `g` lanes: copy_in fills default 0, but results are not stored (real_shape=[2] only).

No `pad_value` override needed — the default zero is operation-neutral for the sum-of-squares reduction in pass 1 and the normalization multiply in pass 2 (discarded by real_shape).

### 5.4 1-D Flat vs 2-D Addressing

The manual kernel uses 1-D flat addressing (`x_gm = [S*D]`, offsets `row_off + od`). This is simpler than 2-D and avoids alignment issues on the row dimension. The host flattens x to `[S, D]` logically but passes the contiguous tensor directly.

## 6. Numerical Behavior Analysis

### 6.1 Overflow Prevention

**fp16 x^2 overflow**: Maximum fp16 value is 65504. Squaring: 65504^2 ≈ 4.29e9, which overflows fp16 (max 65504).  
**Mitigation**: All computation promotes to f32 before squaring. `xf = x.to(asc.float32)` ensures squares are in f32 (max 3.4e38). This is the standard approach and matches the golden's behavior (PyTorch F.rms_norm internally promotes to f32).

**Case 19 stress test**: fp16 range [-65504, 65504], D=4096. Maximum sum_sq = 65504^2 * 4096 = 1.76e13. Well within f32 range. mean(x^2) = 1.76e13 / 4096 = 4.29e9. sqrt(4.29e9 + 1e-3) = 65504. y = 65504 / 65504 * gamma ≈ gamma. Within fp16 output range.

**Case 5 stress test**: fp32 range [-100, 100], D=8192. Maximum sum_sq = 100^2 * 8192 = 8.192e7. mean = 10000. sqrt(10000 + 1e-6) = 100. y = 100/100 * gamma ≈ gamma. No overflow concern.

### 6.2 Underflow / Small Epsilon

**Case 9**: bf16, epsilon=1e-12. When x values are near zero (e.g., |x| < 0.01):
- sum_sq/D is tiny (e.g., 1e-4/4099 ≈ 2.4e-8)
- sqrt(2.4e-8 + 1e-12) ≈ sqrt(2.4e-8) ≈ 1.55e-4
- y = x / 1.55e-4 * gamma — amplifies |x| by ~6400x
- For |x|=0.5: y ≈ 3200 * gamma. Within bf16 range.
- Golden also uses this computation. Bf16 tolerance (MARE < 7.81e-2) is generous.

**Case 17**: All zeros, epsilon=0.0001. sum_sq=0, sqrt(0+0.0001)=0.01. y=0/0.01*gamma=0. Correct regardless of accumulation path.

### 6.3 Catastrophic Cancellation

RmsNorm does not have a subtraction of near-equal quantities. The formula is `x / sqrt(mean(x^2) + eps) * gamma` — no cancellation risk. The only potential issue is if `mean(x^2) + eps` rounds to `mean(x^2)` when eps is tiny relative to mean(x^2), but this matches the golden's behavior (both produce the same rounding).

### 6.4 Precision: f32 Accumulation Path

Both paths accumulate in f32:
- **Builtin path**: The hardware `asctile.rms_norm` op handles accumulation internally in appropriate precision. For fp16 input, the Ascend RMSNorm instruction uses f32 accumulation. For the bf16 f16-hop, fp16 accumulation is sufficient given bf16's tolerance.
- **Manual path**: Explicit f32 promotion for squaring and accumulation. `acc = sum(xf * xf)` where xf is f32. Then `inv_rms = rsqrt(acc * inv_D + epsilon)` computed in f32. Normalization `y = xf * inv_rms_tile * gf` in f32, cast back to input dtype on output.

For fp32 cases, the entire computation stays in f32 (no promotion needed, input is already f32). The f32 precision threshold (MERE < 1.22e-4) requires full f32 accumulation, which both paths provide.

### 6.5 PlainValue / Scalar Considerations

In the manual path, `asctile.reduce_sum` returns a PlainValue scalar. The pinned v2 `asctile.rsqrt` rejects PlainValue input (requires LocalTensor). The workaround is to broadcast the scalar into an f32 tile before calling rsqrt:

```python
inv_rms_tile = asctile.rsqrt(
    asctile.full([tile_d], acc * inv_D + epsilon, dtype=asc.float32)
)
```

This creates a full-sized f32 tile (4*tile_d bytes in UB). The alternative `1.0 / asctile.sqrt(scalar)` may work if `asctile.sqrt` accepts PlainValue, but the broadcast approach is evidenced-safe.

## 7. Kernel Parameter Design

### 7.1 ConstExpr Parameters

| Parameter | Type | ConstExpr? | Rationale |
|-----------|------|------------|-----------|
| `cols` (builtin) | int | Yes | Used in copy_in tile shape `[rows_per_tile, cols]` |
| `rows_per_tile` | int | Yes | Used in copy_in tile shape |
| `tile_d` (manual) | int | Yes | Used in copy_in tile shape `[tile_d]` |
| `rows`, `S` | int | No | Runtime loop bounds |
| `D` | int | No | Runtime arithmetic |
| `num_d_tiles` | int | No | Runtime loop bound |
| `epsilon` | float | No | Arithmetic scalar |
| `inv_D` | float | No | Arithmetic scalar |
| `total_tiles` | int | No | Runtime loop bound |

### 7.2 Launch Configuration

- `cores = min(72, num_work_units)` where num_work_units depends on the path
- Builtin: `total_tiles = ceildiv(rows, rows_per_tile)`, `cores = min(72, total_tiles)`
- Manual: `cores = min(72, S)` — each core handles at least one row

## 8. Tail Handling

### 8.1 Row Tails (rows % rows_per_tile != 0)

Covered by `active_rows` computation:
```python
active_rows = rows_per_tile if row + rows_per_tile <= rows else rows - row
```
The `real_shape=[active_rows, cols]` tells copy_in/copy_out how many rows are actually valid.

### 8.2 D Tails (D % tile_d != 0)

Covered in the manual path's inner loop:
```python
n = tile_d if od + tile_d <= D else D - od
```
With `real_shape=[n]` for copy_in and copy_out.

Prime D values in the case matrix: 4097, 4099, 2049, 1021, 1023, 769, 373, 67, 2. All produce tails that the real_shape mechanism handles correctly with TILE_D=1024.

## 9. UB Budget Summary

### 9.1 Builtin Path

The builtin `asctile.rms_norm` is a monolithic hardware op. UB consumption is determined by the compiler's lowering of the RmsNormOp. The host must ensure:
- The loaded tile `[rows_per_tile, D]` fits in UB
- External live values (gamma tile, x tile, y tile) don't jointly overflow

Worst case: fp32, D=4096, rows_per_tile=1:
- gamma: 4*4096 = 16384 bytes (fp32 gamma)
- x: 4*1*4096 = 16384 bytes
- Builtin internals: unknown but hardware-optimized
- This is the edge case. The host rejects fp32 D>4096.

### 9.2 Manual Path (TILE_D=1024)

**Pass 1 peak UB** (within inner loop):
| Value | dtype | bytes |
|-------|-------|-------|
| x | input | 2048 (f16) or 4096 (f32) |
| xf | f32 | 4096 |
| xf*xf | f32 | 4096 |
| acc | scalar | ~0 |
| **Visible total** | | **~10240–12288** |
| With 1.6x | | **~16384–19661** |

**Pass 2 peak UB** (within inner loop):
| Value | dtype | bytes |
|-------|-------|-------|
| x | input | 2048–4096 |
| g | gamma | 2048–4096 |
| xf | f32 | 4096 |
| gf | f32 | 4096 |
| inv_rms_tile | f32 | 4096 |
| y | f32 | 4096 |
| **Visible total** | | **~18432–22528** |
| With 1.6x | | **~29491–36045** |

**Between passes**: inv_rms_tile persists = 4096 × 1.6 ≈ 6554 bytes.

**All well within 253952 bytes.**

## 10. Anti-Cheat Compliance

| Constraint | Compliance |
|-----------|-----------|
| All numerical work in @asctile.jit kernels | Yes — reduction, sqrt, multiply all in kernels |
| torch only for allocation/metadata/views | Yes — `torch.empty_like`, `.shape`, `.numel()`, `.contiguous()`, `.view()` only |
| No torch math ops | Yes — no torch.nn.functional, no tensor arithmetic |
| No caching / data_ptr tricks | Yes — fresh output allocated each call |
| Output is contiguous NPU tensor with correct shape/dtype | Yes — `torch.empty_like(x)` |
| No host-side dtype casts of device data | Yes — all casts via `.to()` inside kernels |

## 11. Module Structure

```python
import torch
import asc
import asctile
from ._pyasc_runtime import ensure_npu_platform

MAX_CORES = 72
MAX_TILE_ELEMENTS = 4096
MANUAL_TILE_D = 1024

@asctile.jit(reuse_alloc=1)
def _rms_norm_builtin_kernel(...):
    ...

@asctile.jit
def _rms_norm_manual_kernel(...):
    ...

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    ensure_npu_platform()
    ...
```

Note: `reuse_alloc=1` on the builtin kernel to reuse UB allocation across the repeated grid-stride tiles. The manual kernel does not use `reuse_alloc` because each row iteration already handles its own allocation pattern (single-buffered, no unroll).

## 12. Dispatch Decision Tree

```
rms_norm(x, gamma, epsilon):
    D = x.shape[-1]; S = numel / D
    copy_aligned = (D * element_size) % 32 == 0
    fits_builtin = D <= 4096 and not (fp32 and D > 4096)
    
    if copy_aligned and fits_builtin:
        rows_per_tile = max(1, 4096 // D)
        total_tiles = ceildiv(S, rows_per_tile)
        cores = min(72, total_tiles)
        launch _rms_norm_builtin_kernel[cores](x, gamma, out, S, D, total_tiles, rows_per_tile, epsilon)
    else:
        num_d_tiles = ceildiv(D, 1024)
        inv_D = 1.0 / D
        cores = min(72, S)
        launch _rms_norm_manual_kernel[cores](x, gamma, out, S, D, num_d_tiles, epsilon, inv_D, 1024)
```

## 13. Edge Cases & Degenerate Inputs

| Scenario | Behavior |
|----------|----------|
| S=0 or D=0 (empty tensor) | Host returns `torch.empty_like(x)` immediately |
| All zeros (case 17) | sum_sq=0, rsqrt(0+eps)=1/sqrt(eps). y=0*inv_rms*gamma=0. Correct. |
| D=2 (case 15) | Manual path TILE_D=1024 with real_shape=[2]. Padded lanes zero, no contribution to sum. |
| Large S, small D (case 15: S=1000003, D=2) | Manual path: 72 cores, each handles ~13889 rows. Each row is trivially fast (one D-tile). |
| Small S, large D (case 17: S=231, D=4096) | Builtin path: rows_per_tile=1, total_tiles=231, cores=min(72,231)=72. |
| Non-contiguous input | Host calls `.contiguous()` |
| Very large epsilon (1e-3, case 19) | sqrt(mean_sq + 0.001). When mean_sq >> 0.001, epsilon is absorbed. No issue. |
| Very small epsilon (1e-12, case 9) | sqrt(mean_sq + 1e-12). In f32, 1e-12 has full precision. When mean_sq >> 1e-12, epsilon absorbed. |

DESIGN_DONE.
