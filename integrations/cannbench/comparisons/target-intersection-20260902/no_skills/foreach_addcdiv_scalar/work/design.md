# Design: ForeachAddcdivScalar (pyasc asc2 JIT)

## Operator Summary

**Formula**: `y_i = x1_i + (x2_i / x3_i) * scalar`  (per tensor-list entry)

- 3 TensorList inputs (x1, x2, x3), 1 TensorList output (y)
- Lists have equal length L (1–4 in the 20 cases)
- Corresponding tensors share shape and dtype
- Supported dtypes: float32, float16, bfloat16
- scalar is a runtime float (includes 0.0, ±0.5, ±1.0, 1.5, 2.0, inf, nan)

## Architecture

### Host wrapper: `foreach_addcdiv_scalar(x1, x2, x3, scalar)`

1. `ensure_npu_platform()`
2. Python `range(len(x1))` loop — one kernel launch per list entry
3. Per entry:
   - `t1 = x1[i].contiguous()`, same for t2, t3
   - `out = torch.empty_like(t1)`
   - `size = t1.numel()`
   - `num_tiles = asc.ceildiv(size, TILE)` ; `cores = min(72, num_tiles)`
   - Launch `_addcdiv_kernel[cores](t1, t2, t3, out, size, num_tiles, float(scalar), TILE)`
4. Return list of output tensors

### Single `@asc2.jit` kernel: `_addcdiv_kernel`

**Signature**:
```
_addcdiv_kernel(x1_ptr: GlobalAddress, x2_ptr: GlobalAddress,
                x3_ptr: GlobalAddress, out_ptr: GlobalAddress,
                size: int, num_tiles: int, scalar: float,
                tile_size: ConstExpr[int])
```

**Body** (grid-stride pattern, `unroll_factor=2`):
```
x1_gm = asc2.global_tensor(x1_ptr, [size])
x2_gm = asc2.global_tensor(x2_ptr, [size])
x3_gm = asc2.global_tensor(x3_ptr, [size])
out_gm = asc2.global_tensor(out_ptr, [size])

for t in asc2.range(block_idx(), num_tiles, block_num(), unroll_factor=2):
    off = t * tile_size
    n   = tile_size if off + tile_size <= size else size - off

    x1  = asc2.copy_in(x1_gm, [off], [tile_size], real_shape=[n])
    x2  = asc2.copy_in(x2_gm, [off], [tile_size], real_shape=[n])
    x3  = asc2.copy_in(x3_gm, [off], [tile_size], real_shape=[n])

    x1f = x1.to(float32)       # identity for f32, promote for f16/bf16
    x2f = x2.to(float32)
    x3f = x3.to(float32)

    div = x2f / x3f
    mul = div * scalar          # scalar on RIGHT (tile * scalar)
    res = x1f + mul

    asc2.copy_out(res.to(x1.dtype), out_gm, [off], real_shape=[n])
```

## TILE selection

**TILE = 1024** (single tile size for all cases and dtypes).

### UB budget analysis (TILE=1024, unroll=2)

| Category | Values | Bytes/value | Subtotal |
|----------|--------|-------------|----------|
| f16/bf16 input tiles (x1,x2,x3) | 3 | 2×1024×2 | 12 288 |
| f32 promoted (x1f,x2f,x3f) | 3 | 4×1024×2 | 24 576 |
| f32 temps (div, mul, res) | 3 | 4×1024×2 | 24 576 |
| f16/bf16 output cast | 1 | 2×1024×2 | 4 096 |
| **Naive total** | | | **65 536** |
| ×1.6 compiler overhead | | | **≈ 104 858** |
| **Budget** | | | **253 952** |

Headroom ≈ 2.4×. For f32 inputs the f16 tiles disappear but the f32 tiles grow to 7 values: 7×4×1024×2×1.6 ≈ 91 750 — still safe.

TILE=2048 with unroll=2 would be ≈ 209 715 for f32 (safe) but ≈ 262 144 for f16/bf16 with compiler overhead (OVERFLOW). **TILE=1024 is the safe choice**.

## Core utilization check

All 20 cases have tensor element counts ≥ 1 046 529 (case 6: 1023×1023).
At TILE=1024: num_tiles ≥ 1022, cores = min(72, 1022) = **72 always**. Full core utilisation.

## Case-by-case coverage

| # | L | Shape | dtype | Elements | scalar | Notes |
|---|---|-------|-------|----------|--------|-------|
| 1 | 2 | [1024,1024] | f32 | 1 048 576 | 1.0 | Standard f32, TILE=1024 perfect alignment |
| 2 | 3 | [2048,2048] | f16 | 4 194 304 | 1.0 | f16→f32 promotion, 3 list entries |
| 3 | 1 | [4096,4096] | bf16 | 16 777 216 | 1.0 | bf16→f32, large tensor |
| 4 | 1 | [8192,8192] | f32 | 67 108 864 | 0.5 | Largest f32 tensor, scalar=0.5 |
| 5 | 2 | [8192,4096] | f16 | 33 554 432 | 2.0 | Large f16, scalar=2.0 |
| 6 | 1 | [1023,1023] | bf16 | 1 046 529 | -1.0 | Non-power-of-2 dims, tail tile exercise |
| 7 | 1 | [1009,1021] | f32 | 1 030 189 | 1.5 | Prime-ish dims, tail handling |
| 8 | 1 | [1537,769] | f16 | 1 181 953 | 1.0 | Non-aligned dims |
| 9 | 2 | [363,367,373] | bf16 | 49 734 403 (×2 entries) | 1.0 | 3D, ~50M elements per tensor |
| 10 | 1 | [2049,513] | f32 | 1 051 137 | 1.0 | Wide value range [-65504,65504] |
| 11 | 2 | [3,7,13,4001] | f16 | 1 092 273 (×2) | 1.0 | 4D tensor |
| 12 | 1 | [1000003] | f32 | 1 000 003 | inf | **inf scalar**: `(x2/x3)*inf` → inf where x2≠0, nan where x2=0; IEEE propagation |
| 13 | 1 | [11,13,17,67,67] | bf16 | 10 895 489 | nan | **nan everywhere**: inputs nan, scalar nan → output nan; IEEE propagation |
| 14 | 1 | [3,7,11,13,1013] | f32 | 3 096 231 | 1.0 | x1,x2 all zero → `0 + (0/x3)*1 = 0` |
| 15 | 2 | [512,2049] | f32 | 1 049 088 (×2) | 1.0 | Standard f32 |
| 16 | 4 | [255,8193] | bf16 | 2 089 215 (×4 entries) | 1.0 | 4 list entries, largest list |
| 17 | 1 | [4097,511] | f16 | 2 093 567 | 0.0 | **scalar=0**: `y = x1 + 0 = x1`; div result irrelevant (×0) |
| 18 | 2 | [2,511,2049] | f32 | 2 093 958 (×2) | 2.0 | 3D f32 |
| 19 | 1 | [4,255,2049] | bf16 | 2 089 980 | -0.5 | Negative scalar |
| 20 | 1 | [2,3,17,1024,101] | f32 | 10 690 560 | 1.5 | 5D tensor |

All 20 cases handled by the single kernel + host loop.

## Numerical stability

### Division precision
- All computation in f32 (even for f16/bf16 inputs) via `.to(asc.float32)`
- Matches the golden's promotion strategy exactly
- Division `x2f / x3f` in f32 gives full 23-bit mantissa precision
- f16 threshold is 2^-10 ≈ 9.77e-4; f32 division gives ~1e-7 error → passes easily
- bf16 threshold is 2^-7 ≈ 7.8e-3; even more margin
- f32 threshold is 2^-13 ≈ 1.22e-4; f32 division native precision → passes

### No catastrophic cancellation
- The formula `x1 + (x2/x3) * scalar` involves only one division, one multiply, one addition
- No subtraction of nearly-equal quantities
- No exponential/logarithmic functions that could amplify errors

### IEEE special values
- **inf scalar** (case 12): `(x2/x3) * inf` → ±inf (if x2/x3 ≠ 0), nan (if x2/x3 = 0); `x1 + inf` → inf; matches golden
- **nan scalar** (case 13): `(x2/x3) * nan` → nan; `x1 + nan` → nan; matches golden
- **nan inputs** (case 13): all propagate correctly through IEEE arithmetic
- **inf inputs** (case 12): range [-inf, inf] → values include ±inf; IEEE arithmetic handles correctly
- **scalar=0.0** (case 17): `div * 0.0 = 0.0` (finite div); `x1 + 0.0 = x1` — correct
- **x1=0, x2=0** (case 14): `0 + (0/x3)*1.0 = 0 + 0 = 0` — correct
- No host-side special casing needed; all IEEE propagation is correct

## Anti-cheat compliance

- All numerical work inside `@asc2.jit` kernel on NPU
- torch used ONLY for: `.contiguous()`, `torch.empty_like()`, `.numel()`, `.dtype`
- No `torch.add`, `torch.div`, `a + b` on tensors, or any compute via torch
- Each output is a freshly allocated tensor (not a view of input)
- No caching or data_ptr-based shortcuts

## Summary of constants

| Parameter | Value |
|-----------|-------|
| TILE | 1024 |
| unroll_factor | 2 |
| Max cores | 72 |
| Compute dtype | f32 (always) |
| Kernel launches | L per call (1 per list entry) |
