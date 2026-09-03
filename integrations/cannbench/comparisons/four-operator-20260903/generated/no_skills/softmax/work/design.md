# Softmax Design

## Runtime

compiler-team/pyasc v2 @ `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`; module exports `asctile`.

## Layout decomposition

Normalized `dim` (handle negatives: `dim += rank` when `< 0`). Factor shape into:

- `outer = prod(shape[:dim])`
- `axis = shape[dim]`
- `inner = prod(shape[dim+1:])`

## Dispatch

| condition | path |
|-----------|------|
| `inner == 1` | row kernel (1 launch) |
| `inner > 1` | transpose-fwd kernel, row kernel, transpose-bwd kernel (3 launches) |

## Kernels

### `_softmax_row_kernel`

- 1-D `global_tensor` of length `outer * axis`.
- Grid-stride over `outer` slices (one slice = one softmax unit).
- `asctile.copy_in(x_gm, [slice_off], [TILE], real_shape=[axis], pad_value=NEG_INF)` with `_NEG_INF = float('-inf')` module-level constant.
- Promote to f32: `xf = x_tile.to(asc.float32)`.
- `max_s = asctile.reduce_max(xf)` (scalar; padding is `-inf`, excluded automatically).
- `shifted = xf - max_s` (scalar on right).
- `exp_t = asctile.exp(shifted)` (padding: `exp(-inf)=0`).
- `sum_s = asctile.reduce_sum(exp_t)` (scalar; padding contributes 0).
- `result = exp_t / sum_s` (scalar on right; padding: `0/sum = 0`).
- `asctile.copy_out(result.to(input_dtype), out_gm, [slice_off], real_shape=[axis])`.

Distinct live f32 tiles: `xf`, `shifted`, `exp_t`, `result` = 4.

### `_transpose_kernel`

- 2-D `global_tensor` `[outer_dim * dim_a, dim_b]` (source) and `[outer_dim * dim_b, dim_a]` (dest).
- Grid-stride over linearized `(outer_idx, a_block, b_block)` triples.
- `asctile.copy_in(src_gm, [r_off, c_off], [TA, TB], real_shape=[ra, rb])`.
- `tile_t = asctile.transpose(tile)` (`[TA, TB]` -> `[TB, TA]`).
- `asctile.copy_out(tile_t, dst_gm, [c_off, r_off], real_shape=[rb, ra])`.

Tile sizes: `TA=128, TB=128`, `unroll_factor=1`. 2-D f32 tiles: `128*128*4 = 65536` each, two live = 131072, x1.6 = 209715 < 253952.

## TILE / UNROLL buckets

| axis range | `TILE` | `UNROLL` | UB estimate (f32, 4 tiles) |
|------------|--------|----------|-----------------------------|
| 1-2048 | 2048 | 2 | 2 * 4 * 4 * 2048 * 1.6 = 104858 |
| 2049-4096 | 4096 | 2 | 2 * 4 * 4 * 4096 * 1.6 = 209715 |
| 4097-8448 | 8448 | 1 | 1 * 4 * 4 * 8448 * 1.6 = 216269 |

Budget: 253952 bytes. All buckets fit.

`select_tile(axis)`: host-side function, returns `(TILE, UNROLL)` tuple.

## Case coverage

| case | shape | dim | outer | axis | inner | path | TILE |
|------|-------|-----|-------|------|-------|------|------|
| 1 | [1024,1024] | -1 | 1024 | 1024 | 1 | row | 2048 |
| 2 | [2048,2048] | -1 | 2048 | 2048 | 1 | row | 2048 |
| 3 | [4096,4096] | -1 | 4096 | 4096 | 1 | row | 4096 |
| 4 | [8192,8192] | 0 | 1 | 8192 | 8192 | transpose | 8448 |
| 5 | [8192,8192] | 1 | 8192 | 8192 | 1 | row | 8448 |
| 6 | [31,67,127,257] | 2 | 2077 | 127 | 257 | transpose | 2048 |
| 7 | [1023,2047] | -1 | 1023 | 2047 | 1 | row | 2048 |
| 8 | [2049,4097] | -1 | 2049 | 4097 | 1 | row | 8448 |
| 9 | [127,257,1023] | -2 | 127 | 257 | 1023 | transpose | 2048 |
| 10 | [1009,1021] | -1 | 1009 | 1021 | 1 | row | 2048 |
| 11 | [367,373,379] | 1 | 367 | 373 | 379 | transpose | 2048 |
| 12 | [11,13,17,4001] | -1 | 2431 | 4001 | 1 | row | 4096 |
| 13 | [1000003,2] | -1 | 1000003 | 2 | 1 | row | 2048 |
| 14 | [11,13,17,67,67] | -1 | 162877 | 67 | 1 | row | 2048 |
| 15 | [3,7,11,13,1013] | -1 | 3003 | 1013 | 1 | row | 2048 |
| 16 | [512,2049] | 0 | 1 | 512 | 2049 | transpose | 2048 |
| 17 | [255,8193] | 1 | 255 | 8193 | 1 | row | 8448 |
| 18 | [2,511,2049] | -1 | 1022 | 2049 | 1 | row | 4096 |
| 19 | [4,255,2049] | 1 | 4 | 255 | 2049 | transpose | 2048 |
| 20 | [2,3,17,1024,101] | 3 | 102 | 1024 | 101 | transpose | 2048 |

## Numerical stability

### Core algorithm

`exp(x_i - max(x)) / sum(exp(x_j - max(x)))`. The max subtraction guarantees no `exp` overflow for finite inputs since `x_i - max <= 0`.

### pad_value=-inf correctness chain

| step | real elements | padding |
|------|--------------|---------|
| copy_in with pad=-inf | x_i | -inf |
| cast to f32 | x_f32 | -inf |
| reduce_max | max(x_f32) (excludes -inf) | - |
| subtract max | x_i - max | -inf - max = -inf (finite max) or NaN (+inf max) |
| exp | exp(x_i - max) | exp(-inf) = 0 |
| reduce_sum | sum over reals (padding = 0) | - |
| divide | exp / sum | 0 / sum = 0 |
| copy_out real_shape | written | discarded |

### Special values (per spec)

| input slice | max | output |
|------------|-----|--------|
| contains +inf | +inf | NaN (inf-inf = NaN propagates) |
| all -inf | -inf | NaN (-inf-(-inf) = NaN) |
| -inf + finite | max(finite) | -inf -> 0, finite normalize |
| contains NaN | NaN or max(finite) | NaN propagates through arithmetic |

All handled by IEEE propagation; no host or kernel special-casing.

### Precision thresholds

f32 compute throughout. Thresholds:

| dtype | MERE threshold | MARE threshold | f32 precision |
|-------|---------------|---------------|---------------|
| float16 | 2^-10 ~ 9.8e-4 | ~9.8e-3 | ~1.2e-7 |
| bfloat16 | 2^-7 ~ 7.8e-3 | ~7.8e-2 | ~1.2e-7 |
| float32 | 2^-13 ~ 1.2e-4 | ~1.2e-3 | ~1.2e-7 |

f32 meets all thresholds with >3 orders-of-magnitude margin.

## UB budget rationale

Calibrated overhead factor 1.6x from sigmoid reference (6 visible f32 tiles at TILE=2048, unroll=2 -> 155648 bytes; naive = 6*4*2048*2 = 98304; ratio = 1.58).

Softmax uses 4 distinct f32 tiles. All bucket UB estimates are below 253952 bytes by margins of 48K-149K bytes.

## JIT compilation count

3 softmax (`select_tile` returns 3 unique `(TILE,UNROLL)` pairs) + 1 transpose = 4 total compilations. Each (TILE, UNROLL) is a distinct ConstExpr combination triggering separate codegen.

## Host logic

```
softmax(x, dim=-1):
  ensure_npu_platform()
  x = x.contiguous() if not x.is_contiguous()
  rank = len(x.shape)
  dim = dim + rank if dim < 0 else dim
  outer = prod(shape[:dim]);  axis = shape[dim];  inner = prod(shape[dim+1:])
  if inner == 1:
    out = empty_like(x)
    TILE, UNROLL = select_tile(axis)
    cores = min(72, outer)
    _softmax_row_kernel[cores](x, out, outer, axis, outer, TILE, UNROLL)
    return out
  else:
    tmp1 = empty(N, dtype=x.dtype, device=x.device)  # N = outer*axis*inner
    tmp2 = empty_like(tmp1)
    num_t_blocks = outer * ceil(axis/TA) * ceil(inner/TB)
    cores_t = min(72, num_t_blocks)
    _transpose_kernel[cores_t](x, tmp1, outer, axis, inner, num_t_blocks, TA, TB)
    soft_out = empty_like(tmp1)
    TILE, UNROLL = select_tile(axis)
    outer_soft = outer * inner
    cores_s = min(72, outer_soft)
    _softmax_row_kernel[cores_s](tmp1, soft_out, outer_soft, axis, outer_soft, TILE, UNROLL)
    out = empty_like(x)
    # backward transpose: [outer*inner, axis] -> [outer*axis, inner]
    # = transpose of [outer, inner, axis] -> [outer, axis, inner]
    # same kernel: src=[outer, inner, axis], dest=[outer, axis, inner]
    # parameters: outer_dim=outer, dim_a=inner, dim_b=axis
    num_b_blocks = outer * ceil(inner/TA) * ceil(axis/TB)
    cores_b = min(72, num_b_blocks)
    _transpose_kernel[cores_b](soft_out, out, outer, inner, axis, num_b_blocks, TA, TB)
    return out
```

## Anti-cheat

- All numerical compute in `@asctile.jit` kernels on NPU.
- torch usage limited to: `empty_like`, `empty`, `.shape`, `.numel()`, `.is_contiguous()`, `.contiguous()`, `.view()`.
- No torch math ops, no data pointer caching.
- Output is a fresh contiguous NPU tensor with correct shape/dtype.

## Tail handling

- Row kernel: `real_shape=[axis]` in copy_in/copy_out. `TILE >= axis` (guaranteed by bucket selection). Padding filled with `-inf` (operation-neutral for max, propagates to 0 through exp).
- Transpose kernel: 2-D `real_shape=[ra, rb]` for partial tiles. Padding filled with default 0 (irrelevant -- discarded on copy_out).

DESIGN_DONE.
