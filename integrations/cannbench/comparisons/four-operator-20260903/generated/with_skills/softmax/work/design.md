# Softmax Design

## 1. Operator summary

`softmax(Tensor x, int dim=-1) -> Tensor y`

Numerically stable softmax: `y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))` along `dim`.

Dtypes: float16, float32, bfloat16. Ranks: 2-5. Output shape/dtype same as input.

## 2. Case matrix

| case | shape | dtype | dim | dim_norm | outer | axis | inner | value_range | path |
|------|-------|-------|-----|----------|-------|------|-------|-------------|------|
| 1 | [1024,1024] | f16 | -1 | 1 | 1024 | 1024 | 1 | [-1,1] | full-row |
| 2 | [2048,2048] | f32 | -1 | 1 | 2048 | 2048 | 1 | [-2,2] | full-row |
| 3 | [4096,4096] | bf16 | -1 | 1 | 4096 | 4096 | 1 | [-3,3] | full-row |
| 4 | [8192,8192] | f16 | 0 | 0 | 1 | 8192 | 8192 | [-10,10] | **global-xpose** |
| 5 | [8192,8192] | f32 | 1 | 1 | 8192 | 8192 | 1 | [-100,100] | full-row |
| 6 | [31,67,127,257] | bf16 | 2 | 2 | 2077 | 127 | 257 | [-5,5] | local-xpose |
| 7 | [1023,2047] | f16 | -1 | 1 | 1023 | 2047 | 1 | [-0.1,0.1] | full-row |
| 8 | [2049,4097] | f32 | -1 | 1 | 2049 | 4097 | 1 | [-1,1] | full-row |
| 9 | [127,257,1023] | bf16 | -2 | 1 | 127 | 257 | 1023 | [-0.5,0.5] | local-xpose |
| 10 | [1009,1021] | f16 | -1 | 1 | 1009 | 1021 | 1 | [-1,2] | full-row |
| 11 | [367,373,379] | f32 | 1 | 1 | 367 | 373 | 379 | [-50,100] | local-xpose |
| 12 | [11,13,17,4001] | bf16 | -1 | 3 | 2431 | 4001 | 1 | [-3,6] | full-row |
| 13 | [1000003,2] | f16 | -1 | 1 | 1000003 | 2 | 1 | inf/nan | full-row |
| 14 | [11,13,17,67,67] | f32 | -1 | 4 | 162371 | 67 | 1 | nan | full-row |
| 15 | [3,7,11,13,1013] | bf16 | -1 | 4 | 3003 | 1013 | 1 | zeros | full-row |
| 16 | [512,2049] | f16 | 0 | 0 | 1 | 512 | 2049 | [-0.5,0.5] | local-xpose |
| 17 | [255,8193] | f32 | 1 | 1 | 255 | 8193 | 1 | [-1000,1000] | full-row |
| 18 | [2,511,2049] | bf16 | -1 | 2 | 1022 | 2049 | 1 | [-0.2,0.2] | full-row |
| 19 | [4,255,2049] | f16 | 1 | 1 | 4 | 255 | 2049 | [-65504,65504] | local-xpose |
| 20 | [2,3,17,1024,101] | f32 | 3 | 3 | 102 | 1024 | 101 | [-20,40] | local-xpose |

## 3. Dispatch

### 3.1 Host dispatch

```
softmax(x, dim=-1):
    1. ensure_npu_platform()
    2. x = x.contiguous() if not already
    3. rank = x.dim(); if dim < 0: dim += rank
    4. outer = prod(shape[:dim]) or 1
       axis_size = shape[dim]
       inner = prod(shape[dim+1:]) or 1
    5. out = torch.empty_like(x)
    6. if inner == 1:
         full_row_softmax(out, x, outer, axis_size, dtype)
       elif axis_size * ALIGN_F16 * 2 > 253000:
         global_transpose_path(out, x, outer, axis_size, inner, dtype)
       else:
         local_transpose_softmax(out, x, outer, axis_size, inner, dtype)
    7. return out
```

Only case 4 (`[8192,8192]` f16, dim=0, axis=8192, inner=8192) hits the global-transpose path. All other inner>1 cases fit the local-transpose kernel's UB budget.

### 3.2 Kernel inventory

| kernel | purpose | launch signature |
|--------|---------|-----------------|
| `_softmax_full_row` | full-row softmax for inner==1 | `[cores](x, out, total_rows, axis_size, axis_padded, num_row_tiles, tile_rows_CE, unroll_CE)` |
| `_softmax_full_row_bf16` | same but bf16→f32→softmax→bf16 | same signature |
| `_softmax_xpose_local` | load [axis,inner_tile], transpose, softmax, transpose back, store | `[cores](x, out, outer, axis_size, inner, axis_padded, inner_tile_CE, inner_padded_CE, num_work, unroll_CE)` |
| `_softmax_xpose_local_bf16` | same but bf16 compute path | same signature |
| `_transpose_2d_fwd` | block-transpose [R,C]→[C,R] for global-xpose path | `[cores](src, dst, R, C, R_padded, C_padded, tile_r, tile_c, num_tiles, unroll_CE)` |
| `_transpose_2d_bwd` | reverse: [C,R]→[R,C] (identical kernel, swapped shapes) | same |

All kernels use `import asctile`, `@asctile.jit`, `asctile.global_tensor`, `asctile.copy_in`, `asctile.copy_out`, `asctile.range`.

## 4. Kernel details

### 4.1 Full-row kernel (`_softmax_full_row`)

**Applies to**: cases 1,2,5,7,8,10,17 (f16/f32 with inner==1)

**View**: x as 2D `[outer, axis_size]`, global tensor `[outer, axis_padded]`.

**Algorithm**:
```python
@asctile.jit
def _softmax_full_row(x_ptr, out_ptr, total_rows, num_cols,
                       axis_padded, num_row_tiles,
                       tile_rows: asc.ConstExpr[int],
                       tile_cols: asc.ConstExpr[int],
                       unroll: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [total_rows, axis_padded])
    out_gm = asctile.global_tensor(out_ptr, [total_rows, axis_padded])
    for t in asctile.range(asctile.block_idx(), num_row_tiles,
                           asctile.block_num(), unroll_factor=unroll):
        row_off = t * tile_rows
        real_r = tile_rows if row_off + tile_rows <= total_rows else total_rows - row_off
        tile = asctile.copy_in(x_gm, [row_off, 0], [tile_rows, tile_cols],
                               real_shape=[real_r, num_cols],
                               pad_value=float('-inf'))
        result = asctile.softmax(tile)
        asctile.copy_out(result, out_gm, [row_off, 0],
                         real_shape=[real_r, num_cols])
```

**BF16 variant** (`_softmax_full_row_bf16`): identical except after `copy_in`, cast `tile` to f32 with `tile_f32 = tile.to(asc.float32)`, apply `asctile.softmax(tile_f32)`, cast result back `result.to(asc.bfloat16)` before `copy_out`.

**Rationale for `pad_value=float('-inf')`**: padded column lanes in the softmax dimension receive -inf. In max computation, -inf is ignored (max of reals vs -inf = reals). In exp(-inf - max) = 0, so padded lanes contribute 0 to the sum. Division: 0/sum = 0. Padded lanes never stored due to `real_shape` on `copy_out`.

### 4.2 Local-transpose kernel (`_softmax_xpose_local`)

**Applies to**: cases 6,9,11,16,19,20 (inner>1, axis*min_inner_tile fits UB)

**View**: x as 3D `[outer, axis_size, inner]`. Flattened to 2D global tensor `[outer*axis_size, inner]`.

**Algorithm**:
```python
@asctile.jit
def _softmax_xpose_local(x_ptr, out_ptr,
                          outer_size, axis_size, inner_size,
                          axis_padded, inner_padded,
                          num_work,
                          inner_tile: asc.ConstExpr[int],
                          tile_cols: asc.ConstExpr[int],
                          unroll: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [outer_size * axis_size, inner_size])
    out_gm = asctile.global_tensor(out_ptr, [outer_size * axis_size, inner_size])
    num_inner_tiles = (inner_size + inner_tile - 1) // inner_tile
    for w in asctile.range(asctile.block_idx(), num_work,
                           asctile.block_num(), unroll_factor=unroll):
        outer_idx = w // num_inner_tiles
        inner_idx = w - outer_idx * num_inner_tiles
        row_off = outer_idx * axis_size
        col_off = inner_idx * inner_tile
        real_inner = inner_tile if col_off + inner_tile <= inner_size else inner_size - col_off
        tile = asctile.copy_in(x_gm, [row_off, col_off],
                               [axis_size, tile_cols],
                               real_shape=[axis_size, real_inner],
                               pad_value=float('-inf'))
        tile_t = tile.transpose()
        result_t = asctile.softmax(tile_t)
        result = result_t.transpose()
        asctile.copy_out(result, out_gm, [row_off, col_off],
                         real_shape=[axis_size, real_inner])
```

**Note**: `tile_cols` is `inner_tile` padded to 32-byte alignment for the copy_in/copy_out last dimension. After transpose, the tile shape is `[tile_cols, axis_size]`. `asctile.softmax` operates on the last dim (axis_size). The padded rows `[real_inner, tile_cols)` contain pad_value=-inf which yields uniform softmax output (harmless since masked by `real_shape` on store). Padded axis columns `[axis_size, axis_padded)` are irrelevant here because the first dim of copy_in is not padded — the full axis_size rows are loaded.

Wait — correction: the copy_in shape is `[axis_size, tile_cols]` (no axis padding). After transpose: `[tile_cols, axis_size]`. Softmax on last dim = axis_size. No padding in the softmax dimension. This is correct because axis_size elements are all real data loaded from global memory.

**BF16 variant**: cast input tile to f32 before transpose, apply softmax on f32, cast result back to bf16 after transpose-back.

### 4.3 Global-transpose path (case 4 only)

**Case 4**: `[8192,8192]` f16, dim=0, outer=1, axis=8192, inner=8192.

**Problem**: `[8192, 16]` in f16 = 262144 bytes > 253952 UB limit. Cannot use local-transpose path.

**Solution**: 3-step pipeline:
1. `_transpose_2d_fwd`: transpose x `[8192, 8192]` → temp `[8192, 8192]` in global memory
2. `_softmax_full_row`: softmax on temp viewed as `[8192, 8192]` (last dim) → result in temp2
3. `_transpose_2d_bwd`: transpose temp2 `[8192, 8192]` → out `[8192, 8192]`

**Transpose kernel** (`_transpose_2d_fwd`):
```python
@asctile.jit
def _transpose_2d(src_ptr, dst_ptr, R, C, R_padded, C_padded,
                   tile_r: asc.ConstExpr[int], tile_c: asc.ConstExpr[int],
                   num_tiles, unroll: asc.ConstExpr[int]):
    src_gm = asctile.global_tensor(src_ptr, [R_padded, C_padded])
    dst_gm = asctile.global_tensor(dst_ptr, [C_padded, R_padded])
    num_c_tiles = (C + tile_c - 1) // tile_c
    for t in asctile.range(asctile.block_idx(), num_tiles,
                           asctile.block_num(), unroll_factor=unroll):
        rt = t // num_c_tiles
        ct = t - rt * num_c_tiles
        r_off = rt * tile_r
        c_off = ct * tile_c
        real_r = tile_r if r_off + tile_r <= R else R - r_off
        real_c = tile_c if c_off + tile_c <= C else C - c_off
        block = asctile.copy_in(src_gm, [r_off, c_off], [tile_r, tile_c],
                                real_shape=[real_r, real_c], pad_value=0.0)
        block_t = block.transpose()
        asctile.copy_out(block_t, dst_gm, [c_off, r_off],
                         real_shape=[real_c, real_r])
```

Uses `tile_r=128, tile_c=128` for f16 (128*128*2*2 = 65536 bytes for input+output, safe).

**Why a separate temp buffer**: anti-cheat rules forbid torch-based numerical ops. The transpose kernel does all movement. The softmax kernel does all computation. All three kernels are `@asctile.jit`.

**Memory**: 2 temp buffers of size `8192*8192*2` = 128 MB. The NPU should have sufficient memory for this. Alternatively, reuse one buffer by doing softmax in-place on temp.

**Revised 3-step for case 4**:
1. temp1 = torch.empty_like(x)  # [8192, 8192]
2. transpose_fwd: x → temp1  (8192x8192 block transpose)
3. temp2 = torch.empty_like(x)
4. full_row_softmax: temp1 (viewed as [8192, 8192]) → temp2
5. transpose_bwd: temp2 → out

## 5. Tile sizing strategy

### 5.1 Alignment

```
dtype       align_elem    32/itemsize
float32     8             8
float16     16            16
bfloat16    16            16
```

`axis_padded = ceil(axis_size / align_elem) * align_elem` (for copy_in last dim in full-row).
`inner_padded = ceil(inner_tile / align_elem) * align_elem` (for copy_in last dim in xpose).

### 5.2 Full-row tile sizing

| axis range | dtype | tile_cols | tile_rows | unroll | est UB |
|-----------|-------|-----------|-----------|--------|--------|
| ≤64 | f16 | 64 | 256 | 2 | ~33KB |
| ≤64 | f32 | 64 | 128 | 2 | ~66KB |
| 65-256 | f16 | 256 | 128 | 2 | ~66KB |
| 65-256 | f32 | 256 | 64 | 2 | ~131KB |
| 257-1024 | f16 | 1024 | 32 | 2 | ~131KB |
| 257-1024 | f32 | 1024 | 16 | 2 | ~131KB |
| 1025-2048 | f16 | 2048 | 16 | 2 | ~131KB |
| 1025-2048 | f32 | 2048 | 8 | 1 | ~131KB |
| 2049-4096 | f16 | 4096 | 8 | 2 | ~131KB |
| 2049-4096 | f32 | 4096 | 4 | 1 | ~131KB |
| 4097-8192 | f16 | 8192 | 4 | 1 | ~131KB |
| 4097-8192 | f32 | 8200 | 2 | 1 | ~131KB |
| 8193+ | f16 | 8208 | 4 | 1 | ~131KB |
| 8193+ | f32 | 8200 | 2 | 1 | ~131KB |

BF16 full-row: same as f32 but add ~2x for bf16 input tile. Use tile_rows halved vs f32.

These are initial estimates. Actual UB measured by the exact-v2 compile gate may differ. Fallback: halve tile_rows if UB overflow.

### 5.3 Local-transpose tile sizing

| axis range | dtype | inner_tile | inner_padded | est tile bytes | est UB (6x) |
|-----------|-------|------------|--------------|----------------|-------------|
| ≤128 | f16/bf16 | 64 | 64 | 128*64*2=16KB | ~98KB |
| ≤128 | f32 | 32 | 32 | 128*32*4=16KB | ~98KB |
| 129-256 | f16/bf16 | 32 | 32 | 256*32*2=16KB | ~98KB |
| 129-256 | f32 | 16 | 16 | 256*16*4=16KB | ~98KB |
| 257-512 | f16/bf16 | 32 | 32 | 512*32*2=32KB | ~197KB |
| 257-512 | f32 | 16 | 16 | 512*16*4=32KB | ~197KB |
| 513-1024 | f16 | 16 | 16 | 1024*16*2=32KB | ~197KB |
| 513-1024 | f32 | 8 | 8 | 1024*8*4=32KB | ~197KB |
| 1025-4096 | f16 | 8 | 16 | 4096*16*2=128KB | would overflow |
| 1025-4096 | f32 | 4 | 8 | 4096*8*4=128KB | would overflow |

Cases requiring axis > 1024 with inner > 1: **only case 4** (axis=8192, inner=8192, f16) — routed to global-transpose path. All other local-xpose cases have axis ≤ 1024.

Concrete per-case tile sizes for local-xpose:

| case | axis | inner | dtype | inner_tile | inner_padded | axis_padded (n/a) |
|------|------|-------|-------|------------|--------------|-------------------|
| 6 | 127 | 257 | bf16 | 32 | 32 | — |
| 9 | 257 | 1023 | bf16 | 16 | 16 | — |
| 11 | 373 | 379 | f32 | 16 | 16 | — |
| 16 | 512 | 2049 | f16 | 16 | 16 | — |
| 19 | 255 | 2049 | f16 | 16 | 16 | — |
| 20 | 1024 | 101 | f32 | 8 | 8 | — |

**UB check for case 6 (worst local-xpose for bf16)**:
- axis=127, inner_tile=32, bf16 input: 127*32*2 = 8128 bytes
- Cast to f32 for softmax: 127*32*4 = 16256 (but after transpose: [32, 127])
- After transpose [32, 127]: softmax on axis 127 → ~5 * 32 * 127 * 4 = 81280
- Total (~input + f32 cast + softmax internals) ≈ 105KB → safe with unroll=1

**UB check for case 20 (largest axis in local-xpose)**:
- axis=1024, inner_tile=8, f32: 1024*8*4 = 32768 bytes input
- After transpose [8, 1024]: softmax on 1024 → ~5 * 8 * 1024 * 4 = 163840
- Total ≈ 196KB → tight but under 253952
- Use unroll=1 for safety

## 6. Special values and numerical behavior

### 6.1 Spec-mandated behavior (from `torch.nn.functional.softmax`)

| scenario | expected output |
|----------|----------------|
| Slice contains +inf (with any mix) | entire slice = NaN (inf - inf = NaN in `x - max`) |
| Slice all -inf | entire slice = NaN (max = -inf, -inf - (-inf) = NaN) |
| Slice has -inf and finite values | -inf positions → 0, finite positions normalize normally |
| Input NaN in slice | entire slice = NaN |

### 6.2 How our kernels handle this

The numerically stable form `exp(x_i - max) / sum(exp(x_j - max))` naturally produces these results through IEEE 754:

- **+inf in slice**: max = +inf. For finite x_i: exp(x_i - inf) = exp(-inf) = 0. For +inf: exp(inf - inf) = exp(NaN) = NaN. Numerator has NaN, sum has NaN, output = NaN. Correct.
- **All -inf**: max = -inf. exp(-inf - (-inf)) = exp(NaN) = NaN. Sum = NaN. Output = NaN. Correct.
- **-inf with finite**: max = finite. exp(-inf - finite) = exp(-inf) = 0. Finite positions: exp(finite - max) normal. Sum excludes -inf contributions (they're 0). Correct.
- **NaN in slice**: max = NaN (NaN propagates through reduce_max). All exp(x - NaN) = NaN. Output = NaN. Correct.

These are inherent to IEEE arithmetic and the `asctile.softmax` built-in. No special-casing needed.

### 6.3 Extreme value ranges

- Case 5: [-100, 100] f32, axis=8192 → exp(x - max) with max shift ensures no overflow in exp
- Case 17: [-1000, 1000] f32, axis=8193 → exp(1000-1000)=1, exp(-1000-1000)=exp(-2000)→0 in f32. No overflow.
- Case 19: [-65504, 65504] f16, axis=255 → max shift prevents overflow. exp(-65504-65504) in f16 → exp(-131008) → +0. No overflow.
- Case 13: inf/nan values → handled by IEEE propagation above.
- Case 14: NaN values → handled by IEEE propagation above.
- Case 15: all zeros → max=0, exp(0)=1, sum=axis_size, output=1/axis_size. For bf16 axis=1013: 1/1013 ≈ 9.87e-4. Within bf16 precision.

### 6.4 Precision compliance

| dtype | MERE threshold | MARE threshold |
|-------|---------------|----------------|
| f16 | 2^-10 ≈ 9.77e-4 | 2^-7 ≈ 7.81e-3 |
| bf16 | 2^-7 ≈ 7.81e-3 | 2^-4 ≈ 6.25e-2 |
| f32 | 2^-13 ≈ 1.22e-4 | 2^-10 ≈ 9.77e-4 |

**f32 internal compute** for all dtypes:
- f32 input: native f32 softmax → exact.
- f16 input: cast to f32 → softmax in f32 → cast to f16. Precision limited by f16 output rounding (≈ 2^-11 relative). Well within 2^-10 threshold.
- bf16 input: cast to f32 → softmax in f32 → cast to bf16. Precision limited by bf16 output rounding (≈ 2^-8 relative). Within 2^-7 threshold.

## 7. UB budget analysis

UB capacity: **253952 bytes**. Measured calibration: real usage ≈ 1.6x naive estimate for sigmoid chain. For softmax: `asctile.softmax` built-in manages its own internal allocation, but the input/output tiles plus any dtype-cast intermediates are our responsibility.

### 7.1 Full-row kernel UB

Per iteration (tile_rows=1, unroll=1):
- Input tile (f16): 1 × axis_padded × 2
- Softmax internal f32 (estimated ~4 tiles): 4 × axis_padded × 4
- Output tile (f16): 1 × axis_padded × 2
- Total (f16, axis=8192): 8192×2 + 4×8192×4 + 8192×2 = 16384 + 131072 + 16384 = 163840 → OK

With unroll=2: double → 327680 → OVERFLOW for axis=8192. Use unroll=1 for axis>4096.

For bf16 full-row:
- bf16 input: axis_padded × 2
- f32 cast: axis_padded × 4
- Softmax f32 (4 tiles): 4 × axis_padded × 4
- f32 result: axis_padded × 4
- bf16 output: axis_padded × 2
- Total (bf16, axis=4096): 4096×(2+4+16+4+2) = 4096×28 = 114688 → OK (unroll=1)
- With unroll=2: 229376 → tight but under 253952

### 7.2 Local-xpose kernel UB

Per iteration (tile_rows=1 effectively since we process one [axis,inner_tile] block, unroll=1):
- Input tile (f16, axis=512, inner=16): 512 × 16 × 2 = 16384
- Transposed tile (same buffer): 16 × 512 × 2 = 16384
- Softmax internals (~4 tiles in f16 or f32): 4 × 16 × 512 × 2or4
- Total (f16, case 16): 16384 + 4×16×512×2 = 16384 + 65536 = 81920 → OK

For bf16 (cast to f32, case 6, axis=127, inner=32):
- bf16 input: 127 × 32 × 2 = 8128
- f32 cast: 127 × 32 × 4 = 16256
- Transposed f32: 32 × 127 × 4 = 16256
- Softmax f32 (4 tiles): 4 × 32 × 127 × 4 = 65024
- f32 result: 16256
- bf16 output: 8128
- Total: ~130048 → OK (unroll=1)

For case 20 (f32, axis=1024, inner=8):
- f32 input: 1024 × 8 × 4 = 32768
- Transposed: 8 × 1024 × 4 = 32768
- Softmax f32 (4 tiles): 4 × 8 × 1024 × 4 = 131072
- Total: 196608 → tight but under 253952 with unroll=1

### 7.3 Transpose kernel UB

Simple block transpose: input tile + output tile = 2 × tile_r × tile_c × dtype_size.
For f16, [128, 128]: 2 × 128 × 128 × 2 = 65536 → very safe.

## 8. Work distribution

### 8.1 Full-row path

Total rows = `outer`. Distribute across `cores = min(72, num_row_tiles)` with grid-stride.

For case 13 (outer=1000003, axis=2): num_row_tiles with tile_rows=256 → 3907 tiles, cores=72, ~54 iterations/core.
For case 17 (outer=255, axis=8193): num_row_tiles with tile_rows=2 → 128 tiles, cores=72, ~2 iterations/core.

### 8.2 Local-xpose path

Total work items = `outer * num_inner_tiles`. Distribute across `cores = min(72, num_work)`.

For case 20 (outer=102, axis=1024, inner=101): inner_tile=8, num_inner_tiles=ceil(101/8)=13, total=102×13=1326, cores=72, ~18 iterations/core.
For case 16 (outer=1, axis=512, inner=2049): inner_tile=16, num_inner_tiles=ceil(2049/16)=129, total=129, cores=72, ~2 iterations/core.

### 8.3 Global-transpose path (case 4)

Transpose kernel: num_tiles = num_r_tiles × num_c_tiles. With [128,128] tiles on [8192,8192]: 64×64=4096 tiles, cores=72, ~57 iterations/core.

## 9. Anti-cheat compliance

- ALL numerical work in `@asctile.jit` kernels on the NPU.
- torch used ONLY for: `torch.empty_like(x)` (allocation), `.shape`, `.dim()`, `.numel()`, `.dtype`, `.is_contiguous()`, `.contiguous()`, `.view()`, `.reshape()`.
- NO `torch.nn.functional.softmax`, NO `torch.softmax`, NO tensor arithmetic (`a + b`), NO `.to(dtype)` on device tensors, NO `torch.cat`, NO `torch.sum`.
- Output is a fresh contiguous tensor allocated with `torch.empty_like`, not a view of input.
- No caching, no pointer reuse between calls.

## 10. Module structure

```python
import torch
import asc
import asctile
from ._pyasc_runtime import ensure_npu_platform

_MAX_CORES = 72

@asctile.jit
def _softmax_full_row(...): ...

@asctile.jit
def _softmax_full_row_bf16(...): ...

@asctile.jit
def _softmax_xpose_local(...): ...

@asctile.jit
def _softmax_xpose_local_bf16(...): ...

@asctile.jit
def _transpose_2d(...): ...

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    ensure_npu_platform()
    ...host dispatch...
    return out
```

## 11. ConstExpr parameters

All tile sizes (`tile_rows`, `tile_cols`, `inner_tile`, `inner_padded`, `unroll_factor`) are passed as `asc.ConstExpr[int]` because they determine copy_in shapes and are resolved at compile time.

Runtime parameters (`total_rows`, `num_cols`, `num_row_tiles`, `num_work`, `outer_size`, `axis_size`, `inner_size`) are passed as plain `int`.

## 12. Tail handling

- **Full-row row tails**: `real_rows = tile_rows if row_off + tile_rows <= total_rows else total_rows - row_off`. Applied to both copy_in and copy_out.
- **Full-row column**: axis_size is always the exact softmax dimension. `tile_cols` is axis_size padded to alignment. `real_shape` columns = axis_size (exact).
- **Local-xpose inner tails**: `real_inner = inner_tile if col_off + inner_tile <= inner_size else inner_size - col_off`. Applied to both copy_in and copy_out.
- **Local-xpose axis**: axis_size is always loaded fully (no axis tiling in local-xpose path). `real_shape` rows = axis_size (exact).
- **Transpose 2D tails**: both row and column tails handled with `real_shape`.

## 13. Risk register

| risk | mitigation |
|------|-----------|
| `asctile.softmax` rejects bf16 | Explicit bf16→f32 cast kernel path |
| UB overflow for large axis in f32 | Conservative tile_rows=1, unroll=1 for axis>4096 |
| `num_inner_tiles` computed at runtime cannot use `//` in kernel | Precompute on host, pass as runtime int, or use `asctile.ceildiv` |
| Case 4 global transpose memory | Two temp buffers of 128MB each; NPU should handle |
| `tile.transpose()` after `copy_in` may have alignment issues | Ensure both dims aligned; test with exact-v2 compile gate |
| `asctile.softmax` on transposed tile shape may not match expected layout | Verify with camodel smoke |
| f16 extreme values (case 19: [-65504, 65504]) with axis=255 | Max shift prevents overflow in all intermediate exp; f16 min normal ≈ 6.1e-5, output resolution OK |
| `real_shape` padded lanes computed but not stored | pad_value=-inf ensures no exceptions in padded lanes |

## 14. Verification plan

1. `python3 -m py_compile candidate.py` — syntax check.
2. Exact-v2 compile gate for all 20 cases — UB and codegen validation.
3. Camodel smoke on representative cases for each path:
   - Case 1 (f16 full-row, small range)
   - Case 5 (f32 full-row, large range)
   - Case 3 (bf16 full-row)
   - Case 4 (f16 global-xpose)
   - Case 20 (f32 local-xpose, large axis)
   - Case 13 (inf/nan special values)
   - Case 14 (NaN special values)
   - Case 15 (all zeros)
4. Full camodel on all 20 cases before promotion.
5. CANNBench on hardware (requires submission credit).

## 15. Summary of kernel launch matrix

| path | kernels launched | cases |
|------|-----------------|-------|
| full-row (f16/f32) | `_softmax_full_row[cores](...)` | 1,2,5,7,8,10,17 |
| full-row (bf16) | `_softmax_full_row_bf16[cores](...)` | 3,12,15,18 |
| local-xpose (f16/f32) | `_softmax_xpose_local[cores](...)` | 11,16,19,20 |
| local-xpose (bf16) | `_softmax_xpose_local_bf16[cores](...)` | 6,9 |
| global-xpose (f16) | `_transpose_2d[cores]` + `_softmax_full_row[cores]` + `_transpose_2d[cores]` | 4 |
