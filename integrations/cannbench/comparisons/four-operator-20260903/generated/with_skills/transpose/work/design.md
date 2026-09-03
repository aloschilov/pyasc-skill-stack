# Transpose Operator — Design Document

## 1. Operator Summary

CANNBench transpose: `transpose(Tensor x, int[] perm) -> Tensor y`.
Formula: `y[i0,...,i_{n-1}] = x[i_{perm[0]},...,i_{perm[n-1]}]`.
Single input, single output. Output shape = input shape reordered by perm.
Output dtype = input dtype. Rank 2–8 (cases span 2–5).

## 2. Runtime Pin

pyasc v2 commit `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`.
Package name: `asctile` (not `asc2`). All kernel decorators and API calls
use `@asctile.jit`, `asctile.range`, `asctile.copy_in`, `asctile.copy_out`,
`asctile.global_tensor`, `asctile.ceildiv`, `asctile.ConstExpr`.

## 3. Case Inventory

| case | raw shape | dtype | perm | elements | value range |
|------|-----------|-------|------|----------|-------------|
| 1 | [64,32,512,128] | f16 | [0,2,1,3] | 128M | [-1,1] |
| 2 | [2048,2048] | f32 | [1,0] | 4.2M | [-2,2] |
| 3 | [4096,4096] | bf16 | [1,0] | 16.8M | [-3,3] |
| 4 | [8192,8192] | int32 | [1,0] | 67.1M | [-10k,10k] |
| 5 | [4096,8192] | int64 | [1,0] | 33.6M | [-100k,100k] |
| 6 | [2,9,256,256] | int16 | [0,2,3,1] | 1.2M | [-1k,1k] |
| 7 | [1023,1023] | f16 | [1,0] | 1.0M | [-0.1,0.1] |
| 8 | [1009,1021] | f32 | [1,0] | 1.0M | [-1,2] |
| 9 | [1537,769] | bf16 | [1,0] | 1.2M | [-5,10] |
| 10 | [363,367,373] | int32 | [2,0,1] | 50M | [-50,100] |
| 11 | [2049,513] | f16 | [1,0] | 1.1M | [-65504,65504] |
| 12 | [3,7,13,4001] | f32 | [0,3,1,2] | 1.1M | [-88,88] |
| 13 | [2,7,256,256] | bf16 | [0,1,3,2] | 940K | [-0.01,0.01] |
| 14 | [2,511,7,127] | f32 | [0,2,1,3] | 911K | [None,None] |
| 15 | [11,13,17,67,67] | f16 | [4,3,2,1,0] | 10.7M | [None,None] |
| 16 | [3,7,11,13,1013] | int64 | [4,3,2,1,0] | 33.4M | [0,0] |
| 17 | [512,2049] | f32 | [1,0] | 1.0M | [-0.5,0.5] |
| 18 | [255,8193] | bf16 | [1,0] | 2.1M | [-1,3] |
| 19 | [4097,511] | int8 | [1,0] | 2.1M | [-128,127] |
| 20 | [2,511,2049] | f16 | [2,1,0] | 2.1M | [-3,6] |

## 4. Shape Simplification

Before selecting a tiling strategy, simplify the shape+perm pair on the
host (pure Python, no JIT). This reduces dispatch branches and improves
tiling efficiency.

### 4.1 Remove size-1 dimensions

If `input_shape[i] == 1`, remove it. Adjust perm indices accordingly:
every perm value > i is decremented, and perm positions whose value
was i are removed.

Case application: Case 16 has `shape[0]=3` not 1, so nothing is removed
there. No case in the 20 has a size-1 dim. This step is still implemented
for correctness but does not trigger for any evaluation case.

### 4.2 Merge adjacent dimensions preserved by perm

If `perm[i-1] + 1 == perm[i]` (the two input dimensions stay adjacent
under perm), merge them: new dimension size = `shape[perm[i-1]] * shape[perm[i]]`.
Re-index perm after merging.

**Case-by-case simplification:**

| case | raw perm | merged dims | simplified shape | simplified perm | reduced rank |
|------|----------|-------------|------------------|-----------------|-------------|
| 1 | [0,2,1,3] | 1↔3 adj→yes | [64, 512, 32, 128] after merging inner: actually dims 0 is alone, and perm[2]=1,p[3]=3 not adj | — | see below |
| 2 | [1,0] | none | [2048,2048] | [1,0] | 2 |
| 3 | [1,0] | none | [4096,4096] | [1,0] | 2 |
| 4 | [1,0] | none | [8192,8192] | [1,0] | 2 |
| 5 | [1,0] | none | [4096,8192] | [1,0] | 2 |
| 6 | [0,2,3,1] | 2↔3 adj→yes | [2,2304,9] | [0,2,1] | 3 |
| 7 | [1,0] | none | [1023,1023] | [1,0] | 2 |
| 8 | [1,0] | none | [1009,1021] | [1,0] | 2 |
| 9 | [1,0] | none | [1537,769] | [1,0] | 2 |
| 10 | [2,0,1] | 0↔1 adj→yes | [363,136_891] | [1,0] | 2 |
| 11 | [1,0] | none | [2049,513] | [1,0] | 2 |
| 12 | [0,3,1,2] | 1↔2 adj→yes | [3,4001,91] | [0,2,1] | 3 |
| 13 | [0,1,3,2] | 0↔1 adj→yes | [14,256,256] | [0,2,1] | 3 |
| 14 | [0,2,1,3] | none | [2,511,7,127] | [0,2,1,3] | 4 |
| 15 | [4,3,2,1,0] | none | [11,13,17,67,67] | [4,3,2,1,0] | 5 |
| 16 | [4,3,2,1,0] | none | [3,7,11,13,1013] | [4,3,2,1,0] | 5 |
| 17 | [1,0] | none | [512,2049] | [1,0] | 2 |
| 18 | [1,0] | none | [255,8193] | [1,0] | 2 |
| 19 | [1,0] | none | [4097,511] | [1,0] | 2 |
| 20 | [2,1,0] | none | [2,511,2049] | [2,1,0] | 3 |

**Re-derivation of case 1**: `[64,32,512,128]`, `perm=[0,2,1,3]`.
Check consecutive pairs in perm: (0,2)→not adj, (2,1)→not adj, (1,3)→not adj.
No merge possible. Shape stays `[64,32,512,128]`, perm `[0,2,1,3]`, rank 4.

**Re-derivation of case 14**: `[2,511,7,127]`, `perm=[0,2,1,3]`.
Pairs: (0,2)→no, (2,1)→no, (1,3)→no. Shape stays, perm stays, rank 4.

### 4.3 Identity detection

If simplified perm == `[0,1,...,n-1]`, the tensor is not permuted.
Still copy (contract: do not return input alias). Use the simple 1-D
copy kernel.

No case produces identity after simplification in this campaign.

## 5. Tiling Strategies

Four kernel families, selected by the simplified shape and perm.

### 5.1 K0: simple_copy — identity or 1-D

Grid-stride 1-D copy over the flat element count. Used when perm is
identity. Not triggered by any case in this campaign but included for
correctness.

```
@asctile.jit
def simple_copy(in_ptr, out_ptr, n, tile_elems, UB_TILE):
    gm_in  = asctile.global_tensor(in_ptr,  [n])
    gm_out = asctile.global_tensor(out_ptr, [n])
    total = asctile.ceildiv(n, UB_TILE)
    for i in asctile.range(asctile.block_idx(), total, asctile.block_num(), unroll_factor=2):
        off = i * UB_TILE
        nelem = UB_TILE if off + UB_TILE <= n else n - off
        data = asctile.copy_in(gm_in, [off], [UB_TILE], real_shape=[nelem])
        asctile.copy_out(data, gm_out, [off], real_shape=[nelem])
```

### 5.2 K1: transpose_line / transpose_column — 2-D transpose

Applies when simplified perm == `[1,0]` and rank == 2.
Two sub-variants selected by aspect ratio:

**transpose_line**: When `height >= width`. Tiles dim-0 in blocks of
`BLOCK_H`. Each iteration loads `[BLOCK_H, WIDTH]` from input, transposes
locally to `[WIDTH, BLOCK_H]`, stores to output at offset `[0, i*BLOCK_H]`.

```
@asctile.jit
def transpose_line(in_ptr, out_ptr, WIDTH, HEIGHT, BLOCK_H,
                   TILE_W, TILE_H, total_tiles, UB_TILE):
    gm_in  = asctile.global_tensor(in_ptr,  [HEIGHT, WIDTH])
    gm_out = asctile.global_tensor(out_ptr, [WIDTH, HEIGHT])
    for i in asctile.range(asctile.block_idx(), total_tiles, asctile.block_num(), unroll_factor=2):
        off_y = i * BLOCK_H
        load_h = BLOCK_H if BLOCK_H < HEIGHT - off_y else HEIGHT - off_y
        data = asctile.copy_in(gm_in, [off_y, 0], [TILE_H, TILE_W],
                               real_shape=[load_h, WIDTH])
        transposed = data.transpose()
        asctile.copy_out(transposed, gm_out, [0, off_y],
                         real_shape=[WIDTH, load_h])
```

**transpose_column**: When `height < width`. Tiles dim-1 in blocks of
`BLOCK_W`. Loads `[HEIGHT, BLOCK_W]` columns, transposes, stores.

```
@asctile.jit
def transpose_column(in_ptr, out_ptr, WIDTH, HEIGHT, BLOCK_W,
                     TILE_W, TILE_H, total_tiles, UB_TILE):
    gm_in  = asctile.global_tensor(in_ptr,  [HEIGHT, WIDTH])
    gm_out = asctile.global_tensor(out_ptr, [WIDTH, HEIGHT])
    for i in asctile.range(asctile.block_idx(), total_tiles, asctile.block_num(), unroll_factor=2):
        off_x = i * BLOCK_W
        load_w = BLOCK_W if BLOCK_W < WIDTH - off_x else WIDTH - off_x
        data = asctile.copy_in(gm_in, [0, off_x], [TILE_H, TILE_W],
                               real_shape=[HEIGHT, load_w])
        transposed = data.transpose()
        asctile.copy_out(transposed, gm_out, [off_x, 0],
                         real_shape=[load_w, HEIGHT])
```

**Cases using K1:** 2, 3, 4, 5, 7, 8, 9, 11, 17, 18, 19.

| case | simplified | sub-variant | BLOCK | TILE (UB padded) |
|------|-----------|-------------|-------|------------------|
| 2 | [2048,2048] f32 | line (h>=w: yes) | BLOCK_H= ceildiv(2048,72)=29 | TILE_H=32, TILE_W=2048 |
| 3 | [4096,4096] bf16 | line | BLOCK_H=57 | TILE_H=64, TILE_W=4096 |
| 4 | [8192,8192] i32 | line | BLOCK_H=114 | TILE_H=128, TILE_W=8192 |
| 5 | [4096,8192] i64 | line | BLOCK_H=57 | TILE_H=64, TILE_W=8192 |
| 7 | [1023,1023] f16 | line | BLOCK_H=15 | TILE_H=16, TILE_W=1024 |
| 8 | [1009,1021] f32 | line | BLOCK_H=15 | TILE_H=16, TILE_W=1024 |
| 9 | [1537,769] bf16 | line (1537>=769) | BLOCK_H=22 | TILE_H=32, TILE_W=784→aligned=800→items_in_block=16, ceildiv(769,16)*16=784 |
| 11 | [2049,513] f16 | line (2049>=513) | BLOCK_H=29 | TILE_H=32, TILE_W=528→aligned=528 |
| 17 | [512,2049] f32 | column (h<w) | BLOCK_W=29 | TILE_W=32, TILE_H=512 |
| 18 | [255,8193] bf16 | column (h<<w) | BLOCK_W=115 | TILE_W=128, TILE_H=256 |
| 19 | [4097,511] i8 | line (4097>511) | BLOCK_H=57 | TILE_H=64, TILE_W=544→aligned=544 |

Alignment computation: `items_in_block = 32 / element_size`.
The tile dimension matching the DMA axis must be padded:
- Input last dim: `TILE_W = ceildiv(width, items_in_block) * items_in_block`
- Output last dim: `TILE_H_out = ceildiv(height, items_in_block) * items_in_block`
Both are padded. `real_shape` uses actual dims.

### 5.3 K2: transpose_one_axis — single-axis tiling

Applies when exactly one input dimension is permuted (moved to a different
position in the output) and all other dimensions remain in their original
positions (after simplification collapses adjacent dims).

Identifies the permuted axis. Loads a `FULL_INPUT_SHAPE` slice with one
dimension tiled to `axis_step`, applies `tile.transpose(*permute)`,
writes the transposed block to the output.

```
@asctile.jit
def transpose_one_axis(
    in_ptr, out_ptr,
    input_shape,      # simplified full input shape
    axis_step,        # tile count along permuted axis
    load_shape_axis,  # dimension index in INPUT where axis sits
    store_shape_axis, # dimension index in OUTPUT where tiled dim goes
    ub_load_shape,    # full input shape with axis dim = axis_step,
                      # last dim padded to items_in_block
    load_shape,       # read real_shape (unpadded)
    permute,          # simplified perm
    block_count,      # ceildiv(full_dim / axis_step)
    unroll_factor
):
    output_shape = [input_shape[permute[d]] for d in range(rank)]
    ub_store_shape = [ub_load_shape[permute[d]] for d in range(rank)]
    store_shape = [load_shape[permute[d]] for d in range(rank)]
    gm_in  = asctile.global_tensor(in_ptr,  input_shape)
    gm_out = asctile.global_tensor(out_ptr, output_shape)
    for i in asctile.range(asctile.block_idx(), block_count,
                           asctile.block_num(),
                           unroll_factor=unroll_factor):
        offset = i * axis_step
        read_offsets = [0]*rank
        read_offsets[load_shape_axis] = offset
        tile = asctile.copy_in(gm_in, read_offsets, ub_load_shape,
                               real_shape=load_shape)
        transposed = tile.transpose(*permute)
        write_offsets = [0]*rank
        write_offsets[store_shape_axis] = offset
        asctile.copy_out(transposed, gm_out, write_offsets,
                         real_shape=store_shape)
```

The host computes `axis_step` so that the UB tile fits:
`tile_elements = product of all dims in ub_load_shape`
`tile_bytes = tile_elements * element_size`
`tile_bytes * unroll_factor_budget <= UB_LIMIT (253952)`

The permuted axis is the one whose position differs in perm.
Host computes `axis_step = UB_LIMIT // (total_elements // dim_size * element_size * unroll_factor * 2)`.

**Cases using K2:** 10, 12, 13, 14, 15, 16, 20.

| case | simplified shape | simplified perm | axis moved | axis_step | tile_bytes (approx) |
|------|-----------------|-----------------|------------|-----------|---------------------|
| 10 | [363,136891] | [1,0] | dim0(363) | 7 | 7*136892*4 ≈ 3.8MB |

Wait — for case 10 the tile is too large for a single-axis full load.
The dimension `136891 * 4 = 547564` bytes already exceeds UB.
This means K2 as a single-axis full-load does NOT work for case 10 in
simplified 2-D form.

**Correction:** When simplified rank <= 2 and the tile along the non-
tiled dimension fits in UB, use K1 (2-D transpose). Case 10 simplifies
to 2-D [363, 136891] with perm [1,0]. Use K1 (transpose_column since
width > height).

Revised case 10: K1 (transpose_column), BLOCK_W = ceildiv(136891,72) ≈ 1902.
TILE_H = 368 (aligned from 363), TILE_W = 1920 (aligned from 1902).
Tile bytes = 368*1920*4 = 2.83 MB. STILL too large.

**Resolution for case 10:** The tile must fit in UB. With
`height=363` and `width=136891`, we need BLOCK_W such that
`363 * BLOCK_W * 4 <= 253952` → `BLOCK_W <= 174`. Use BLOCK_W=174 (aligned 176).
TILE_H=368, TILE_W=176. Tile bytes ≈ 259K. Fits with unroll_factor=1.
Total tiles = ceildiv(136891,174) = 787. With 72 cores, ~11 iterations per core.

| case | simplified shape | simplified perm | strategy | notes |
|------|-----------------|-----------------|----------|-------|
| 10 | [363,136891] | [1,0] | K1 column | BLOCK_W=174, unroll=1 |
| 12 | [3,4001,91] | [0,2,1] | K2 one_axis | axis=dim2(91)→store_axis=1, step=91 fits |
| 13 | [14,256,256] | [0,2,1] | K2 one_axis | axis=dim2(256)→store_axis=2, step=256 |
| 14 | [2,511,7,127] | [0,2,1,3] | K2 one_axis or K4 | axis=dim1(511)→store_axis=2 |
| 15 | [11,13,17,67,67] | [4,3,2,1,0] | K2 one_axis | longest axis tiled |
| 16 | [3,7,11,13,1013] | [4,3,2,1,0] | K2 one_axis | longest axis tiled |
| 20 | [2,511,2049] | [2,1,0] | K2 one_axis | axis=dim0(2)→step=2 |

#### 5.3.1 tile_bytes for K2 cases

- Case 12: input=[3,4001,91], perm=[0,2,1]. Tiling dim2: ub_load=[3,4001,96],
  tile_bytes = 3*4001*96*4 ≈ 4.6MB. Way too large. **Must use K3 (2-axis) or
  tile along the large dimension.**

**Revised strategy for large inner dims:** When ANY non-tiled dimension
exceeds the UB budget for a full load, the single-axis approach must tile
that dimension too. Fall back to K3 (two-axis) or further reduce to
row-copy patterns.

### 5.4 K3: transpose_2_axis — two-axis tiling (general fallback)

When one-axis tiling overflows UB, tile TWO dimensions simultaneously.
The kernel loads a block with two dimensions sliced and the rest at full
extent, applies `tile.transpose(*permute)`, and writes.

```
@asctile.jit
def transpose_2_axis(
    in_ptr, out_ptr,
    input_shape, axis_step_pair,
    store_axis_pair,   # which output dims are tiled
    ub_load_shape, load_shape,
    permute, block_count, block_width,
    unroll_factor
):
    output_shape = [input_shape[permute[d]] for d in range(rank)]
    ub_store_shape = [ub_load_shape[permute[d]] for d in range(rank)]
    store_shape = [load_shape[permute[d]] for d in range(rank)]
    load_axis0 = permute[store_axis_pair[0]]
    load_axis1 = permute[store_axis_pair[1]]
    gm_in  = asctile.global_tensor(in_ptr,  input_shape)
    gm_out = asctile.global_tensor(out_ptr, output_shape)
    for i in asctile.range(asctile.block_idx(), block_count,
                           asctile.block_num(),
                           unroll_factor=unroll_factor):
        off0 = (i % block_width) * axis_step_pair[0]
        off1 = (i // block_width) * axis_step_pair[1]
        cnt0 = axis_step_pair[0] if off0 + axis_step_pair[0] < input_shape[load_axis0] else input_shape[load_axis0] - off0
        cnt1 = axis_step_pair[1] if off1 + axis_step_pair[1] < input_shape[load_axis1] else input_shape[load_axis1] - off1
        read_offsets = [0]*rank
        read_offsets[load_axis0] = off0
        read_offsets[load_axis1] = off1
        read_shape = list(load_shape)
        read_shape[load_axis0] = cnt0
        read_shape[load_axis1] = cnt1
        tile = asctile.copy_in(gm_in, read_offsets, ub_load_shape,
                               real_shape=read_shape)
        transposed = tile.transpose(*permute)
        write_offsets = [0]*rank
        write_offsets[store_axis_pair[0]] = off0
        write_offsets[store_axis_pair[1]] = off1
        write_shape = list(store_shape)
        write_shape[store_axis_pair[0]] = cnt0
        write_shape[store_axis_pair[1]] = cnt1
        asctile.copy_out(transposed, gm_out, write_offsets,
                         real_shape=write_shape)
```

### 5.5 K4: transpose_nlast_axis — permute only non-last dims

When `perm[-1] == ndim-1` (last output dim = last input dim), the last
dimension is unchanged. Load rows of the collapsed trailing dims,
permute only the leading dims in-UB, and store rows.

```
@asctile.jit
def transpose_nlast_axis(
    in_ptr, out_ptr,
    axis_step, repeats,
    permute, gm_read_shape, gm_write_shape,
    ub_shape, read_shape, unroll_factor
):
    gm_in  = asctile.global_tensor(in_ptr,  gm_read_shape)
    gm_out = asctile.global_tensor(out_ptr, gm_write_shape)
    for i in asctile.range(asctile.block_idx(), repeats,
                           asctile.block_num(),
                           unroll_factor=unroll_factor):
        store_offsets = [0]*len(permute)
        store_offsets[permute[0]] = i * axis_step
        tile = asctile.copy_in(gm_in, [i*axis_step, 0], read_shape,
                               real_shape=read_shape)
        reshaped = tile.reshape(*ub_shape)
        transposed = reshaped.transpose(*permute)
        asctile.copy_out(transposed, gm_out, store_offsets)
```

**Cases using K4:** 1, 6 (after simplification).

- Case 1: raw `[64,32,512,128]`, perm `[0,2,1,3]`. No merge.
  perm[-1]=3=ndim-1, so last dim (128) is unchanged.
  Collapse dims 1..3 → inner_dim = 32*512*128 = 2M.
  UB: load `[axis_step, 2M]` → tile_bytes = axis_step * 2M * 2 (f16).
  Even axis_step=1 → 4MB. Too large.
  **Must collapse to `[64, 32*512, 128]` → treat dims 2,3 as trailing rows.**
  With `gm_read_shape = [64, 32*512*128]` → still 2M elements.

  **Revised for case 1:** Decompose differently. The permutation [0,2,1,3]
  only swaps dims 1 and 2. Dims 0 and 3 stay. Use K2 (one_axis) on the
  simplified form `[64,32,512,128]` → identify the axis that moves (dim 1
  with size 32 goes to output dim 2). Load a slice with dim1 tiled:
  tile = `[64, 1, 512, 128]`.
  tile_bytes = 64*1*512*128*2 = 8MB. Way too large.

  **Must use K3 (two-axis)**: tile the two swapping dims.
  The effective problem is transposing a `[32, 512]` block within the
  `[64, _, _, 128]` frame. Tile `[32, step_512]` of dim1×dim2.
  Load `[64, block_d1, block_d2, 128]` → `64*block_d1*block_d2*128*2` bytes.
  To fit 128KB: `64*block_d1*block_d2*128*2 <= 128000` →
  `block_d1*block_d2 <= 0.78`. Must tile to 1×1, still ~16KB for the
  frame. Actually:
  `64*1*1*128*2 = 16384` bytes. That fits! With block_d1=1 and block_d2=1,
  each tile copies one "row" of `[64,1,1,128]` elements (16KB for the
  outer frame). But this would require 32*512 = 16384 tiles — very many.

  Better: tile dim2 (size 512) in blocks of STEP_D2, and collapse
  dims 0×dim1 into the outer iteration.
  Input = `[64, 32, 512, 128]`. Treat as `[2048, 512, 128]`
  (merge dims 0+1). Permute swaps dim1↔dim2: output `[2048, 512, 128]`.
  Wait, the perm is [0,2,1,3] on the original.
  Output `y[d0,d1,d2,d3] = x[d0,d2,d1,d3]`.
  Merge d0→outer_d0: `y[outer,d1,d2,d3] = x[outer,d2,d1,d3]` where
  `outer = d0*32 + d1` (input) and output `outer = d0*512 + d2`.
  This is NOT a simple merge since dim1 and dim2 swap.

  **Final case 1 approach:** Use K3 with two-axis tiling on (dim1, dim2).
  Collapse dims 0 and 3 as outer/inner.
  Input viewed as `[outer(64), dim1(32), dim2(512), inner(128)]`.
  Output: `[outer(64), dim2(512), dim1(32), inner(128)]`.
  Tile `(dim1, dim2)` with steps `(S1, S2)`:
  Load `[outer, S1, S2, inner]` from input → tile `[64, S1, S2, 128]`.
  But we still carry the full outer dimension. Instead, iterate over
  `outer` on the host.

  **Host loop approach** (acceptable: the outer dim is only 64):

  For each `d0` in `range(64)`:
    Launch a 2-D transpose kernel over `[32, 512]` with inner=128.
    Input slice: `x[d0, :, :, :]` → shape `[32, 512, 128]`, contiguous,
    offset = `d0 * 32 * 512 * 128`.
    Output slice: `y[d0, :, :, :]` → shape `[512, 32, 128]`, contiguous,
    offset = `d0 * 512 * 32 * 128`.

  Inner kernel (2-D transpose_line on `[32, 512]` with trailing inner=128):
  Tile dim0(dim1=32) in blocks of BLOCK_H.
  Load `[BLOCK_H, 512, 128]`, permute to `[512, BLOCK_H, 128]`, store.

  UB: `[step, 512, 128]` × 2 = step * 512 * 128 * 2 * 2 bytes.
  step * 131072 * 2 <= 253952 → step <= 0.97. Step = 1.
  Per tile: 1*512*128*2 = 131072 bytes (for each of load and store).
  Total = ~262KB. Exceeds UB with unroll_factor=2.
  With unroll_factor=1: 131KB for load + 131KB for store = 262KB. Still exceeds 254KB.

  **Must merge inner dim into load width.**
  View input slice `[32, 512, 128]` as `[32, 512*128]` = `[32, 65536]`.
  2-D transpose [32, 65536] → [65536, 32].
  Tile line: BLOCK_H rows. Load `[BLOCK_H, 65536]`.
  tile_bytes = BLOCK_H * 65536 * 2. For BLOCK_H=1: 131KB. Fits in 254KB?
  131072 + some overhead for transposed [65536, 1]*2 = 131KB total. Yes, barely.
  With unroll_factor=1: ~131KB fits.

  **Case 1 final routing:** host iterates d0∈[0,64). Inner: K1 transpose_line
  on [32, 65536] f16, BLOCK_H=1, TILE_H=16 (aligned), TILE_W=65536.
  32 rows per slice, 32 tiles per slice, 64 slices → 2048 total tiles, 72 cores.

  **OR: use a single K2 call with d0 collapsed into iteration:**
  Reshape input to `[64*32, 512, 128]` = `[2048, 512, 128]`.
  But the permute [0,2,1,3] doesn't simplify to a 3-D perm. We need to
  iterate d0 externally.

  **Chosen approach:** 64 separate K1 launches, each transposing [32, 512]
  with inner=128 elements per row.

- Case 6: `[2,9,256,256]`, perm `[0,2,3,1]`.
  After merge dims 2+3: `[2,9,65536]`, perm `[0,2,1]`.
  After merge dims 0 alone: `[2,9,65536]`.
  3-D one-axis: perm moves dim2→dim1 position (output dim 1 is input dim 2).
  K2: tile dim1(9). Load `[2,1,65536]`. Tile_bytes = 2*1*65536*2 = 262KB.
  Exceeds 254KB! With step=1: 131KB. Fits with unroll=1.
  But wait: the output dim 2 of the simplified perm maps to the merged
  dim (65536), not dim2=256. Let me re-check.

  Case 6 simplified: `[2, 2304, 9]`, perm `[0,2,1]`.
  (dims 1 and 2 merged: 9*256=2304, dim 3=256 collapsed... no.)

  **Re-derivation case 6:** `[2,9,256,256]`, perm `[0,2,3,1]`.
  Check consecutive in perm: (0,2)→no, (2,3)→YES adjacent!
  Merge dims 2 and 3 (in INPUT these are dims 2 and 3): 256*256=65536.
  New shape: `[2, 9, 65536]`. New perm indices: dim2→dim1, dim3→dim2.
  perm becomes: `[0, ? , 1]`. The input dims are now 0(orig 0), 1(orig 1), 2(merged 2+3).
  perm[0]=0, perm[1]=2(maps output dim1 to input dim2 which is the merged),
  perm[2]=1(maps output dim2 to input dim1).
  Simplified perm: `[0, 2, 1]`, simplified shape: `[2, 9, 65536]`.

  K2 one-axis: perm=[0,2,1] moves dim1(9) to output dim2 and dim2(65536)
  to output dim1. Two dims move. Not a simple one-axis.

  Since only one "independent" dim moves (dims 1 and 2 swap positions),
  this is equivalent to a 2-D transpose of the last two dims [9, 65536]
  with perm [1,0], within an outer dim of size 2.

  **Case 6 approach:** Host iterates d0∈[0,2). Inner: K1 transpose on
  [9, 65536] i16, transpose_line (9<65536, so transpose_column variant).
  BLOCK_W = ceildiv(65536, 72) = 911. TILE_H=16, TILE_W=928.
  tile_bytes = 9*928*2 = 16704. Fits easily.
  Total tiles per slice: ceildiv(65536,911) = 72. 2 slices → 144 tiles, 72 cores.

  **OR (simpler for UB):** Transpose [9, 65536] with K1 transpose_line:
  height=9, width=65536. Since height<width, use transpose_column.
  Load [9, BLOCK] blocks. BLOCK=912 (aligned). TILE_H=16, TILE_W=928.
  tile_bytes = 16*928*2 = 29696. Well within 254KB. Good.

## 6. Final Dispatch Table

| case | raw shape | simplified | simplified perm | strategy | kernel |
|------|-----------|-----------|-----------------|----------|--------|
| 1 | [64,32,512,128] f16 | [64,32,512,128] | [0,2,1,3] | host-iter d0, K1 on [32,65536] | transpose_line |
| 2 | [2048,2048] f32 | [2048,2048] | [1,0] | K1 | transpose_line |
| 3 | [4096,4096] bf16 | [4096,4096] | [1,0] | K1 | transpose_line |
| 4 | [8192,8192] i32 | [8192,8192] | [1,0] | K1 | transpose_line |
| 5 | [4096,8192] i64 | [4096,8192] | [1,0] | K1 | transpose_line |
| 6 | [2,9,256,256] i16 | [2,9,65536] → [9,65536] | [1,0] | host-iter d0, K1 | transpose_column |
| 7 | [1023,1023] f16 | [1023,1023] | [1,0] | K1 | transpose_line |
| 8 | [1009,1021] f32 | [1009,1021] | [1,0] | K1 | transpose_line |
| 9 | [1537,769] bf16 | [1537,769] | [1,0] | K1 | transpose_line |
| 10 | [363,367,373] i32 | [363,136891] | [1,0] | K1 | transpose_column (BLOCK_W=174, unroll=1) |
| 11 | [2049,513] f16 | [2049,513] | [1,0] | K1 | transpose_line |
| 12 | [3,7,13,4001] f32 | [3,4001,91] → see below | [0,2,1] | host-iter d0, K1 on [4001,91] | transpose_line |
| 13 | [2,7,256,256] bf16 | [14,256,256] | [0,2,1] | host-iter d0, K1 on [256,256] | transpose_line |
| 14 | [2,511,7,127] f32 | [2,511,7,127] | [0,2,1,3] | K4 nlast_axis on [14,511,7,127]... see below |
| 15 | [11,13,17,67,67] f16 | [11,13,17,67,67] | [4,3,2,1,0] | K2 one_axis (tile longest) | transpose_one_axis |
| 16 | [3,7,11,13,1013] i64 | [3,7,11,13,1013] | [4,3,2,1,0] | K2 one_axis | transpose_one_axis |
| 17 | [512,2049] f32 | [512,2049] | [1,0] | K1 | transpose_column |
| 18 | [255,8193] bf16 | [255,8193] | [1,0] | K1 | transpose_column |
| 19 | [4097,511] i8 | [4097,511] | [1,0] | K1 | transpose_line |
| 20 | [2,511,2049] f16 | [2,511,2049] | [2,1,0] | K2 one_axis | transpose_one_axis |

### Case 12 detail
`[3,7,13,4001]` perm `[0,3,1,2]`. Pairs: (0,3)→no, (3,1)→no, (1,2)→YES.
Merge dims 1+2 (INPUT dims 1 and 2): 7*13=91.
Simplified: `[3, 4001, 91]`, perm: `[0, 2, 1]`.
Further check: perm[0]=0 stays, perm[1:3]=[2,1] swaps dims 1↔2.
Host iterates d0∈[0,3). Inner: 2-D transpose [4001, 91] f32, perm [1,0].
4001 > 91, so transpose_line.
BLOCK_H = ceildiv(4001, 72) = 56. TILE_H=64, TILE_W=96 (aligned).
tile_bytes = 64*96*4 = 24576. Well within UB.
Total tiles per slice: ceildiv(4001, 56) = 72. 3 slices → 216 tiles.

### Case 13 detail
`[2,7,256,256]` perm `[0,1,3,2]`. Pairs: (0,1)→YES, (1,3)→no, (3,2)→no.
Merge dims 0+1: 14. Simplified: `[14, 256, 256]`, perm `[0, 2, 1]`.
Host iterates d0∈[0,14). Inner: 2-D transpose [256, 256] bf16, perm [1,0].
h==w, use transpose_line.
BLOCK_H = ceildiv(256, 72) = 4. TILE_H=8, TILE_W=256.
tile_bytes = 8*256*2 = 4096. Easy.
Total tiles per slice: ceildiv(256, 4) = 64. 14 slices → 896 tiles.

### Case 14 detail
`[2,511,7,127]` perm `[0,2,1,3]`. No consecutive pairs.
perm[-1]=3=ndim-1 → last dim unchanged.
K4 nlast_axis approach:
- Collapse inner dims 1,2,3: inner = 511*7*127 = 454239.
- `gm_read_shape = [2, 454239]`, `gm_write_shape = [2, 7, 511, 127]`.
- axis_step: tile dim0(2). axis_step=2. Loads `[2, 454239]` f32 = 3.6MB. TOO LARGE.
- axis_step=1: loads `[1, 454239]` f32 = 1.8MB. STILL TOO LARGE.

**Must decompose case 14 further.** perm=[0,2,1,3] swaps dims 1 and 2.
Treat d0 and d3 as outer/inner:
- Host iterates d0∈[0,2), d3∈[0,127).
- Inner: 2-D transpose of `[511, 7]` f32, perm [1,0].
- `511 > 7`, use transpose_line.
- BLOCK_H = ceildiv(511, 72) = 8. TILE_H=8, TILE_W=8 (aligned).
- tile_bytes = 8*8*4 = 256. Trivial.
- Total: 2*127 * ceildiv(511,8) = 2*127*64 = 16256 tiles.

**OR (more efficient):** Iterate d0∈[0,2). Inner: transpose `[511, 7]` with
inner=127 for each of the 127 trailing elements. But those 127 are not
contiguous as a row. View input slice `x[d0, :, :, :]` as `[511, 7, 127]`.
The permuted output is `y[d0, :, :, :]` = `[511, 127, 7, 127→wrong]`.

Actually `y[d0, d1, d2, d3] = x[d0, d2, d1, d3]`.
Output = `[2, 7, 511, 127]`. For fixed d0,d3:
Output `[d1, d2] = Input[d2, d1]` — yes, pure 2-D transpose of [511, 7].

The inner 127 elements are contiguous for each (d0, d1, d2) triple.
So for each d0 and each d1 in output:
- Output row: `y[d0, d1, :, :]` has shape `[511, 127]`, contiguous.
- Input rows: `x[d0, :, d1, :]` has shape `[511, 127]`, stride: dim1 has
  stride 7*127=889, dim3 has stride 1. NOT contiguous as a single memcpy.

**Chosen approach for case 14:** 2*127 = 254 host iterations, each doing K1
transpose_line on [511, 7] f32. This is the most UB-friendly decomposition.

### Case 15 detail
`[11,13,17,67,67]` f16, perm `[4,3,2,1,0]`. No simplification.
All dims permute. K2 one_axis: pick the longest axis (dim4=67) to tile.
Load full input with dim4 sliced to step=1: `[11,13,17,67,1]`.
tile_bytes = 11*13*17*67*1*2 = 323K. Exceeds 254KB!
step=1, tile_bytes = 161K? Let me recalculate: 11*13=143, *17=2431, *67=162877, *1=162877, *2=325754.
Still 326KB! Exceeds UB even at step=1.

**Must reduce.** Pick axis dim0=11 (smallest):
step=1: `[1,13,17,67,67]` → 1*13*17*67*67*2 = 1988K. Way too large.
Pick axis dim4=67, step=1: `[11,13,17,67,1]` → 326K. Still too large.

**Must use two-axis tiling (K3).** Tile dims 0 and 4:
Load `[S0, 13, 17, 67, S4]`.
S0=1, S4=1: `[1,13,17,67,1]` → 326K. Still too large because of dim3=67.

Tile dims 3 AND 4: Load `[11, 13, 17, S3, S4]`.
S3=1, S4=1: `[11,13,17,1,1]` → 11*13*17*2 = 4862. Tiny! Good.
S3=2, S4=2: `[11,13,17,2,2]` → 19448. Still small.
S3=67, S4=67: full `[11,13,17,67,67]` → 10.7M elements → 21.4MB. Way too large.

Find max S3*S4 fitting UB:
`11*13*17*S3*S4*2 <= 253952` → `S3*S4 <= 28`. So e.g. S3=6, S4=4.
tile_bytes ≈ 11*13*17*6*4*2 = 114K. Fits.
Total tiles: ceildiv(67,6)*ceildiv(67,4) = 12*17 = 204.

**Case 15: K3 two-axis tiling.** axes to tile: dim3 and dim4.
store_axes in output: perm=[4,3,2,1,0]. Output dim0 ← input dim4, output dim3 ← input dim1.
Input dims being tiled: 3 and 4. Output dim for input dim3 = perm_inv[3] = 1.
Output dim for input dim4 = perm_inv[4] = 0.
So `store_axis_pair = [1, 0]` (output dims 0 and 1 correspond to input dims 4 and 3).

### Case 16 detail
`[3,7,11,13,1013]` i64, perm `[4,3,2,1,0]`. No simplification.
Longest axis: dim4=1013. Tile dim4 with step=1:
`[3,7,11,13,1]` → 3*7*11*13*1*8 = 24024. Fits (24KB).
step=2: 48KB. Fits. step=8: 192KB. Fits with unroll=1.
step=15: 24024*15 = 360KB. Exceeds.
Max step: 253952 / 24024 / 2 = 5. step=5: 120KB. Fits with unroll=2 (if 2x=240K≤254K, close). Use unroll=1, step=8.

**Case 16: K2 one_axis.** Tile dim4(1013) with step=8.
tile `[3,7,11,13,8]` → 24024 elements × 8 bytes = 192KB. Fits with unroll=1.
Output tile: perm → `[8,13,11,7,3]`.
Total tiles: ceildiv(1013, 8) = 127.

### Case 20 detail
`[2,511,2049]` f16, perm `[2,1,0]`. No simplification.
perm reverses all 3 dims. Tile longest axis dim2=2049.
step=1: `[2,511,1]` → 2*511*1*2 = 2044. Tiny.
step=100: 204400. Fits.
Max step: 253952 / 2044 / 2 = 62. step=62: 126728. Fits with unroll=2 (if 2x=254K, tight). Use step=62 unroll=1 or step=30 unroll=2.

**Case 20: K2 one_axis.** Tile dim2(2049) with step ≈ 60.
tile `[2,511,60]` → 122640. 122KB fits. Output: `[60,511,2]`.
Total tiles: ceildiv(2049, 60) = 35.

## 7. Alignment Rules

**32-byte DMA alignment** (mandatory for all copy_in/copy_out last dims).

```
items_in_block = 32 // element_size
```

| dtype | element_size | items_in_block |
|-------|-------------|----------------|
| int8 | 1 | 32 |
| f16, bf16, int16 | 2 | 16 |
| f32, int32 | 4 | 8 |
| int64 | 8 | 4 |

**Padded tile dimensions:**
- For any tile dim that is the last physical dimension of a global_tensor
  or local tile, round up: `ceildiv(dim, items_in_block) * items_in_block`.
- `real_shape` uses the actual logical dim.
- Both the input global_tensor and the output global_tensor must have
  their last dimension padded.

**Case 19 special (int8):**
`items_in_block = 32`. Input last dim 511 → pad to 512. Output last dim
4097 → pad to 4128. Both fit in UB for the line variant.
Note: int8 tiles in asctile have NO vector ops, but transpose is a layout
operation (not arithmetic), so `tile.transpose()` and copy_in/copy_out
should work on int8 tiles. If int8 copy_in is rejected at compile,
promote to int16 before transpose and demote before copy_out (per
pyasc-v2-blockers: "Direct arithmetic/cast paths from int8 tiles are
incomplete; the observed conversion route is int8 → f16 → f32").

**Fallback for int8 cast failure:** If `asctile.copy_in` on int8 global_tensor
succeeds but the subsequent `tile.transpose()` or `copy_out` fails, use:
`tile_i16 = asctile.cast(tile, asc.int16)`, transpose, cast back to int8
before store. Verify at exact-v2 compile gate.

## 8. Tail Handling

All kernels use `real_shape` to handle non-divisible dimensions:

```python
load_h = BLOCK_H if off_y + BLOCK_H < HEIGHT else HEIGHT - off_y
data = asctile.copy_in(gm_in, [off_y, 0], [TILE_H, TILE_W],
                       real_shape=[load_h, actual_w])
```

The `real_shape` limits the DMA transfer to valid elements. Padded lanes
(local tile beyond `real_shape`) are NOT copied back by `copy_out`.

**Pad value:** For transpose (pure data movement), padding values do not
participate in any arithmetic. The default pad_value=0 is safe. No
divisors, no transcendental functions, no cancellation risk.

**Tail-specific cases to watch:**

| case | tail dim | tail size | padded size | note |
|------|----------|-----------|-------------|------|
| 7 | 1023 | 1023 | 1024 (f16) | 1 element tail |
| 8 | 1009,1021 | 1009,1021 | 1016,1024 (f32) | prime dims |
| 9 | 1537,769 | 1537,769 | 1537→1552(bf16), 769→784 | prime |
| 11 | 2049,513 | 2049,513 | 2056(f16), 513→528 | prime |
| 13 after merge | 256,256 bf16 | exact | exact |
| 16 | 1013 | 1013 | 1016(i64) | off-by-3 |
| 17 | 2049 | 2049 | 2056(f32) | off-by-7 |
| 18 | 8193 | 8193 | 8208(bf16) | off-by-15 |
| 19 | 4097,511 | 4097,511 | 4128(i8), 512 | prime sizes |

All handled by `real_shape` with aligned pad dims.

## 9. UB Budget

**Hard limit:** 253952 bytes.

**UB estimation per tile:**
- Input tile: `product(ub_load_shape) * element_size * unroll_factor`
- Transposed tile (if distinct allocation): same size
- Compiler temporaries: typically 1.0x for pure data movement (no live
  intermediaries since transpose is an in-place permutation)

**Budget check per case (worst case):**

| case | tile elements | elem size | UB (in+out) × unroll | fits? |
|------|--------------|-----------|----------------------|-------|
| 1 (inner K1) | 1*65536 | 2 | 1*65536*2*2 = 262K | tight, unroll=1 |
| 2 | 32*2048 | 4 | 32*2048*4*2 = 524K → must use smaller block | recalc |
| 3 | 64*4096 | 2 | 64*4096*2*2 = 1.05M → must use smaller block | recalc |
| 4 | 128*8192 | 4 | 128*8192*4*2 = 8.4M → must use smaller block | recalc |

**CORRECTION for 2-D cases:** The BLOCK_H for K1 must be chosen so the tile
fits in UB, NOT simply `ceildiv(height, 72)`.

Recalculation:
```
tile_bytes = TILE_H * TILE_W * element_size
UB for in+out ≈ 2 * tile_bytes * unroll_factor
```

For TILE_W padded to items_in_block:

| case | W | items_in_block | TILE_W | max TILE_H (unroll=2) | max TILE_H (unroll=1) | chosen |
|------|---|----------------|--------|----------------------|----------------------|--------|
| 2 | 2048 | 8 | 2048 | 31 | 62 | TILE_H=32, unroll=1 |
| 3 | 4096 | 16 | 4096 | 31 | 62 | TILE_H=32, unroll=1 |
| 4 | 8192 | 8 | 8192 | 7 | 15 | TILE_H=8, unroll=1 |
| 5 | 8192 | 4 | 8192 | 7 | 15 | TILE_H=8, unroll=1 |
| 7 | 1023 | 16 | 1024 | 62 | 124 | TILE_H=64, unroll=1 |
| 8 | 1021 | 8 | 1024 | 62 | 124 | TILE_H=64, unroll=1 |
| 9 | 769 | 16 | 784 | 81 | 163 | TILE_H=80, unroll=1 |
| 10 (col) | 363→368(h) | 8 | — | — | BLOCK_W=174, unroll=1 | see above |
| 11 | 513 | 16 | 528 | 120 | 240 | TILE_H=128, unroll=1 |
| 17 (col) | 512→512(h) | 8 | — | — | BLOCK_W=31, unroll=1 | |
| 18 (col) | 255→256(h) | 16 | — | — | BLOCK_W=96, unroll=1 | |
| 19 | 511 | 32 | 512 | 123 | 247 | TILE_H=128, unroll=1 |

**Unroll factor selection:**
- All K1 2-D kernels use `unroll_factor=1` due to the large tile size
  (output tile is a separate allocation after `.transpose()`).
- K2 one-axis kernels use `unroll_factor=2` when tiles are small enough
  (tile_bytes * 4 ≤ 253952), otherwise `unroll_factor=1`.
- K3 two-axis: `unroll_factor=1` for case 15.

## 10. Numerical Behavior

**Transpose is exact.** No floating-point arithmetic is performed. Data
is read from input and written to output in a different order. The
relative error for every element is identically 0.0 for all dtypes.

**Precision thresholds from desc.md:**

| dtype | MERE threshold | MARE threshold |
|-------|---------------|----------------|
| f16 | 2^-10 ≈ 9.77e-4 | 9.77e-3 |
| bf16 | 2^-7 ≈ 7.81e-3 | 7.81e-2 |
| f32 | 2^-13 ≈ 1.22e-4 | 1.22e-3 |

Since transpose produces exact copies, MERE=0 and MARE=0 for all cases.
Well within every threshold.

**Integer dtypes:** int8, int16, int32, int64 — exact copy, no precision
concern. The CANNBench harness may still evaluate relative error; integer
transposes produce exactly matching goldens.

**Value range coverage:**

| case | range | concern? |
|------|-------|----------|
| 11 | [-65504, 65504] | f16 max finite. Exact copy, no overflow. |
| 14 | [None, None] | default f32 range. Exact copy. |
| 15 | [None, None] | default f16 range. Exact copy. |
| 16 | [0, 0] | all zeros. Trivially exact. |

**NaN/Inf handling:** If the golden input contains NaN or Inf values,
transpose copies them to the correct output positions. No special casing
needed. The data is not modified, only reordered.

## 11. Anti-Cheat Compliance

### What torch does (allowed):
- `ensure_npu_platform()` — required.
- `x.contiguous()` — ensure contiguous input.
- `torch.empty(output_shape, dtype=x.dtype, device=x.device)` — allocate output.
- `.shape`, `.numel()`, `.dtype`, `.element_size()`, `.is_contiguous()` — metadata.
- `.view()`, `.reshape()` — metadata views (no data copy).

### What torch does NOT do (forbidden):
- `x.permute()` — this is the golden's operation, not ours.
- `x.to(dtype)` — host cast (forbidden).
- Any `torch.cat`, `torch.clone`, arithmetic ops.
- `x.data_ptr()` — never pass raw pointers to JIT.

### Compliance checklist:
- [ ] All data movement in `@asctile.jit` kernels.
- [ ] Host does not call `.permute()` or `.contiguous()` on a permuted view.
- [ ] Output is freshly allocated, not a view of input.
- [ ] Output is contiguous (guaranteed by `torch.empty` + kernel write order).
- [ ] No `data_ptr()` calls.
- [ ] No caching of outputs by `data_ptr`.

## 12. Kernel List

| kernel | type | purpose | used by cases |
|--------|------|---------|--------------|
| `simple_copy` | 1-D | identity copy | (not triggered) |
| `transpose_line` | 2-D | tile rows, transpose | 2,3,4,5,7,8,9,11,19 (and inner for 1,6,12,13,14) |
| `transpose_column` | 2-D | tile cols, transpose | 10,17,18 (and inner for 6) |
| `transpose_one_axis` | N-D | single axis tiling | 15,16,20 |
| `transpose_2_axis` | N-D | two axis tiling | (fallback, not triggered in final table) |
| `transpose_nlast_axis` | N-D | permute leading dims, last dim unchanged | (not triggered directly; case 14 decomposed) |

## 13. Host Dispatch Logic

```python
def transpose(x: torch.Tensor, perm: list) -> torch.Tensor:
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()

    shape = list(x.shape)
    out_shape = [shape[perm[i]] for i in range(len(perm))]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    if x.numel() == 0:
        return out

    simplified_shape, simplified_perm = simplify_shape(shape, perm)
    rank = len(simplified_perm)

    if simplified_perm == list(range(rank)):
        # Identity: simple copy
        launch_simple_copy(x, out, simplified_shape)
    elif rank == 2:
        # 2-D transpose
        launch_2d(x, out, simplified_shape, simplified_perm)
    elif rank == 3 and simplified_perm[-1] == rank - 1:
        # 3-D with last dim unchanged: one-axis on leading dims
        launch_one_axis(...)
    elif rank <= 5:
        # General: pick longest permuted axis or two-axis
        launch_general(x, out, simplified_shape, simplified_perm)

    return out
```

The dispatcher MUST NOT use `torch.permute()` and MUST always launch a
kernel that performs the data movement, even for identity (to satisfy
anti-aliasing: "must still launch the handwritten copy kernel rather than
returning the input alias").

## 14. int8 Special Handling (case 19)

int8 global tensors load via `copy_in` successfully. The question is
whether `tile.transpose()` works on int8 local tiles.

**Primary path:** Try `copy_in` as int8 → `transpose` → `copy_out` as int8.
If this compiles on the exact-v2 gate, use it.

**Fallback path:** If int8 local ops fail at compile:
```
tile_i8 = asctile.copy_in(gm_in, ..., real_shape=...)
tile_wide = asctile.cast(tile_i8, asc.int16)  # widen to i16
tile_t = tile_wide.transpose()
tile_out = asctile.cast(tile_t, asc.int8)  # narrow back
asctile.copy_out(tile_out, gm_out, ...)
```
Cast int8→f16→i16 or int8→i16 direct, depending on what the pinned API
supports. The pyasc-v2-blockers note says the observed path is
int8→f16→f32, but for transpose we only need int8→int16→int8 widening/
narrowing, which may work directly.

**Verify at compile gate.** Both paths should be checked with
`run_local_compile_gate.sh --candidate candidate.py --op transpose`.

## 15. Module Structure

```python
import torch
import asc
import asctile
import math

from ._pyasc_runtime import ensure_npu_platform

_MAX_CORES = 72
_UB_LIMIT = 253952

# --- Kernel definitions ---

@asctile.jit
def simple_copy(in_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                n: int, tile_elems: asc.ConstExpr[int]):
    ...

@asctile.jit
def transpose_line(in_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                   WIDTH: asc.ConstExpr[int], HEIGHT: asc.ConstExpr[int],
                   BLOCK_H: asc.ConstExpr[int],
                   TILE_W: asc.ConstExpr[int], TILE_H: asc.ConstExpr[int],
                   total_tiles: int):
    ...

@asctile.jit
def transpose_column(in_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                     WIDTH: asc.ConstExpr[int], HEIGHT: asc.ConstExpr[int],
                     BLOCK_W: asc.ConstExpr[int],
                     TILE_W: asc.ConstExpr[int], TILE_H: asc.ConstExpr[int],
                     total_tiles: int):
    ...

@asctile.jit
def transpose_one_axis(in_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                       input_shape: asc.ConstExpr,
                       axis_step: asc.ConstExpr,
                       load_shape_axis: asc.ConstExpr,
                       store_shape_axis: asc.ConstExpr,
                       ub_load_shape: asc.ConstExpr,
                       load_shape: asc.ConstExpr,
                       permute: asc.ConstExpr,
                       block_count: int,
                       unroll_factor: asc.ConstExpr[int]):
    ...

@asctile.jit
def transpose_2_axis(in_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                     input_shape: asc.ConstExpr,
                     axis_step: asc.ConstExpr,
                     store_axis: asc.ConstExpr,
                     ub_load_shape: asc.ConstExpr,
                     load_shape: asc.ConstExpr,
                     permute: asc.ConstExpr,
                     block_count: int,
                     block_width: int,
                     unroll_factor: asc.ConstExpr[int]):
    ...

# --- Host helpers ---

def simplify_shape(shape, perm):
    ...

# --- Public callable ---

def transpose(x: torch.Tensor, perm: list) -> torch.Tensor:
    ...
```

## 16. Validation Plan

1. **Syntax check:** `python3 -m py_compile candidate.py`
2. **Static contract check:** Verify all imports, no `asc2` references,
   only `asctile.*` APIs, `asc.ConstExpr` for compile-time params.
3. **Exact-v2 compile gate:**
   `run_local_compile_gate.sh --candidate candidate.py --op transpose`
   Expect 20/20 case routes to lower successfully.
4. **No camodel numerical smoke needed:** transpose is exact; compile
   evidence is sufficient for the compile component (0.2 weight).
5. **NPU evaluation:** CANNBench hardware run for accuracy + performance.

## 17. Risk Register

| risk | impact | mitigation |
|------|--------|------------|
| int8 `.transpose()` rejected by asctile | case 19 fails | fallback: cast to int16, transpose, cast back |
| `asctile.transpose(*perm)` rejected for rank 5 | cases 15,16 fail | reduce to rank 4 by host-iter on smallest dim |
| UB overflow on 2-D transpose_block | cases 2-5,10 | use unroll_factor=1, reduce BLOCK_H |
| DMA alignment reject on odd last-dim sizes | multiple | always pad tile dims to items_in_block |
| Large host iteration count (case 14: 254 iters) | perf | batch host loops into fewer kernel calls if possible |
| `store_offsets` list mutation inside JIT | compile error | construct as `ConstExpr` from host, use `static_range` |
| `asctile.range` `parallel` keyword NOT available | compile error | not used; only `unroll_factor` and `gm_barrier` |

DESIGN_DONE.
