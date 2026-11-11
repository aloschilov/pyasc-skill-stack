# Tile contiguous BF16 pairs

For SwiGLU on a contiguous final axis of length two, flatten the outer
dimensions into a pair count. One contiguous BF16 read covers both operands.
Select the implementation according to the target and the candidate contract.
On the archive-bundled 950PR public PyASC build with `asctile.reindex` and
reusable int32 indices, route eligible pairs through the existing
[short-half slab gather](reindex-slabs.md). For H=1 the two vectors contain
even and odd element offsets. Load each vector once per block, gather BF16
from a single contiguous input tile and cast each half to FP32 for SwiGLU.
Derive the row count and loop unroll using the same DMA, UB and work bounds
as the other qualified gathers. Keep the partial tile in the main grid.

On the native-enabled 950PR pin qualified below, use the
[register kernel and selector](../templates/register-bf16-swiglu.py).
For a public pin without a qualified gather, the
[packed transpose template](../templates/packed-bf16-swiglu.py) supplies
the two-plane path. Select among these implementations from the candidate
contract and the APIs verified on the active compiler build.

### Native BF16 register path

Inside the contiguous BF16, normalized-last-axis, width-two branch, query UB
and the active vector count. Pass both to
`_register_pair_tile_size(rows, ub_bytes, vector_count)`. A returned tile
selects `_swiglu_register_pairs[vector_blocks(groups)](input, out, rows, groups)`
where groups is ceil(rows / tile). Return the output after launch. On a
declined tile, try the public packed path and then the general row path.
The qualified specialization takes four arguments and fixes its physical
tile at 2048 pairs. Its compiled UB requirement was 12288 bytes on PyASC
`6b3447b4` with CANN 9.2. Requalify for other compiler or register targets.

Load the interleaved input once. Two consecutive 128-lane BF16 registers
contain 128 complete pairs. `DeInterleave` separates even and odd lanes.
Widen both register layouts to FP32, evaluate the exponential on each gate
lane once, multiply by its value, round to BF16 and combine the disjoint
output lanes. Only the input and output tiles occupy UB. Keep the valid
extents on the copies so the last block writes exactly the remaining rows.

### Public transpose path on the earlier pin

After platform setup and empty-input handling, integrate the template at the
start of the existing contiguous SwiGLU dispatch. The surrounding operator
has already validated contiguity and normalized the split axis.

```python
if dtype == torch.bfloat16 and d == rank - 1 and size == 2:
    tile = _pair_tile_size(rows, _query_platform_ub_bytes(), vector_blocks(rows))
    if tile:
        groups = (rows + tile - 1) // tile
        _swiglu_pair_groups[vector_blocks(groups)](input, out, rows, groups, tile)
        return out
# Continue into the existing general row kernel.
```

Pass the effective UB capacity in bytes. When the caller supplies an additional
runner limit, take the smaller capacity at the query boundary. The selector
returns None when the capacity is missing or no qualified tile fits. Falling
through here preserves the general operator's own resource handling.

Each block visits whole tiles in a block-stride loop. For each tile the valid
pair count is the smaller of the tile size and the remaining rows. Keep static
physical shapes and pass dynamic valid extents to the input and output copies.
This assigns a partial tile to one block in the same launch, including when the
whole input fits in one partial tile.

The two aligned BF16 transposes move parity to the leading axis. Select the
gate and value planes independently before applying the nonlinear function.
On PyASC 6b3447b4, selecting with `where` and reducing over that two-element
axis produces two single-plane tensors. Zero the unselected lanes using
`where` so NaNs in the other plane cannot leak through multiplication by zero.
Compute FP32 sigmoid only on the gate and multiply by the original value.

The allocation table in the template is specific to this computation, layout,
compiler and JIT options. In 950PR CaModel the 4096-pair specialization
requested 369664 bytes at a 253952-byte capacity. Recheck allocation before
extending the size ladder or changing the kernel. Validate multiple tile sizes,
an exact tile, a partial-only input, a block-stride tail, nonfinite lanes and
the output write boundary before timing.
