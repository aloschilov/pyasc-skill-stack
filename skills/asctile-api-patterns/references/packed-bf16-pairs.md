# Tile contiguous BF16 pairs

For SwiGLU on a contiguous final axis of length two, flatten the outer
dimensions into a pair count. One contiguous BF16 read covers both operands.
Use the [public AscTile template](../templates/packed-bf16-swiglu.py) for
the device kernel and its host tile selector. Adapt the AscTile import to the
active runtime. Keep the existing dispatch for other layouts.

After platform setup, pass the active UB capacity in bytes and the available
vector count to `_pair_tile_size(rows, ub_bytes, vector_blocks(rows))`. Use
the largest qualified shape that fits the capacity and leaves enough work for
the vector grid. With a positive tile, compute the ceiling tile count, then
launch `_swiglu_pair_groups[vector_blocks(groups)](input, out, rows, groups, tile)`
once. A missing capacity or an unsupported size selects the qualified fallback.
Handle empty inputs on the host. Use the runner's effective UB limit if smaller.

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
