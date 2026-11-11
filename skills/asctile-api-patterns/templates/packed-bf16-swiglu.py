"""Contiguous adjacent BF16 gates and values, processed on the active vector grid."""
from asc.experimental import asctile


@asctile.jit
def _swiglu_pair_groups(xptr: asctile.GlobalAddress, yptr: asctile.GlobalAddress,
                        rows: int, groups: int, tile: asctile.ConstExpr[int]):
    x = asctile.global_tensor(xptr, [2 * rows])
    y = asctile.global_tensor(yptr, [rows])
    for group in asctile.range(asctile.block_idx(), groups, asctile.block_num()):
        first = group * tile
        valid = min(tile, rows - first)
        flat = asctile.copy_in(x, [2 * first], [2 * tile],
                               real_shape=[2 * valid], pad_value=0.0)
        # A pair tile is [16, tile // 16, 2]. Move parity to the leading axis.
        columns = flat.reshape(16, tile // 8).transpose(1, 0)
        planes = columns.reshape(tile // 16, 32).transpose(1, 0).reshape(2, tile)
        xy = asctile.cast(planes, asctile.float32)
        plane_id = asctile.concat(asctile.full([1, tile], 0.0, asctile.float32),
                                  asctile.full([1, tile], 1.0, asctile.float32))
        is_gate = plane_id < 0.5
        gate = asctile.reduce_sum(asctile.where(is_gate, xy, 0.0), 0)
        value = asctile.reduce_sum(asctile.where(is_gate, 0.0, xy), 0)
        sigmoid = asctile.div(1.0, asctile.add(asctile.exp(-gate), 1.0))
        result = asctile.cast(gate * sigmoid * value, asctile.bfloat16)
        asctile.copy_out(result.reshape(tile), y, [first], real_shape=[valid])


def _pair_tile_size(rows, ub_bytes, vector_count):
    """Largest qualified tile that fits UB while keeping the vector grid occupied."""
    if ub_bytes is None or rows <= 0:
        return None
    # CaModel allocations for this kernel and pin, including the runtime tail.
    occupancy_limit = max(256, (rows + vector_count - 1) // vector_count)
    for tile, allocated in ((2048, 184832), (1024, 92416), (512, 46208), (256, 23104)):
        if allocated <= ub_bytes and tile <= occupancy_limit:
            return tile
    return None
