"""Packed contiguous BF16 pair example. Use the AscTile import from your runtime."""
from asc.experimental import asctile


@asctile.jit
def _swiglu_pair_tile(xptr: asctile.GlobalAddress, yptr: asctile.GlobalAddress,
                n: int, base: int, valid: asctile.ConstExpr[int]):
    x = asctile.global_tensor(xptr, [2 * n])
    y = asctile.global_tensor(yptr, [n])
    v = asctile.copy_in(x, [2 * base], [512], real_shape=[2 * valid], pad_value=0.0)
    # Input order is [row, column, parity], with 16 rows and 16 pairs per row.
    # The first transpose yields [column, parity, row].
    columns = v.reshape(16, 32).transpose(1, 0)
    # The second transpose yields [parity, row, column].
    cols = columns.reshape(16, 32).transpose(1, 0).reshape(2, 256)
    xy = asctile.cast(cols, asctile.float32)
    zeros = asctile.full([1, 256], 0.0, asctile.float32)
    ones = asctile.full([1, 256], 1.0, asctile.float32)
    lanes = asctile.concat(zeros, ones)
    is_a = lanes < 0.5
    inv = asctile.div(1.0, asctile.add(asctile.exp(asctile.mul(xy, -1.0)), 1.0))
    selected = asctile.where(is_a, asctile.mul(xy, inv), xy)
    value = asctile.cast(asctile.reduce_prod(selected, 0), asctile.bfloat16)
    out = value.reshape(256)
    asctile.copy_out(out, y, [base], real_shape=[valid])


@asctile.jit
def _swiglu_pair_groups(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                        rows: int, first_group: int, groups: int,
                        valid: asctile.ConstExpr[int]):
    for group in asctile.range(asctile.block_idx(), groups, asctile.block_num()):
        _swiglu_pair_tile(x, y, rows, (first_group + group) * 256, valid)
