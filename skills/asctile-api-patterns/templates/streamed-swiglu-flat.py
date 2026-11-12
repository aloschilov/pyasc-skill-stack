"""Process [R, 2H] in full contiguous chunks and one partial chunk per row."""

import torch
from asc.experimental import asctile


@asctile.jit(reuse_alloc=2)
def swiglu_flat(
    x: asctile.GlobalAddress,
    y: asctile.GlobalAddress,
    R: asctile.ConstExpr[int],
    H: asctile.ConstExpr[int],
    T: asctile.ConstExpr[int],
    U: asctile.ConstExpr[int],
):
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    whole = H // T
    nfull = R * whole
    blocks = asctile.block_num()
    block = asctile.block_idx()
    q = nfull // blocks
    rem = nfull % blocks
    begin = q * block + min(block, rem)
    end = q * (block + 1) + min(block + 1, rem)
    for t in asctile.range(begin, end, unroll_factor=U):
        row = t // max(1, whole)
        col = (t % max(1, whole)) * T
        gate_tile = asctile.copy_in(
            gx, [row * 2 * H + col], [T]
        )
        value_tile = asctile.copy_in(
            gx, [row * 2 * H + H + col], [T]
        )

        gate_f = gate_tile.to(asctile.float32)
        value_f = value_tile.to(asctile.float32)

        sig = gate_f / (1.0 + asctile.exp(-gate_f))
        out_f = sig * value_f

        out_dtype = gate_tile.dtype
        out_tile = out_f.to(out_dtype)

        asctile.copy_out(out_tile, gy, [row * H + col])
    if H % T:
        for row in asctile.range(block, R, blocks):
            col = whole * T
            valid = H % T
            gate_tile = asctile.copy_in(
                gx, [row * 2 * H + col], [T], real_shape=[valid], pad_value=0.0
            )
            value_tile = asctile.copy_in(
                gx, [row * 2 * H + H + col], [T], real_shape=[valid], pad_value=0.0
            )

            gate_f = gate_tile.to(asctile.float32)
            value_f = value_tile.to(asctile.float32)

            sig = gate_f / (1.0 + asctile.exp(-gate_f))
            out_f = sig * value_f

            out_dtype = gate_tile.dtype
            out_tile = out_f.to(out_dtype)

            asctile.copy_out(out_tile, gy, [row * H + col], real_shape=[valid])



def whole_bf16_grid(rows, half, dtype, aivs, usable_ub):
    """Select complete four-stage groups for a grid of aligned BF16 rows."""
    if dtype != torch.bfloat16 or aivs < 1 or rows < 8 * aivs:
        return 0
    if half % 16 or not 8192 <= 2 * half <= 32768 or 32 * half > usable_ub:
        return 0
    low = (7 * aivs + 7) // 8
    for blocks in range(aivs, low - 1, -1):
        if rows % (4 * blocks) == 0:
            return blocks
    return 0
