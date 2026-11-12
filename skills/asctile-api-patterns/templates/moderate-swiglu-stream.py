"""Flat SwiGLU schedules for moderate streams on the public 950PR wheel."""

import torch
from asc.experimental import asctile


@asctile.jit(reuse_alloc=2)
def single_row_edge(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                    half: asctile.ConstExpr[int], tile: asctile.ConstExpr[int],
                    blocks: asctile.ConstExpr[int]):
    gx = asctile.global_tensor(x, [2 * half])
    gy = asctile.global_tensor(y, [half])
    block = asctile.block_idx()
    waves = half // (blocks * tile)
    for step in asctile.range(waves, unroll_factor=2):
        col = (step * blocks + block) * tile
        gate = asctile.copy_in(gx, [col], [tile])
        value = asctile.copy_in(gx, [half + col], [tile])
        out = (gate / (1.0 + asctile.exp(-gate))) * value
        asctile.copy_out(out, gy, [col])
    tail = half % (blocks * tile)
    if tail:
        offset = block * 8
        if offset < tail:
            col = waves * blocks * tile + offset
            count = min(8, tail - offset)
            gate = asctile.copy_in(gx, [col], [8], real_shape=[count], pad_value=0.0)
            value = asctile.copy_in(gx, [half + col], [8], real_shape=[count], pad_value=0.0)
            out = (gate / (1.0 + asctile.exp(-gate))) * value
            asctile.copy_out(out, gy, [col], real_shape=[count])


@asctile.jit(reuse_alloc=2)
def column_stripe(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                  rows: asctile.ConstExpr[int], half: asctile.ConstExpr[int],
                  tile: asctile.ConstExpr[int]):
    gx = asctile.global_tensor(x, [2 * rows * half])
    gy = asctile.global_tensor(y, [rows * half])
    total = rows * asctile.ceildiv(half, tile)
    for t in asctile.range(asctile.block_idx(), total, asctile.block_num(), unroll_factor=2):
        row = t % rows
        col = (t // rows) * tile
        valid = min(tile, half - col)
        gate = asctile.copy_in(gx, [2 * row * half + col], [tile], real_shape=[valid], pad_value=0.0)
        value = asctile.copy_in(gx, [(2 * row + 1) * half + col], [tile], real_shape=[valid], pad_value=0.0)
        gate_f = gate.to(asctile.float32)
        out = (gate_f / (1.0 + asctile.exp(-gate_f))) * value.to(asctile.float32)
        asctile.copy_out(out.to(gate.dtype), gy, [row * half + col], real_shape=[valid])


def launch_moderate_flat(x, y, rows, half, aivs, usable_ub):
    """Return True after launching an eligible schedule on its actual grid."""
    if min(rows, half, aivs, usable_ub) < 1:
        return False
    itemsize = x.element_size()
    if half * itemsize <= 32768 or 2 * rows * half * itemsize > 64 * 1024**2:
        return False

    if rows == 1 and x.dtype == torch.float32:
        waves_needed = (half + aivs * 8192 - 1) // (aivs * 8192)
        waves = 1 << (waves_needed - 1).bit_length()
        tile = (half // (waves * aivs) // 8) * 8
        tail = half - waves * aivs * tile
        if tile >= 8 and 24 * tile <= usable_ub and tail <= aivs * 8:
            single_row_edge[aivs](x, y, half, tile, aivs)
            return True

    if x.dtype == torch.bfloat16 and 1 < rows < aivs and aivs * 8192 < half <= 2 * aivs * 8192:
        chunks = 2 * aivs
        tile = ((half + 16 * chunks - 1) // (16 * chunks)) * 16
        if (half + tile - 1) // tile == chunks and 20 * tile <= usable_ub:
            column_stripe[aivs](x, y, rows, half, tile)
            return True
    return False
