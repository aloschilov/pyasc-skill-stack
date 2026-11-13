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
    cache_input: asctile.ConstExpr[bool] = True,
):
    # Large exact-divisor FP32 streams visit every input lane once.
    gx = asctile.global_tensor(x, [2 * R * H], l2_cache=cache_input)
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



@asctile.jit(reuse_alloc=2)
def whole_fp32_rows(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                    rows: asctile.ConstExpr[int], half: asctile.ConstExpr[int],
                    grid: asctile.ConstExpr[int]):
    src = asctile.global_tensor(x, [2 * rows * half])
    dst = asctile.global_tensor(y, [rows * half])
    block = asctile.block_idx()
    for step in asctile.range(rows // grid, unroll_factor=4):
        row = block + step * grid
        gate = asctile.copy_in(src, [2 * row * half], [half])
        value = asctile.copy_in(src, [(2 * row + 1) * half], [half])
        out = (gate / (1.0 + asctile.exp(-gate))) * value
        asctile.copy_out(out, dst, [row * half])


def generic_flat_geometry(rows, half, itemsize, aivs, usable_ub):
    """Size the ordinary flat full-copy kernel and its same-launch remainder."""
    if min(rows, half, aivs, usable_ub) <= 0 or itemsize not in (2, 4):
        return 0, 0, 0
    cap = min(half, 16384 // itemsize, usable_ub // 32)
    if cap < 1:
        return 0, 0, 0
    tile = 1 << (cap.bit_length() - 1)
    full = rows * (half // tile)
    blocks = min(aivs, rows * ((half + tile - 1) // tile))
    two_stage_bytes = tile * (20 if itemsize == 2 else 24)
    unroll = 2 if full >= 2 * blocks and two_stage_bytes <= usable_ub else 1
    return tile, unroll, blocks


def four_wave_bf16_chunk(rows, half, dtype, aivs, usable_ub):
    """Cover nearly four waves with consecutive chunks on a short BF16 stream."""
    if rows != 1 or dtype != torch.bfloat16 or min(half, aivs, usable_ub) <= 0:
        return 0
    tile = (half // (4 * aivs)) // 16 * 16
    if not 2048 <= tile <= 4096 or 36 * tile > usable_ub:
        return 0
    remainder = half - 4 * aivs * tile
    if remainder > min(tile - 1, 16 * aivs):
        return 0
    return tile


def wide_fp32_chunk(rows, half, dtype, aivs, usable_ub):
    """Choose a full divisor when all rows together sustain three stages."""
    if dtype != torch.float32 or min(half, aivs, usable_ub) <= 0 or rows < aivs or half % 8:
        return 0
    ordinary_cap = min(half, 4096, usable_ub // 32)
    if ordinary_cap < 8:
        return 0
    base = 1 << (ordinary_cap.bit_length() - 1)
    if 4 * (half % base) < 3 * base:
        return 0
    capacity = min(half, usable_ub // 36)
    for width in range(capacity - capacity % 8, base, -8):
        if half % width == 0 and rows * (half // width) >= 3 * aivs:
            return width
    return 0


def whole_fp32_grid(rows, half, dtype, aivs, usable_ub):
    """Select four-stage block-stride ownership of moderate aligned FP32 rows."""
    if dtype != torch.float32 or aivs < 1 or not 4 * aivs <= rows <= 8 * aivs:
        return 0
    if half % 8 or not 8192 <= 4 * half <= 16384 or 48 * half > usable_ub:
        return 0
    if 8 * rows * half > 64 * 1024**2:
        return 0
    for grid in range(aivs - aivs % 2, (7 * aivs + 7) // 8 - 1, -2):
        if rows % (4 * grid) == 0:
            return grid
    return 0


def whole_bf16_grid(rows, half, dtype, aivs, usable_ub):
    """Select complete four-stage groups for a grid of aligned BF16 rows."""
    if dtype != torch.bfloat16 or aivs < 1 or rows < 8 * aivs:
        return 0
    if half % 16 or not 8192 <= 2 * half <= 32768 or 32 * half > usable_ub:
        return 0
    return _uniform_four_stage_grid(rows, aivs)


def long_fp16_grid(rows, half, dtype, aivs, usable_ub):
    """Select a uniform grid of full 16 KiB DMA chunks for a long FP16 row."""
    tile = 8192
    if dtype != torch.float16 or rows != 1 or aivs < 1 or half % tile:
        return 0
    chunks = half // tile
    if chunks < 64 * aivs or 28 * tile > usable_ub:
        return 0
    return _uniform_four_stage_grid(chunks, aivs)


def _uniform_four_stage_grid(items, aivs):
    low = (7 * aivs + 7) // 8
    for blocks in range(aivs, low - 1, -1):
        if items % (4 * blocks) == 0:
            return blocks
    return 0
