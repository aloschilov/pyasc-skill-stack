"""Two bounded contiguous BF16 row tiles per AIV on the public 950PR build."""

import torch
from asc.experimental import asctile


def two_wave_tile(rows, half, dtype, aivs, usable_ub):
    """Return a row height or zero so the caller can try the ordinary slab."""
    if dtype != torch.bfloat16 or rows <= 0 or aivs <= 0 or not 1 < half <= 16:
        return 0
    waves = 2 * aivs
    rows_per_tile = (rows + waves - 1) // waves
    tile = (rows_per_tile + 15) // 16 * 16
    lanes = tile * half
    if (rows + tile - 1) // tile != waves:
        return 0
    if 2 * lanes - 1 > 32767 or 28 * lanes > usable_ub:
        return 0
    return tile


def two_wave_gate_index(half, tile, device, cache):
    """Build one signed 16-bit index per gate lane for a selected two-wave tile."""
    if half <= 1 or tile <= 0 or 2 * tile * half - 1 > 32767:
        raise ValueError("two-wave index exceeds its selected geometry")
    stream_id = torch.npu.current_stream(device).npu_stream if device.type == "npu" else None
    key = ("two_wave_i16", device, stream_id, half, tile)
    if key not in cache:
        k = torch.arange(tile * half, device=device, dtype=torch.int32)
        cache[key] = (k + half * (k // half)).to(torch.int16)
    return cache[key]


@asctile.jit(reuse_alloc=2)
def two_wave_swiglu(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                    indices: asctile.GlobalAddress, rows: asctile.ConstExpr[int],
                    half: asctile.ConstExpr[int], tile: asctile.ConstExpr[int]):
    lanes = tile * half
    source = asctile.global_tensor(x, [2 * rows * half])
    output = asctile.global_tensor(y, [rows * half])
    table = asctile.global_tensor(indices, [lanes])
    gate_index = asctile.copy_in(table, [0], [lanes])
    value_index = gate_index + half
    block = asctile.block_idx()
    blocks = asctile.block_num()
    count = (rows + tile - 1) // tile
    q = count // blocks
    rem = count - q * blocks
    begin = q * block + min(block, rem)
    end = begin + q + min(1, max(0, rem - block))
    for t in asctile.range(begin, end, unroll_factor=2):
        first = t * tile
        valid = min(tile, rows - first) * half
        slab = asctile.copy_in(source, [2 * first * half], [2 * lanes],
                               real_shape=[2 * valid], pad_value=0.0)
        gate = asctile.reindex(slab, gate_index).to(asctile.float32)
        active = gate / (asctile.exp(-gate) + 1.0)
        value = asctile.reindex(slab, value_index).to(asctile.float32)
        asctile.copy_out((active * value).to(slab.dtype), output, [first * half],
                         real_shape=[valid])
