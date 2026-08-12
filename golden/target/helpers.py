# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Shared tiling helpers for the asc2 target op tests (addcdiv, reciprocal,
# reduce_max). Kept here so the UB-budget query and the elementwise/reduce tile
# selectors are defined once instead of copy-pasted per test file.

import math

STATIC = "static"
DYNAMIC = "dynamic"


def ub_budget_bytes(default=192 * 1024):
    # Physical Unified Buffer of the target SoC, queried from the runtime so the
    # tiling tracks the part (e.g. 192 KB on Ascend910B*, 248 KB on Ascend950*)
    # instead of hardcoding one box. current_platform() reflects the REAL device;
    # get_soc_version() is a cached module default (Ascend910B1) that the autouse
    # set_platform fixture only updates at run time -- too late for this
    # import-time tiling. Fall back to the cached SoC, then 192 KB, when the
    # device runtime is unavailable (e.g. host-only test collection).
    try:
        from asc.lib import runtime
        from asc.runtime.launcher import get_platform_info
        soc = runtime.get_soc_version()
        return get_platform_info(soc).ub_size
    except Exception:
        return default


UB_BUDGET_BYTES = ub_budget_bytes()
UB_RESERVE_BYTES = 1024
CORE_NUM = 72
MIN_TILE_ELEMS = 128
TILES_PER_CORE = 2
BUFFER_NUM = 4  # 2 (ping-pong) * unroll_factor (2)


# Elementwise tiling selector:
# Mirrors the canonical test_vadd.py pattern: no host padding, no tail branch,
# a uniform in-kernel block loop, and per-shape (block_num, tile_length) chosen
# from the physical UB budget so every AI core is used on large tensors.
# Three levers:
#   1. tile_length capped by the UB budget with double buffering, sized against
#      the number of tiles simultaneously live (live_tensors * unroll_factor);
#   2. ~2 tiles per core so unroll_factor=2 can overlap load/store across tiles;
#   3. block_num = min(72, ceildiv(length, tile_length)) -- spread across the
#      full grid for big tensors, collapse to few cores for tiny ones.
def select_elementwise_tile(shape, itemsize, live_tensors, unroll_factor=2, ub_budget=UB_BUDGET_BYTES,
                            reserve=UB_RESERVE_BYTES, core_num=CORE_NUM, min_tile=MIN_TILE_ELEMS,
                            tiles_per_core=TILES_PER_CORE):
    length = math.prod(shape)
    align = 32 // itemsize
    # Lever 1: the largest tile that fits the UB budget with double buffering.
    per_buffer = (ub_budget - reserve) // itemsize // (live_tensors * unroll_factor)
    ub_tile = max(align, (per_buffer // align) * align)
    # Lever 2/3: aim for ~tiles_per_core tiles per core across the full grid,
    # never smaller than a useful floor, never larger than the UB tile or length.
    per_core = -(-length // core_num)
    tile = -(-per_core // tiles_per_core)
    tile = -(-tile // align) * align
    tile = max(min_tile, tile)
    tile = min(tile, ub_tile)
    length_aligned = -(-length // align) * align
    tile = max(align, min(tile, length_aligned))
    block_num = min(core_num, -(-length // tile))
    return (length, tile, block_num, unroll_factor)


# Reduction tiling selector (see pyasc-api-patterns/reduction-tiling.md).
# Three levers drive near-CANN parity on this memory-bound op:
#   1. all AI cores -- the kernel uses a grid-stride loop over row-tiles, so
#      block_num = min(72, ceildiv(R, tile_rows)) spreads R across the full grid;
#   2. contiguous unpadded reduce axis -- tile_cols = C. When C is a 32-byte
#      multiple a plain 2-D tile is already contiguous. When C * itemsize < 32 B
#      a 2-D tile would have to pad the inner dim up to 32 B and move 2-4x the
#      bytes, so the kernel instead loads tile_rows*C elements as ONE contiguous
#      1-D run and reshapes to [tile_rows, C] in UB (no padded DMA);
#   3. tile_rows sized to the per-core block against the physical UB, in a few
#      large tiles (double buffering is secondary to contiguity + cores).
# Bounds are handled entirely in-kernel: copy_in(pad_value=-inf) pads reads past
# the source edge with the reduction identity and copy_out clamps writes to the
# declared output extent, so there is NO host-side padding or tail branch.
def select_reduce_tile(shape, itemsize=4, ub_budget=UB_BUDGET_BYTES, reserve=UB_RESERVE_BYTES, buffer_num=BUFFER_NUM,
                       core_num=CORE_NUM):
    dims = [d for d in shape if d != 1]  # reshape short-circuit (drop size-1 dims)
    R = 1
    for d in dims[:-1]:
        R *= d
    C = dims[-1] if dims else 1
    align = 32 // itemsize
    per_buffer = (ub_budget - reserve) // itemsize // buffer_num
    rows_per_block = -(-R // core_num)  # ceildiv: spread across ALL cores
    # A reduce axis narrower than 32 bytes would force the 2-D copy_in to pad the
    # inner dim (C=4 -> 8, C=2 -> 8), moving 2-4x the bytes on a memory-bound op.
    # Flag it so the kernel takes the contiguous 1-D-load + in-UB reshape path.
    contiguous = C > 1 and C * itemsize < 32
    if contiguous or C <= per_buffer:  # pack rows into a [tile_rows, tc] block
        tc = C if (contiguous or C % align == 0) else -(-C // align) * align
        ub_cap = max(1, per_buffer // tc)
        n_tiles = -(-rows_per_block // ub_cap)  # tiles to cover a core block
        tr = -(-rows_per_block // n_tiles)  # even, large tiles
    else:  # large-C: tile the column axis
        tr = 1
        tc = (per_buffer // align) * align
    return (R, C, tr, tc, contiguous)
