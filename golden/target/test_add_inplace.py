# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""In-place add target test: a <- a + b (aclnnInplaceAdd reference)."""

import math

import asc2
import pytest
import torch


STATIC = "static"
DYNAMIC = "dynamic"


# ---------------------------------------------------------------------------
# Host-side tiling selector (UB-budget, mirror of test_reciprocal.py /
# test_addcdiv.py). In-place add keeps two tiles live per iteration
# (the a tile and the b tile) because the output aliases a.
# ---------------------------------------------------------------------------
_UB_BUDGET_BYTES = 192 * 1024
_UB_RESERVE_BYTES = 1024
_CORE_NUM = 72
_MIN_TILE_ELEMS = 128
_TILES_PER_CORE = 2


def _ub_budget_bytes():
    return _UB_BUDGET_BYTES


def _select_elementwise_tile(shape, itemsize, live_tensors, unroll_factor=2):
    length = math.prod(shape)
    align = 32 // itemsize

    # Lever 1: largest tile that fits UB with double buffering, sized against
    # the number of tiles live at once = live_tensors * unroll_factor.
    per_buffer = (_ub_budget_bytes() - _UB_RESERVE_BYTES) // itemsize // (live_tensors * unroll_factor)
    ub_tile = max(align, (per_buffer // align) * align)

    # Lever 2/3: ~tiles_per_core tiles per core across the full grid, floored
    # to a useful size and capped by the UB tile and the length.
    per_core = -(-length // _CORE_NUM)
    tile = -(-per_core // _TILES_PER_CORE)
    tile = -(-tile // align) * align
    tile = max(_MIN_TILE_ELEMS, min(tile, ub_tile))
    tile = max(align, min(tile, -(-length // align) * align))

    block_num = min(_CORE_NUM, -(-length // tile))
    return (length, tile, block_num, unroll_factor)


# ---------------------------------------------------------------------------
# In-place add kernel: a <- a + b.
#
# a_ptr is BOTH the first input AND the destination; b_ptr is a read-only
# addend. A single a_gm handle is wrapped once and used as both the copy_in
# source and the copy_out destination. No separate output global_tensor,
# no third pointer. copy_in past the extent auto-pads and copy_out clamps
# to the declared global_tensor shape, so block_length * block_num may
# exceed input_length safely (no host padding, no tail branch).
# ---------------------------------------------------------------------------
@asc2.jit(reuse_alloc=0)
def add_inplace(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, input_length,
                tile_length: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    a_gm = asc2.global_tensor(a_ptr, [input_length])   # a_gm is BOTH input and output
    b_gm = asc2.global_tensor(b_ptr, [input_length])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        at = asc2.copy_in(a_gm, [current_offset], [tile_length])
        bt = asc2.copy_in(b_gm, [current_offset], [tile_length])
        zt = at + bt
        asc2.copy_out(zt, a_gm, [current_offset])   # store back into a_gm


# ---------------------------------------------------------------------------
# ~10 representative shapes spanning small to large, 1-D and multi-D.
# float32 primary, plus two float16 cases. itemsize=4 for f32, 2 for f16;
# live_tensors=2 (a tile + b tile; the output aliases a).
# ---------------------------------------------------------------------------
_TILINGS = [
    # (test_name, input_shape, input_dtype, tiling)
    ("add_inplace_test_1",  [8192],        torch.float32, _select_elementwise_tile([8192],        4, 2)),
    ("add_inplace_test_2",  [9216],        torch.float32, _select_elementwise_tile([9216],        4, 2)),
    ("add_inplace_test_3",  [16, 2048],    torch.float32, _select_elementwise_tile([16, 2048],    4, 2)),
    ("add_inplace_test_4",  [32, 4096],    torch.float32, _select_elementwise_tile([32, 4096],    4, 2)),
    ("add_inplace_test_5",  [87768],       torch.float32, _select_elementwise_tile([87768],       4, 2)),
    ("add_inplace_test_6",  [395520],      torch.float32, _select_elementwise_tile([395520],      4, 2)),
    ("add_inplace_test_7",  [979139],      torch.float32, _select_elementwise_tile([979139],      4, 2)),
    ("add_inplace_test_8",  [1024, 1024],  torch.float32, _select_elementwise_tile([1024, 1024],  4, 2)),
    ("add_inplace_test_9",  [1024, 1024],  torch.float16, _select_elementwise_tile([1024, 1024],  2, 2)),
    ("add_inplace_test_10", [98166, 128],  torch.float16, _select_elementwise_tile([98166, 128],  2, 2)),
]


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("test_name, input_shape, input_dtype, tiling", _TILINGS)
def test_add_inplace(profiler, runs, kernel_type, test_name, input_shape, input_dtype, tiling):
    length, tile_length, block_num, unroll_factor = tiling

    # 1-D flatten of the (possibly multi-D) shape; the kernel is uniform 1-D.
    length = math.prod(input_shape)

    a = torch.randn([length], dtype=input_dtype, device="cpu")
    b = torch.randn([length], dtype=input_dtype, device="cpu")

    # Snapshot the ORIGINAL a BEFORE any launch mutates it. Each launch does
    # a <- a + b, so after `runs` launches a == a0 + runs * b. Reconstruct
    # that exactly so the assert stays correct for both --runs 1 and
    # --profile --runs N. Do NOT re-seed a inside the profiler loop.
    a0 = a.clone()
    expected = a0 + runs * b

    params = [a, b]                       # NO separate output tensor — a is the output
    if kernel_type == STATIC:
        params.append(asc2.ConstExpr(length))
    else:
        params.append(length)
    params.extend([tile_length, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            add_inplace[block_num](*params)   # mutates a in place each launch

    if input_dtype == torch.float16:
        atol, rtol = 4e-3, 4e-3
    else:
        atol, rtol = 1e-3, 1e-3
    torch.testing.assert_close(a, expected, atol=atol, rtol=rtol)
