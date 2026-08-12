# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math

import asc2
import pytest
import torch

from .helpers import CORE_NUM, DYNAMIC, STATIC, select_reduce_tile


@asc2.jit(reuse_alloc=1)
def reduce_max_d_last_axis(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_num_rows,
                           input_num_cols, output_length, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr,
                           contiguous: asc2.ConstExpr):
    out_gm = asc2.global_tensor(output_ptr, [output_length])
    tile_rows = tile_shape[0]
    max_blocks = asc2.ceildiv(input_num_rows, tile_rows)

    if contiguous:
        # Narrow reduce axis: aligning tile_cols up to 32 B would read 2-4x the
        # bytes, so load tile_rows*C as one contiguous 1-D run and reshape to
        # [tile_rows, C] in UB (tile_rows is a 32-byte multiple, so tile_rows*C is
        # too). Reads past the last row are padded with the -inf identity.
        flat_gm = asc2.global_tensor(input_ptr, [input_num_rows * input_num_cols])
        tile_elems = tile_shape[0] * tile_shape[1]
        for i in asc2.range(asc2.block_idx(), max_blocks, asc2.block_num(), unroll_factor=unroll_factor):
            flat = asc2.copy_in(flat_gm, [i * tile_elems], [tile_elems], pad_value=float("-inf"))
            block = flat.reshape([tile_shape[0], tile_shape[1]])
            asc2.copy_out(asc2.reduce_max(block, 1), out_gm, [i * tile_rows])
    else:
        in_gm = asc2.global_tensor(input_ptr, [input_num_rows, input_num_cols])
        iters = asc2.ceildiv(input_num_cols, tile_shape[1])
        for i in asc2.range(asc2.block_idx(), max_blocks, asc2.block_num(), unroll_factor=unroll_factor):
            cache = asc2.full([tile_rows], float("-inf"), dtype=in_gm.dtype)
            for j in asc2.range(iters):
                block = asc2.copy_in(in_gm, [i * tile_rows, j * tile_shape[1]], tile_shape, pad_value=float("-inf"))
                cache = asc2.maximum(cache, asc2.reduce_max(block, 1))
            asc2.copy_out(cache, out_gm, [i * tile_rows])


@asc2.jit(reuse_alloc=1)
def reduce_max_d_middle_axis(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, outer, mid: asc2.ConstExpr,
                             inner: asc2.ConstExpr, tile_cols: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    # Reduce the middle axis of [outer, mid, inner], viewed as [outer*mid, inner]:
    # each `outer` index owns exactly `mid` consecutive rows. `mid`/`inner` define
    # the tile shape so they are ConstExpr (copy_in shape must be static); only
    # `outer` (the batch) is dynamic. Loading [mid, tile_cols] at row o*mid folds
    # the middle axis via reduce over axis 0. tile_cols is `inner` aligned up to
    # 32 bytes; the padded columns (>= inner) may read the next outer's row but are
    # excluded by copy_out real_shape=[inner], so the result is unaffected.
    in_gm = asc2.global_tensor(input_ptr, [outer * mid, inner])
    out_gm = asc2.global_tensor(output_ptr, [outer * inner])

    for o in asc2.range(asc2.block_idx(), outer, asc2.block_num(), unroll_factor=unroll_factor):
        block = asc2.copy_in(in_gm, [o * mid, 0], [mid, tile_cols], pad_value=float("-inf"))
        reduced = asc2.reduce_max(block, 0)
        asc2.copy_out(reduced, out_gm, [o * inner], real_shape=[inner])


def run_last_axis(profiler, runs, kernel_type, in_tensor, out_tensor, num_rows, num_cols, tile_rows, tile_cols,
                  itemsize, contiguous, unroll_factor=2):
    align = 32 // itemsize
    # tile_rows is aligned to 32 B; for the contiguous path this also makes the
    # 1-D run tile_rows*C a 32-byte multiple.
    tile_shape = [asc2.ceildiv(tile_rows, align) * align, tile_cols]
    block_num = min(CORE_NUM, asc2.ceildiv(num_rows, tile_shape[0]))

    params = [in_tensor, out_tensor]
    if kernel_type == STATIC:
        params.extend([asc2.ConstExpr(num_rows), asc2.ConstExpr(num_cols), asc2.ConstExpr(num_rows)])
    else:
        params.extend([num_rows, num_cols, num_rows])
    params.extend([tile_shape, unroll_factor, contiguous])

    with profiler.profile():
        for _ in range(runs):
            reduce_max_d_last_axis[block_num](*params)


# Last-axis reduction cases (reduce the final dimension).
@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("test_name, input_shape, input_dtype, tiling", [
    ("reduce_max_last_1", [200, 10], torch.float32, select_reduce_tile([200, 10])),
    ("reduce_max_last_2", [13, 2048, 32], torch.float32, select_reduce_tile([13, 2048, 32])),
    ("reduce_max_last_3", [10, 2048, 64], torch.float32, select_reduce_tile([10, 2048, 64])),
    ("reduce_max_last_4", [45, 2048, 4], torch.float32, select_reduce_tile([45, 2048, 4])),
    ("reduce_max_last_5", [64, 2048, 8], torch.float32, select_reduce_tile([64, 2048, 8])),
    ("reduce_max_last_6", [70, 2048, 16], torch.float32, select_reduce_tile([70, 2048, 16])),
    ("reduce_max_last_7", [2048, 83, 18], torch.float32, select_reduce_tile([2048, 83, 18])),
    ("reduce_max_last_8", [1500, 1, 61], torch.float32, select_reduce_tile([1500, 1, 61])),
    ("reduce_max_last_9", [3072, 113, 24], torch.float32, select_reduce_tile([3072, 113, 24])),
    ("reduce_max_last_10", [4608, 115, 12], torch.float32, select_reduce_tile([4608, 115, 12])),
    ("reduce_max_last_11", [1500, 61, 61], torch.float32, select_reduce_tile([1500, 61, 61])),
])
def test_reduce_max_last_axis(profiler, runs, kernel_type, test_name, input_shape, input_dtype, tiling):
    num_rows_flattened, num_cols, tile_rows, tile_cols, contiguous = tiling

    in_tensor = torch.randn([num_rows_flattened, num_cols], dtype=input_dtype)
    out_tensor = torch.zeros([num_rows_flattened], dtype=input_dtype)

    run_last_axis(profiler, runs, kernel_type, in_tensor, out_tensor, num_rows_flattened, num_cols, tile_rows,
                  tile_cols, input_dtype.itemsize, contiguous)

    expected = torch.amax(in_tensor, dim=1)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)


# Middle-axis reduction cases (reduce a non-final dimension).
@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("test_name, input_shape, reduce_axis, input_dtype", [
    ("reduce_max_mid_1", [1, 128, 144], 1, torch.float32),
    ("reduce_max_mid_2", [1024, 100, 2, 1], 2, torch.float32),
    ("reduce_max_mid_3", [64, 32, 48], 1, torch.float32),
    ("reduce_max_mid_4", [8, 4, 2, 64], 1, torch.float32),
])
def test_reduce_max_middle_axis(profiler, runs, kernel_type, test_name, input_shape, reduce_axis, input_dtype):
    outer = math.prod(input_shape[:reduce_axis])
    mid = input_shape[reduce_axis]
    inner = math.prod(input_shape[reduce_axis + 1:])
    unroll_factor = 2

    in_tensor = torch.randn(input_shape, dtype=input_dtype)
    out_tensor = torch.zeros([outer * inner], dtype=input_dtype)

    if inner == 1:
        # inner == 1 makes the middle-axis reduction identical to a last-axis
        # reduce of [outer, mid]; route it to the faster grid-stride last-axis
        # kernel instead of one tiny per-outer tile.
        _, _, tile_rows, tile_cols, contiguous = select_reduce_tile([outer, mid], input_dtype.itemsize)
        run_last_axis(profiler, runs, kernel_type, in_tensor, out_tensor, outer, mid, tile_rows, tile_cols,
                      input_dtype.itemsize, contiguous)
    else:
        ALIGNMENT_ELEMENTS = 32 // input_dtype.itemsize
        tile_cols = asc2.ceildiv(inner, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS
        block_num = min(CORE_NUM, outer)
        params = [in_tensor, out_tensor]
        if kernel_type == STATIC:
            params.append(asc2.ConstExpr(outer))
        else:
            params.append(outer)
        params.extend([mid, inner, tile_cols, unroll_factor])
        with profiler.profile():
            for _ in range(runs):
                reduce_max_d_middle_axis[block_num](*params)

    expected = torch.amax(in_tensor, dim=reduce_axis).reshape([outer * inner])
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
