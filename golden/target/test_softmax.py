# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2
import pytest
import torch

STATIC = "static"
DYNAMIC = "dynamic"


@asc2.jit(reuse_alloc=1)
def softmax_fused(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_num_rows: asc2.ConstExpr,
                  input_num_cols: asc2.ConstExpr, tile_shape: asc2.ConstExpr, rows_per_core: asc2.ConstExpr,
                  unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_num_rows, input_num_cols])
    out_gm = asc2.global_tensor(output_ptr, [input_num_rows, input_num_cols])
    rows_per_block = rows_per_core
    start_offset = asc2.block_idx() * rows_per_block

    ub_loop = asc2.number(asc2.ceildiv(rows_per_block, tile_shape[0]), asc2.int_)
    tail_rows = asc2.number(tile_shape[0], asc2.int_)
    #TODO: remove redundant tail handling when the accuracy issue is resolved
    if asc2.block_idx() == asc2.block_num() - 1:
        tail_rows_per_block = input_num_rows - rows_per_block * (asc2.block_num() - 1)
        ub_loop = asc2.ceildiv(tail_rows_per_block, tile_shape[0])
        tail_rows = tail_rows_per_block - tile_shape[0] * (ub_loop - 1)

    for i in asc2.range(ub_loop, unroll_factor=unroll_factor):
        row_start_offset = start_offset + i * tile_shape[0]
        real_rows = tail_rows if i == ub_loop - 1 and asc2.block_idx() == asc2.block_num() - 1 else tile_shape[0]
        rows = asc2.copy_in(in_gm, [row_start_offset, 0], [tile_shape[0], tile_shape[1]],
                            real_shape=[real_rows, input_num_cols], pad_value=float('-inf'))
        out = asc2.softmax(rows)
        asc2.copy_out(out, out_gm, [row_start_offset, 0], real_shape=[real_rows, input_num_cols])


@asc2.jit(reuse_alloc=1)
def softmax_small_row(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_num_rows, input_num_cols,
                      tile_shape: asc2.ConstExpr, ub_loop, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_num_rows, input_num_cols])
    out_gm = asc2.global_tensor(output_ptr, [input_num_rows, input_num_cols])
    transposed_shape = tile_shape[::-1]

    for i in range(asc2.block_idx(), ub_loop, asc2.block_num(), unroll_factor=unroll_factor):
        rows = asc2.copy_in(in_gm, [i * tile_shape[0], 0], tile_shape, pad_value=float('-inf'), real_shape=tile_shape)
        rows = rows.transpose()
        row_max = asc2.reduce_max(rows, 0)
        row_max = row_max.broadcast_to(*transposed_shape)
        row_minus_max = rows - row_max
        numerator = row_minus_max.exp()
        denominator = asc2.reduce_sum(numerator, 0)
        denominator = denominator.broadcast_to(*transposed_shape)
        out = numerator / denominator
        out = out.transpose()
        asc2.copy_out(out, out_gm, [i * tile_shape[0], 0])


op_name = ["softmax_v2"]


# yapf: disable
@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params", [
# PYASC_TESTS_BEGIN
    ("softmax_test_1", 1, ([128, 1, 4], ), (torch.float32, ), ([128, 1, 4], ), (torch.float32, ), ([-1], ), (2, [-1]), 500, (128, 4, 1, 1, 128, 128, 8, 8)),
    ("softmax_test_2", 1, ([200, 1, 4], ), (torch.float32, ), ([200, 1, 4], ), (torch.float32, ), ([-1], ), (2, [-1]), 500, (200, 4, 1, 1, 256, 200, 8, 8)),
    ("softmax_test_3", 1, ([8, 5, 5], ), (torch.float32, ), ([8, 5, 5], ), (torch.float32, ), ([-1], ), (2, [-1]), 500, (40, 5, 1, 1, 64, 40, 8, 8)),
    ("softmax_test_4", 1, ([4, 5], ), (torch.float32, ), ([4, 5], ), (torch.float32, ), ([-1], ), (2, [-1]), 500, (4, 5, 1, 1, 64, 4, 8, 8)),
    ("softmax_test_5", 72, ([1024, 4, 4], ), (torch.float32, ), ([1024, 4, 4], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (4096, 4, 8, 57, 57, 1)),
    ("softmax_test_6", 1, ([256, 1, 4], ), (torch.float32, ), ([256, 1, 4], ), (torch.float32, ), ([-1], ), (2, [-1]), 500, (256, 4, 1, 1, 256, 256, 8, 8)),
    ("softmax_test_7", 72, ([2500, 8], ), (torch.float32, ), ([2500, 8], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (2500, 8, 8, 35, 35, 1)),
    ("softmax_test_8", 12, ([12, 2500], ), (torch.float32, ), ([12, 2500], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (12, 2500, 2504, 1, 1, 40)),
    ("softmax_test_9", 1, ([100, 4], ), (torch.float32, ), ([100, 4], ), (torch.float32, ), ([-1], ), (2, [-1]), 500, (100, 4, 1, 1, 128, 100, 8, 8)),
    ("softmax_test_10", 67, ([100, 2, 300], ), (torch.float32, ), ([100, 2, 300], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (200, 300, 304, 3, 3, 5)),
    ("softmax_test_11", 67, ([100, 4, 100], ), (torch.float32, ), ([100, 4, 100], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (400, 100, 104, 6, 6, 2)),
    ("softmax_test_12", 67, ([100, 2, 100], ), (torch.float32, ), ([100, 2, 100], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (200, 100, 104, 3, 3, 2)),
    ("softmax_test_13", 1, ([100, 1, 2], ), (torch.float32, ), ([100, 1, 2], ), (torch.float32, ), ([-1], ), (2, [-1]), 500, (100, 2, 1, 1, 128, 100, 8, 8)),
    ("softmax_test_14", 70, ([700, 1, 4], ), (torch.float32, ), ([700, 1, 4], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (700, 4, 8, 10, 10, 1)),
    ("softmax_test_15", 70, ([700, 6], ), (torch.float32, ), ([700, 6], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (700, 6, 8, 10, 10, 1)),
    ("softmax_test_16", 69, ([750, 1, 4], ), (torch.float32, ), ([750, 1, 4], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (750, 4, 8, 11, 11, 1)),
    ("softmax_test_17", 4, ([4, 2048], ), (torch.float32, ), ([4, 2048], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (4, 2048, 2048, 1, 1, 32)),
    ("softmax_test_18", 67, ([100, 8, 1, 64], ), (torch.float32, ), ([100, 8, 1, 64], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (800, 64, 64, 12, 12, 1)),
    ("softmax_test_19", 4, ([1, 4, 1, 300], ), (torch.float32, ), ([1, 4, 1, 300], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (4, 300, 304, 1, 1, 5)),
    ("softmax_test_20", 67, ([100, 2, 1, 302], ), (torch.float32, ), ([100, 2, 1, 302], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (200, 302, 304, 3, 3, 5)),
    ("softmax_test_21", 44, ([100, 551, 10], ), (torch.float32, ), ([100, 551, 10], ), (torch.float32, ), ([-1], ), (2, [-1]), 500, (55100, 10, 87, 2, 640, 60, 8, 16)),
    ("softmax_test_22", 65, ([100, 412, 10], ), (torch.float32, ), ([100, 412, 10], ), (torch.float32, ), ([-1], ), (2, [-1]), 500, (41200, 10, 65, 1, 640, 240, 8, 16)),
    ("softmax_test_23", 67, ([400, 2, 300], ), (torch.float32, ), ([400, 2, 300], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (800, 300, 304, 12, 12, 5)),
    ("softmax_test_24", 72, ([7376, 50], ), (torch.float32, ), ([7376, 50], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (7376, 50, 56, 103, 103, 1)),
    ("softmax_test_25", 67, ([100, 8, 1, 128], ), (torch.float32, ), ([100, 8, 1, 128], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (800, 128, 128, 12, 12, 2)),
    ("softmax_test_26", 70, ([200, 8, 1, 200], ), (torch.float32, ), ([200, 8, 1, 200], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (1600, 200, 200, 23, 23, 4)),
    ("softmax_test_27", 70, ([200, 8, 1, 256], ), (torch.float32, ), ([200, 8, 1, 256], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (1600, 256, 256, 23, 23, 4)),
    ("softmax_test_28", 72, ([7000, 1, 10], ), (torch.float32, ), ([7000, 1, 10], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (7000, 10, 16, 98, 98, 1)),
    ("softmax_test_29", 70, ([200, 8, 1, 300], ), (torch.float32, ), ([200, 8, 1, 300], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (1600, 300, 304, 23, 23, 5)),
    # TODO: UB overflow ("softmax_test_30", 72, ([800, 8, 256], ), (torch.float32, ), ([800, 8, 256], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (6400, 256, 256, 49, 89, 4)),
    # TODO: UB overflow ("softmax_test_31", 72, ([10000, 100, 100], ), (torch.float32, ), ([10000, 100, 100], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (1000000, 100, 104, 121, 13889, 2)),
    # TODO: UB overflow ("softmax_test_32", 72, ([800, 185, 100], ), (torch.float32, ), ([800, 185, 100], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (148000, 100, 104, 121, 2056, 2)),
    # TODO: UB overflow ("softmax_test_33", 72, ([512, 150, 150], ), (torch.float32, ), ([512, 150, 150], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (76800, 150, 152, 83, 1067, 3)),
    # TODO: UB overflow ("softmax_test_34", 72, ([1024, 1000, 50], ), (torch.float32, ), ([1024, 1000, 50], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (1024000, 50, 56, 225, 14223, 1)),
    # TODO: UB overflow ("softmax_test_35", 72, ([4, 1500, 27, 27], ), (torch.float32, ), ([4, 1500, 27, 27], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (162000, 27, 32, 394, 2250, 1)),
    # TODO: UB overflow ("softmax_test_36", 72, ([4096, 2, 39, 39], ), (torch.float32, ), ([4096, 2, 39, 39], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (319488, 39, 40, 315, 4438, 1)),
    # TODO: UB overflow ("softmax_test_37", 72, ([4096, 50, 50], ), (torch.float32, ), ([4096, 50, 50], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (204800, 50, 56, 225, 2845, 1)),
    # TODO: UB overflow ("softmax_test_38", 72, ([256, 200, 200], ), (torch.float32, ), ([256, 200, 200], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (51200, 200, 200, 63, 712, 4)),
    ("softmax_test_39", 72, ([8, 1500, 1, 512], ), (torch.float32, ), ([8, 1500, 1, 512], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (12000, 512, 512, 24, 167, 8)),
# TODO: UB overflow ("softmax_test_40", 72, ([512, 100, 100], ), (torch.float32, ), ([512, 100, 100], ), (torch.float32, ), ([-1], ), (2, [-1]), 1000, (51200, 100, 104, 121, 712, 2)),
# PYASC_TESTS_END
])
# yapf: enable
def test_softmax(profiler, runs, kernel_type, test_name, block_num, input_shapes, input_dtypes, output_shapes,
                 output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):
    unroll_factor = runtime_params[0]
    input_shape, input_dtype = input_shapes[0], input_dtypes[0]
    # Convert any shape to 2D
    if len(input_shape) == 1:
        input_shape_2d = [1, input_shape[0]]
    else:
        input_shape_2d = [torch.prod(torch.tensor(input_shape[:-1])).item(), input_shape[-1]]
    num_rows, num_cols = input_shape_2d
    in_tensor = torch.randn(input_shape_2d, dtype=input_dtype)
    out_tensor = torch.zeros(input_shape_2d, dtype=input_dtype)

    if tiling_key == 500:
        ALIGNMENT_ELEMENTS = max(16, 32 // input_dtype.itemsize)
        rows_per_iter = tiling_params[4]
        tile_shape = [
            asc2.ceildiv(rows_per_iter, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS,
            asc2.ceildiv(num_cols, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS
        ]
        ub_loop = asc2.ceildiv(input_shape_2d[0], rows_per_iter)
        assert tile_shape[0] % ALIGNMENT_ELEMENTS == 0
        assert tile_shape[1] % ALIGNMENT_ELEMENTS == 0
        with profiler.profile():
            for _ in range(runs):
                softmax_small_row[block_num](
                    in_tensor, out_tensor,
                    asc2.ConstExpr(input_shape_2d[0]) if kernel_type == STATIC else input_shape_2d[0],
                    asc2.ConstExpr(input_shape_2d[1]) if kernel_type == STATIC else input_shape_2d[1], tile_shape,
                    ub_loop, unroll_factor)
    else:
        ALIGNMENT_ELEMENTS = 32 // input_dtype.itemsize
        rows_per_iter, rows_per_core = None, None
        if tiling_key == 1000:
            rows_per_iter = tiling_params[3]
            rows_per_core = tiling_params[4]
        if tiling_key == 10000:
            rows_per_iter = tiling_params[1]
            rows_per_core = tiling_params[2]
        tile_shape = [rows_per_iter, asc2.ceildiv(num_cols, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS]
        with profiler.profile():
            for _ in range(runs):
                softmax_fused[block_num](
                    in_tensor, out_tensor,
                    asc2.ConstExpr(input_shape_2d[0]) if kernel_type == STATIC else input_shape_2d[0],
                    asc2.ConstExpr(input_shape_2d[1]) if kernel_type == STATIC else input_shape_2d[1], tile_shape,
                    rows_per_core, unroll_factor)

    expected = torch.softmax(in_tensor, dim=1)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
