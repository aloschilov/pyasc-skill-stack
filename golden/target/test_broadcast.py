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

STATIC = "static"
DYNAMIC = "dynamic"


@asc2.jit(reuse_alloc=1)
def broadcast_scalar(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, output_num_rows,
                     output_num_cols, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_length])
    out_gm = asc2.global_tensor(output_ptr, [output_num_rows, output_num_cols])

    cols_per_block = asc2.ceildiv(output_num_cols, asc2.block_num())
    start_offset = asc2.block_idx() * cols_per_block
    column_iters = asc2.ceildiv(cols_per_block, tile_shape[1])

    for i in asc2.range(column_iters, unroll_factor=unroll_factor):
        scalar = asc2.copy_in(in_gm, [0])
        res = asc2.full(tile_shape, scalar)
        asc2.copy_out(res, out_gm, [0, start_offset + i * tile_shape[1]])


@asc2.jit(reuse_alloc=1)
def broadcast_first_dim(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, output_num_rows,
                        output_num_cols, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_length])
    out_gm = asc2.global_tensor(output_ptr, [output_num_rows, output_num_cols])

    rows_per_block = asc2.ceildiv(output_num_rows, asc2.block_num())
    start_offset = asc2.block_idx() * rows_per_block
    row_iters = asc2.ceildiv(rows_per_block, tile_shape[0])

    column_iters = asc2.ceildiv(output_num_cols, tile_shape[1])

    for j in asc2.range(column_iters, unroll_factor=unroll_factor):
        col_start_offset = j * tile_shape[1]
        tensor_part = asc2.copy_in(in_gm, [col_start_offset], [tile_shape[1]])
        res = tensor_part.broadcast_to(tile_shape[0], tile_shape[1])
        for i in asc2.range(row_iters):
            asc2.copy_out(res, out_gm, [start_offset + i * tile_shape[0], col_start_offset])


@asc2.jit(reuse_alloc=1)
def broadcast_last_dim(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, output_num_rows,
                       output_num_cols, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_length])
    out_gm = asc2.global_tensor(output_ptr, [output_num_rows, output_num_cols])

    rows_per_block = asc2.ceildiv(output_num_rows, asc2.block_num())
    start_offset = asc2.block_idx() * rows_per_block
    row_iters = asc2.ceildiv(rows_per_block, tile_shape[0])
    column_iters = asc2.ceildiv(output_num_cols, tile_shape[1])

    for i in asc2.range(row_iters, unroll_factor=unroll_factor):
        row_start_offset = start_offset + i * tile_shape[0]
        tensor_part = asc2.copy_in(in_gm, [row_start_offset], [tile_shape[0]]).reshape(tile_shape[0], 1)
        res = tensor_part.broadcast_to(tile_shape[0], tile_shape[1])
        for j in asc2.range(column_iters, gm_barrier=True):
            asc2.copy_out(res, out_gm, [row_start_offset, j * tile_shape[1]])


def get_broadcast_axes(input_shape, output_shape):
    rank_diff = len(output_shape) - len(input_shape)
    assert rank_diff >= 0, f"Input rank={len(input_shape)} cannot exceed output rank={len(output_shape)}"
    # Add new dimensions to the left
    padded_input = [1] * rank_diff + input_shape
    broadcast_axis = []
    for i, (in_dim, out_dim) in enumerate(zip(padded_input, output_shape)):
        if in_dim != out_dim:
            assert in_dim == 1, f"Incompatible: {in_dim} cannot broadcast to {out_dim}"
            broadcast_axis.append(i)
    return broadcast_axis, padded_input


# yapf: disable
@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params", [
# PYASC_TESTS_BEGIN
    ("broadcast_test_1", 4, ([1, 5, 5], ), (torch.int32, ), ([8, 5, 5], ), (torch.int32, ), None, ([8, 5, 5], ), 11001, (11001, 1, 0, 2, 1, 2, 64, 4, 1, 1, 1, 1, 2, 2, 1, 2, 25, 25, 0, 0, 0, 0, [0, 1, 0, 0, 0], [25, 1, 0, 0, 0], [2, 25, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_2", 4, ([1, 5, 8], ), (torch.float32, ), ([4, 5, 8], ), (torch.float32, ), None, ([4, 5, 8], ), 11001, (11001, 1, 0, 2, 1, 2, 64, 4, 1, 1, 1, 1, 1, 1, 1, 1, 40, 40, 0, 0, 0, 0, [0, 1, 0, 0, 0], [40, 1, 0, 0, 0], [1, 40, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_3", 64, ([1], ), (torch.float32, ), ([4096], ), (torch.float32, ), None, ([4096], ), 11003, (11003, 1, 0, 1, 1, 2, 64, 64, 1, 1, 1, 1, 64, 64, 1, 64, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [64, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_4", 64, ([1, 1], ), (torch.float32, ), ([4096, 1], ), (torch.float32, ), None, ([4096, 1], ), 11003, (11003, 1, 0, 1, 1, 2, 64, 64, 1, 1, 1, 1, 64, 64, 1, 64, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [64, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_5", 1, ([1], ), (torch.int8, ), ([50], ), (torch.int8, ), None, ([50], ), 11001, (11001, 1, 0, 1, 1, 0, 256, 1, 1, 1, 1, 1, 50, 50, 1, 50, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [50, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_6", 63, ([1], ), (torch.float32, ), ([11932], ), (torch.float32, ), None, ([11932], ), 11003, (11003, 1, 0, 1, 1, 2, 192, 63, 1, 1, 1, 1, 192, 28, 1, 192, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [192, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_7", 1, ([1, 3, 3], ), (torch.bfloat16, ), ([8, 3, 3], ), (torch.bfloat16, ), None, ([8, 3, 3], ), 11005, (11005, 1, 0, 2, 2, 0, 128, 1, 1, 1, 1, 1, 8, 8, 1, 8, 9, 9, 0, 0, 0, 0, [0, 1, 0, 0, 0], [9, 1, 0, 0, 0], [8, 9, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_8", 14, ([1], ), (torch.bfloat16, ), ([1786], ), (torch.bfloat16, ), None, ([1786], ), 11003, (11003, 1, 0, 1, 1, 2, 128, 14, 1, 1, 1, 1, 128, 122, 1, 128, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [128, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_9", 1, ([1, 1, 1], ), (torch.bfloat16, ), ([8, 1, 1], ), (torch.bfloat16, ), None, ([8, 1, 1], ), 11001, (11001, 1, 0, 1, 1, 0, 128, 1, 1, 1, 1, 1, 8, 8, 1, 8, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [8, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_10", 1, ([1, 2, 2], ), (torch.bfloat16, ), ([8, 2, 2], ), (torch.bfloat16, ), None, ([8, 2, 2], ), 11000, (11000, 1, 0, 2, 2, 0, 128, 1, 1, 1, 1, 1, 8, 8, 1, 8, 4, 4, 0, 0, 0, 0, [0, 1, 0, 0, 0], [4, 1, 0, 0, 0], [8, 4, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_11", 29, ([1], ), (torch.bfloat16, ), ([3618], ), (torch.bfloat16, ), None, ([3618], ), 11003, (11003, 1, 0, 1, 1, 2, 128, 29, 1, 1, 1, 1, 128, 34, 1, 128, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [128, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_12", 43, ([1], ), (torch.bfloat16, ), ([5380], ), (torch.bfloat16, ), None, ([5380], ), 11003, (11003, 1, 0, 1, 1, 2, 128, 43, 1, 1, 1, 1, 128, 4, 1, 128, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [128, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_13", 64, ([2048, 1], ), (torch.float32, ), ([2048, 16], ), (torch.float32, ), None, ([2048, 16], ), 11001, (11001, 1, 0, 2, 1, 2, 512, 64, 1, 1, 1, 1, 32, 32, 1, 32, 1, 16, 1, 1, 0, 0, [1, 0, 0, 0, 0], [16, 1, 0, 0, 0], [32, 16, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_14", 64, ([1, 1], ), (torch.float32, ), ([2048, 16], ), (torch.float32, ), None, ([2048, 16], ), 11003, (11003, 1, 0, 1, 1, 2, 512, 64, 1, 1, 1, 1, 512, 512, 1, 512, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [512, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_15", 66, ([1], ), (torch.int64, ), ([14706], ), (torch.int64, ), None, ([14706], ), 11003, (11003, 1, 0, 1, 1, 2, 224, 66, 1, 1, 1, 1, 224, 146, 1, 224, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [224, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_16", 2, ([1, 5, 5], ), (torch.int32, ), ([4, 5, 5], ), (torch.int32, ), None, ([4, 5, 5], ), 11001, (11001, 1, 0, 2, 1, 2, 64, 2, 1, 1, 1, 1, 2, 2, 1, 2, 25, 25, 0, 0, 0, 0, [0, 1, 0, 0, 0], [25, 1, 0, 0, 0], [2, 25, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_17", 1, ([4, 1], ), (torch.float32, ), ([4, 8], ), (torch.float32, ), None, ([4, 8], ), 11001, (11001, 1, 0, 2, 1, 0, 64, 1, 1, 1, 1, 1, 4, 4, 1, 4, 1, 8, 1, 1, 0, 0, [1, 0, 0, 0, 0], [8, 1, 0, 0, 0], [4, 8, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_18", 8, ([1, 5, 8], ), (torch.float32, ), ([8, 5, 8], ), (torch.float32, ), None, ([8, 5, 8], ), 11001, (11001, 1, 0, 2, 1, 2, 64, 8, 1, 1, 1, 1, 1, 1, 1, 1, 40, 40, 0, 0, 0, 0, [0, 1, 0, 0, 0], [40, 1, 0, 0, 0], [1, 40, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_19", 1, ([1, 4, 4], ), (torch.bfloat16, ), ([8, 4, 4], ), (torch.bfloat16, ), None, ([8, 4, 4], ), 11005, (11005, 1, 0, 2, 2, 0, 128, 1, 1, 1, 1, 1, 8, 8, 1, 8, 16, 16, 0, 0, 0, 0, [0, 1, 0, 0, 0], [16, 1, 0, 0, 0], [8, 16, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_20", 72, ([7000, 4, 1], ), (torch.float32, ), ([7000, 4, 32], ), (torch.float32, ), None, ([7000, 4, 32], ), 11001, (11001, 1, 0, 2, 1, 2, 12480, 72, 1, 1, 1, 1, 390, 310, 1, 390, 1, 32, 1, 1, 0, 0, [1, 0, 0, 0, 0], [32, 1, 0, 0, 0], [390, 32, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_21", 69, ([1024, 1, 1], ), (torch.float32, ), ([1024, 1, 128], ), (torch.float32, ), None, ([1024, 1, 128], ), 11001, (11001, 1, 0, 2, 1, 2, 1920, 69, 1, 1, 1, 1, 15, 4, 1, 15, 1, 128, 1, 1, 0, 0, [1, 0, 0, 0, 0], [128, 1, 0, 0, 0], [15, 128, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_22", 71, ([2400, 1, 1], ), (torch.float32, ), ([2400, 1, 128], ), (torch.float32, ), None, ([2400, 1, 128], ), 11001, (11001, 1, 0, 2, 1, 2, 4352, 71, 1, 1, 1, 1, 34, 20, 1, 34, 1, 128, 1, 1, 0, 0, [1, 0, 0, 0, 0], [128, 1, 0, 0, 0], [34, 128, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_23", 64, ([1, 2048], ), (torch.bfloat16, ), ([128, 2048], ), (torch.bfloat16, ), None, ([128, 2048], ), 11001, (11001, 1, 0, 2, 1, 2, 4096, 64, 1, 1, 1, 1, 2, 2, 1, 2, 2048, 2048, 0, 0, 0, 0, [0, 1, 0, 0, 0], [2048, 1, 0, 0, 0], [2, 2048, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_24", 71, ([8192, 1], ), (torch.float32, ), ([8192, 16], ), (torch.float32, ), None, ([8192, 16], ), 11001, (11001, 1, 0, 2, 1, 2, 1856, 71, 1, 1, 1, 1, 116, 72, 1, 116, 1, 16, 1, 1, 0, 0, [1, 0, 0, 0, 0], [16, 1, 0, 0, 0], [116, 16, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_25", 69, ([1024, 1], ), (torch.float32, ), ([1024, 64], ), (torch.float32, ), None, ([1024, 64], ), 11001, (11001, 1, 0, 2, 1, 2, 960, 69, 1, 1, 1, 1, 15, 4, 1, 15, 1, 64, 1, 1, 0, 0, [1, 0, 0, 0, 0], [64, 1, 0, 0, 0], [15, 64, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_26", 72, ([2048, 8, 1], ), (torch.float32, ), ([2048, 8, 32], ), (torch.float32, ), None, ([2048, 8, 32], ), 11001, (11001, 1, 0, 2, 1, 2, 7296, 72, 1, 1, 1, 1, 228, 196, 1, 228, 1, 32, 1, 1, 0, 0, [1, 0, 0, 0, 0], [32, 1, 0, 0, 0], [228, 32, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_27", 72, ([4096, 16, 1], ), (torch.int64, ), ([4096, 16, 16], ), (torch.int64, ), None, ([4096, 16, 16], ), 11001, (11001, 1, 0, 2, 2, 2, 7296, 72, 1, 1, 1, 1, 912, 784, 1, 456, 1, 16, 1, 1, 0, 0, [1, 0, 0, 0, 0], [16, 1, 0, 0, 0], [456, 16, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_28", 71, ([2048, 1], ), (torch.float32, ), ([2048, 64], ), (torch.float32, ), None, ([2048, 64], ), 11001, (11001, 1, 0, 2, 1, 2, 1856, 71, 1, 1, 1, 1, 29, 18, 1, 29, 1, 64, 1, 1, 0, 0, [1, 0, 0, 0, 0], [64, 1, 0, 0, 0], [29, 64, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_29", 39, ([8192, 1], ), (torch.float32, ), ([8192, 60], ), (torch.float32, ), None, ([8192, 60], ), 11001, (11001, 1, 0, 2, 2, 2, 6848, 39, 1, 1, 1, 1, 214, 60, 1, 107, 1, 60, 1, 1, 0, 0, [1, 0, 0, 0, 0], [60, 1, 0, 0, 0], [107, 60, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_30", 72, ([4096, 2, 39, 1], ), (torch.int64, ), ([4096, 2, 39, 32], ), (torch.int64, ), None, ([4096, 2, 39, 32], ), 11001, (11001, 1, 0, 2, 2, 2, 7936, 72, 1, 1, 1, 1, 4464, 2544, 1, 248, 1, 32, 1, 1, 0, 0, [1, 0, 0, 0, 0], [32, 1, 0, 0, 0], [248, 32, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_31", 72, ([2400, 300, 1], ), (torch.float32, ), ([2400, 300, 128], ), (torch.float32, ), None, ([2400, 300, 128], ), 11001, (11001, 1, 0, 2, 2, 2, 15872, 72, 1, 1, 1, 1, 10044, 6876, 1, 124, 1, 128, 1, 1, 0, 0, [1, 0, 0, 0, 0], [128, 1, 0, 0, 0], [124, 128, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_32", 64, ([2500, 128, 1], ), (torch.float32, ), ([2500, 128, 12], ), (torch.float32, ), None, ([2500, 128, 12], ), 11001, (11001, 1, 0, 2, 2, 2, 13344, 64, 1, 1, 1, 1, 5004, 4748, 1, 834, 1, 12, 1, 1, 0, 0, [1, 0, 0, 0, 0], [12, 1, 0, 0, 0], [834, 12, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_33", 52, ([2400, 8, 1], ), (torch.float32, ), ([2400, 8, 128], ), (torch.float32, ), None, ([2400, 8, 128], ), 11001, (11001, 1, 0, 2, 2, 2, 15872, 52, 1, 1, 1, 1, 372, 228, 1, 124, 1, 128, 1, 1, 0, 0, [1, 0, 0, 0, 0], [128, 1, 0, 0, 0], [124, 128, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_34", 72, ([1024, 1000, 1], ), (torch.float32, ), ([1024, 1000, 80], ), (torch.float32, ), None, ([1024, 1000, 80], ), 11001, (11001, 1, 0, 2, 2, 2, 15872, 72, 1, 1, 1, 1, 14256, 11824, 1, 198, 1, 80, 1, 1, 0, 0, [1, 0, 0, 0, 0], [80, 1, 0, 0, 0], [198, 80, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_35", 72, ([1], ), (torch.bfloat16, ), ([1285783], ), (torch.bfloat16, ), None, ([1285783], ), 11003, (11003, 1, 0, 1, 1, 2, 17920, 72, 1, 1, 1, 1, 17920, 13463, 1, 17920, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [17920, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_36", 72, ([1024, 1000, 1], ), (torch.float32, ), ([1024, 1000, 64], ), (torch.float32, ), None, ([1024, 1000, 64], ), 11001, (11001, 1, 0, 2, 2, 2, 15872, 72, 1, 1, 1, 1, 14384, 2736, 1, 248, 1, 64, 1, 1, 0, 0, [1, 0, 0, 0, 0], [64, 1, 0, 0, 0], [248, 64, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_37", 72, ([1024, 8, 1], ), (torch.float32, ), ([1024, 8, 128], ), (torch.float32, ), None, ([1024, 8, 128], ), 11001, (11001, 1, 0, 2, 1, 2, 14592, 72, 1, 1, 1, 1, 114, 98, 1, 114, 1, 128, 1, 1, 0, 0, [1, 0, 0, 0, 0], [128, 1, 0, 0, 0], [114, 128, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_38", 71, ([1024, 300, 1], ), (torch.float32, ), ([1024, 300, 128], ), (torch.float32, ), None, ([1024, 300, 128], ), 11001, (11001, 1, 0, 2, 2, 2, 15872, 71, 1, 1, 1, 1, 4340, 3400, 1, 124, 1, 128, 1, 1, 0, 0, [1, 0, 0, 0, 0], [128, 1, 0, 0, 0], [124, 128, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("broadcast_test_39", 72, ([1024, 300, 1], ), (torch.float32, ), ([1024, 300, 256], ), (torch.float32, ), None, ([1024, 300, 256], ), 11001, (11001, 1, 0, 2, 2, 2, 15872, 72, 1, 1, 1, 1, 4278, 3462, 1, 62, 1, 256, 1, 1, 0, 0, [1, 0, 0, 0, 0], [256, 1, 0, 0, 0], [62, 256, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
# PYASC_TESTS_END
])
# yapf: enable
def test_broadcast(profiler, runs, kernel_type, test_name, block_num, input_shapes, input_dtypes, output_shapes,
                   output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):
    input_shape = input_shapes[0]
    output_shape = output_shapes[0]
    dtype = input_dtypes[0]
    bufferCnt = tiling_params[4]
    uLpUnit = tiling_params[15]
    uOutOffset = tiling_params[17]

    is_scalar_input = math.prod(input_shape) == 1
    tile_shape = [1, uLpUnit] if is_scalar_input else [uLpUnit, uOutOffset]
    if tiling_key in {11000, 11004}:
        unroll_factor = 1
    elif tiling_key == 11002:
        unroll_factor = 2
    else:
        unroll_factor = bufferCnt

    # Alignment for tile_shape
    ALIGNMENT_ELEMENTS = 32 // dtype.itemsize
    tile_shape = [tile_shape[0], asc2.ceildiv(tile_shape[1], ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS]

    if is_scalar_input:
        input_shape_2d = [1, 1]
        output_shape_2d = [1, math.prod(output_shape)]
    else:
        broadcast_axis, padded_input = get_broadcast_axes(input_shape, output_shape)
        if broadcast_axis[0] == 0:
            num_cols = math.prod(output_shape[1:])
            output_shape_2d = [output_shape[0], num_cols]
            num_cols = math.prod(padded_input[1:])
            input_shape_2d = [padded_input[0], num_cols]
        elif broadcast_axis[0] == len(input_shape) - 1:
            num_rows = math.prod(output_shape[:-1])
            output_shape_2d = [num_rows, output_shape[-1]]
            num_rows = math.prod(padded_input[:-1])
            input_shape_2d = [num_rows, padded_input[-1]]
        else:
            raise NotImplementedError("Broadcast for middle dimension(s) is not implemented yet")
    broadcast_axis, padded_input = get_broadcast_axes(input_shape_2d, output_shape_2d)
    if len(broadcast_axis) != 1:
        raise NotImplementedError("Broadcast for several dimensions is not implemented yet")
    axis = broadcast_axis[0]

    if is_scalar_input:
        in_tensor = torch.arange(1, dtype=dtype) + 1
        out_tensor = torch.ones(output_shape_2d, dtype=dtype)

        params = [in_tensor, out_tensor]
        if kernel_type == STATIC:
            params.extend([asc2.ConstExpr(1), asc2.ConstExpr(output_shape_2d[0]), asc2.ConstExpr(output_shape_2d[1])])
        else:
            params.extend([1, output_shape_2d[0], output_shape_2d[1]])
        params.extend([tile_shape, unroll_factor])

        with profiler.profile():
            for _ in range(runs):
                broadcast_scalar[block_num](*params)
        expected = torch.broadcast_to(in_tensor, output_shape_2d)
    else:
        if input_shape == [1048576, 1]:
            tile_shape = [tile_shape[0] // 3, tile_shape[1]]

        num_rows, num_cols = output_shape_2d
        input_length = num_cols if axis == 0 else num_rows
        in_tensor = torch.arange(input_length, dtype=dtype) + 1
        out_tensor = torch.ones(output_shape_2d, dtype=dtype)

        params = [in_tensor, out_tensor]
        if kernel_type == STATIC:
            params.extend(
                [asc2.ConstExpr(input_length),
                 asc2.ConstExpr(output_shape_2d[0]),
                 asc2.ConstExpr(output_shape_2d[1])])
        else:
            params.extend([input_length, output_shape_2d[0], output_shape_2d[1]])
        params.extend([tile_shape, unroll_factor])

        kernel_impl = broadcast_first_dim if axis == 0 else broadcast_last_dim
        with profiler.profile():
            for _ in range(runs):
                kernel_impl[block_num](*params)
        to_reshape = [1, input_length] if axis == 0 else [input_length, 1]
        reshaped_in_tensor = in_tensor.reshape(to_reshape)
        expected = torch.broadcast_to(reshaped_in_tensor, output_shape_2d)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
