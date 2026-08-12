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
def reduce_sum_rows(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_num_rows, input_num_cols,
                    output_length, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_num_rows, input_num_cols])
    out_gm = asc2.global_tensor(output_ptr, [output_length])
    max_blocks = asc2.ceildiv(input_num_rows, tile_shape[0])
    iters = asc2.ceildiv(input_num_cols, tile_shape[1])

    for i in asc2.range(asc2.block_idx(), max_blocks, asc2.block_num(), unroll_factor=unroll_factor):
        cache = asc2.zeros([tile_shape[0]], dtype=in_gm.dtype)
        for j in asc2.range(iters, gm_barrier=True):
            block = asc2.copy_in(in_gm, [i * tile_shape[0], j * tile_shape[1]], tile_shape, pad_value=0)
            block = asc2.reduce_sum(block, 1)
            cache = cache + block
        asc2.copy_out(cache, out_gm, [i * tile_shape[0]])


@asc2.jit(reuse_alloc=1)
def reduce_sum_cols(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_num_rows, input_num_cols,
                    output_length, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_num_rows, input_num_cols])
    out_gm = asc2.global_tensor(output_ptr, [output_length])

    max_blocks = asc2.ceildiv(input_num_rows, tile_shape[1])
    iters = asc2.ceildiv(input_num_cols, tile_shape[0])

    for j in asc2.range(asc2.block_idx(), max_blocks, asc2.block_num(), unroll_factor=unroll_factor):
        cache = asc2.zeros([tile_shape[1]], dtype=in_gm.dtype)
        for i in asc2.range(iters, gm_barrier=True):
            block = asc2.copy_in(in_gm, [i * tile_shape[0], j * tile_shape[1]], tile_shape, pad_value=0)
            block = asc2.reduce_sum(block, 0)
            cache = cache + block
        asc2.copy_out(cache, out_gm, [j * tile_shape[1]])


@asc2.jit(reuse_alloc=2)
def reduce_none(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_num_rows, input_num_cols,
                output_length, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    total_elements = input_num_rows * input_num_cols
    in_gm = asc2.global_tensor(input_ptr, [total_elements])
    out_gm = asc2.global_tensor(output_ptr, [total_elements])
    block_portion = tile_shape[0] * tile_shape[1]
    total_repeats = asc2.ceildiv(total_elements, block_portion)

    for i in asc2.range(asc2.block_idx(), total_repeats, asc2.block_num(), unroll_factor=unroll_factor):
        data = asc2.copy_in(in_gm, [i * block_portion], [block_portion])
        asc2.copy_out(data, out_gm, [i * block_portion])


# yapf: disable
@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params", [
# PYASC_TESTS_BEGIN
    ("reduce_sum_test_1", 1, ([61], ), (torch.int32, ), ([1], ), (torch.int32, ), (True, ), (2, [0], True), 5143, (1, 1, 1, 1, 1, 1, 1, 1, 58880, 768, 72, 0, 0.016393441706895828, [1, 61, 0, 0, 0, 0, 0, 0, 0], [61, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 61, 0, 0, 0, 0, 0, 0, 0], [61, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_2", 1, ([8], ), (torch.int32, ), ([1], ), (torch.int32, ), (True, ), (2, [0], True), 5143, (1, 1, 1, 1, 1, 1, 1, 1, 55808, 6912, 72, 0, 0.125, [1, 8, 0, 0, 0, 0, 0, 0, 0], [8, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 8, 0, 0, 0, 0, 0, 0, 0], [8, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_3", 1, ([32, 3], ), (torch.float32, ), ([32, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 1, 32, 1, 1, 1, 1, 32, 55808, 6912, 72, 1, 0.3333333432674408, [32, 3, 0, 0, 0, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [32, 3, 0, 0, 0, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_4", 69, ([7000, 10, 1], ), (torch.float32, ), ([7000, 10], ), (torch.float32, ), (False, ), (2, [2], False), 5321, (1, 69, 1024, 1, 1, 1, 1, 70000, 39424, 39424, 72, 0, 1.0, [70000, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [70000, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_5", 1, ([16, 2, 160], ), (torch.float32, ), ([16, 2, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (1, 1, 32, 1, 1, 1, 1, 32, 59136, 512, 72, 0, 0.0062500000931322575, [32, 160, 0, 0, 0, 0, 0, 0, 0], [160, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [32, 160, 0, 0, 0, 0, 0, 0, 0], [160, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_6", 1, ([2], ), (torch.int32, ), ([1], ), (torch.int32, ), (True, ), (2, [0], True), 5143, (1, 1, 1, 1, 1, 1, 1, 1, 55808, 6912, 72, 1, 0.5, [1, 2, 0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_7", 1, ([88], ), (torch.float32, ), ([1], ), (torch.float32, ), (True, ), (2, [0], True), 5143, (1, 1, 1, 1, 1, 1, 1, 1, 59136, 512, 72, 0, 0.011363636702299118, [1, 88, 0, 0, 0, 0, 0, 0, 0], [88, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 88, 0, 0, 0, 0, 0, 0, 0], [88, 1, 0, 0, 0, 0, 0, 0, 0])),
    # TODO: UB overflow ("reduce_sum_test_8", 2, ([16, 1360], ), (torch.float32, ), ([16, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 2, 10, 1, 1, 1, 1, 16, 59136, 512, 72, 0, 0.000735294132027775, [16, 1360, 0, 0, 0, 0, 0, 0, 0], [1360, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [16, 1360, 0, 0, 0, 0, 0, 0, 0], [1360, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_9", 1, ([16, 64], ), (torch.float32, ), ([16, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 1, 16, 1, 1, 1, 1, 16, 58880, 768, 72, 0, 0.015625, [16, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [16, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_10", 1, ([16, 256], ), (torch.float32, ), ([16, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 1, 16, 1, 1, 1, 1, 16, 59136, 512, 72, 0, 0.00390625, [16, 256, 0, 0, 0, 0, 0, 0, 0], [256, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [16, 256, 0, 0, 0, 0, 0, 0, 0], [256, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_11", 1, ([32, 1, 6], ), (torch.float32, ), ([32, 1], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (1, 1, 32, 1, 1, 1, 1, 32, 55808, 6912, 72, 1, 0.1666666716337204, [32, 6, 0, 0, 0, 0, 0, 0, 0], [6, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [32, 6, 0, 0, 0, 0, 0, 0, 0], [6, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_12", 5, ([15, 2048], ), (torch.int64, ), ([15], ), (torch.int64, ), (False, ), (2, [1], False), 5143, (1, 5, 3, 1, 1, 1, 1, 15, 59136, 512, 72, 0, 0.00048828125, [15, 2048, 0, 0, 0, 0, 0, 0, 0], [2048, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [15, 2048, 0, 0, 0, 0, 0, 0, 0], [2048, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_13", 1, ([100, 1], ), (torch.float32, ), ([1, 1], ), (torch.float32, ), (True, ), (2, [0], True), 5143, (1, 1, 1, 1, 1, 1, 1, 1, 59136, 512, 72, 0, 0.009999999776482582, [1, 100, 0, 0, 0, 0, 0, 0, 0], [100, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 100, 0, 0, 0, 0, 0, 0, 0], [100, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_14", 1, ([100, 56], ), (torch.float32, ), ([1, 56], ), (torch.float32, ), (True, ), (2, [0], True), 5161, (1, 1, 1, 1, 1, 1, 1, 56, 59136, 512, 72, 0, 0.009999999776482582, [1, 100, 56, 0, 0, 0, 0, 0, 0], [5600, 56, 1, 0, 0, 0, 0, 0, 0], [56, 56, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 100, 56, 0, 0, 0, 0, 0, 0], [5600, 56, 1, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_15", 1, ([16, 128], ), (torch.float32, ), ([16, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 1, 16, 1, 1, 1, 1, 16, 59136, 512, 72, 0, 0.0078125, [16, 128, 0, 0, 0, 0, 0, 0, 0], [128, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [16, 128, 0, 0, 0, 0, 0, 0, 0], [128, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_16", 1, ([16, 320], ), (torch.float32, ), ([16, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 1, 16, 1, 1, 1, 1, 16, 59136, 512, 72, 0, 0.0031250000465661287, [16, 320, 0, 0, 0, 0, 0, 0, 0], [320, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [16, 320, 0, 0, 0, 0, 0, 0, 0], [320, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_17", 1, ([4, 5, 8], ), (torch.float32, ), ([4, 5, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (1, 1, 20, 1, 1, 1, 1, 20, 55808, 6912, 72, 0, 0.125, [20, 8, 0, 0, 0, 0, 0, 0, 0], [8, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [20, 8, 0, 0, 0, 0, 0, 0, 0], [8, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_18", 1, ([140], ), (torch.float32, ), ([1], ), (torch.float32, ), (True, ), (2, [0], True), 5143, (1, 1, 1, 1, 1, 1, 1, 1, 59136, 512, 72, 0, 0.0071428571827709675, [1, 140, 0, 0, 0, 0, 0, 0, 0], [140, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 140, 0, 0, 0, 0, 0, 0, 0], [140, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_19", 1, ([4, 8], ), (torch.float32, ), ([4, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 1, 4, 1, 1, 1, 1, 4, 55808, 6912, 72, 0, 0.125, [4, 8, 0, 0, 0, 0, 0, 0, 0], [8, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [4, 8, 0, 0, 0, 0, 0, 0, 0], [8, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_20", 16, ([1024, 50], ), (torch.float32, ), ([1024], ), (torch.float32, ), (False, ), (2, [1], False), 5143, (1, 16, 64, 1, 1, 1, 1, 1024, 58880, 1024, 72, 0, 0.019999999552965164, [1024, 50, 0, 0, 0, 0, 0, 0, 0], [50, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1024, 50, 0, 0, 0, 0, 0, 0, 0], [50, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_21", 69, ([7000, 4, 32], ), (torch.float32, ), ([7000, 4], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (1, 69, 411, 1, 1, 1, 1, 28000, 58368, 1792, 72, 0, 0.03125, [28000, 32, 0, 0, 0, 0, 0, 0, 0], [32, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [28000, 32, 0, 0, 0, 0, 0, 0, 0], [32, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_22", 69, ([4096, 50, 16], ), (torch.float32, ), ([4096, 50], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (4, 274, 750, 1, 1, 1, 1, 204800, 57600, 3584, 72, 0, 0.0625, [204800, 16, 0, 0, 0, 0, 0, 0, 0], [16, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [204800, 16, 0, 0, 0, 0, 0, 0, 0], [16, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_23", 69, ([7000, 8, 64], ), (torch.float32, ), ([7000, 8], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (5, 342, 164, 1, 1, 1, 1, 56000, 58880, 768, 72, 0, 0.015625, [56000, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [56000, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_24", 69, ([7000, 56], ), (torch.float32, ), ([7000], ), (torch.float32, ), (False, ), (2, [1], False), 5143, (1, 69, 102, 1, 1, 1, 1, 7000, 58880, 1024, 72, 0, 0.01785714365541935, [7000, 56, 0, 0, 0, 0, 0, 0, 0], [56, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [7000, 56, 0, 0, 0, 0, 0, 0, 0], [56, 1, 0, 0, 0, 0, 0, 0, 0])),
    # TODO: UB overflow ("reduce_sum_test_25", 70, ([11932, 256], ), (torch.float32, ), ([11932, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (3, 210, 57, 1, 1, 1, 1, 11932, 59136, 512, 72, 0, 0.00390625, [11932, 256, 0, 0, 0, 0, 0, 0, 0], [256, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [11932, 256, 0, 0, 0, 0, 0, 0, 0], [256, 1, 0, 0, 0, 0, 0, 0, 0])),
    # TODO: UB overflow ("reduce_sum_test_26", 72, ([1024, 4, 256], ), (torch.float32, ), ([1024, 4], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (1, 72, 57, 1, 1, 1, 1, 4096, 59136, 512, 72, 0, 0.00390625, [4096, 256, 0, 0, 0, 0, 0, 0, 0], [256, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [4096, 256, 0, 0, 0, 0, 0, 0, 0], [256, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_27", 69, ([2400, 8, 128], ), (torch.float32, ), ([2400, 8], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (3, 207, 93, 1, 1, 1, 1, 19200, 59136, 512, 72, 0, 0.0078125, [19200, 128, 0, 0, 0, 0, 0, 0, 0], [128, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [19200, 128, 0, 0, 0, 0, 0, 0, 0], [128, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_28", 16, ([16, 64, 160], ), (torch.float32, ), ([16, 64, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (1, 16, 64, 1, 1, 1, 1, 1024, 59136, 512, 72, 0, 0.0062500000931322575, [1024, 160, 0, 0, 0, 0, 0, 0, 0], [160, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1024, 160, 0, 0, 0, 0, 0, 0, 0], [160, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_29", 16, ([1024, 1, 64], ), (torch.float32, ), ([1024, 1, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (1, 16, 64, 1, 1, 1, 1, 1024, 58880, 768, 72, 0, 0.015625, [1024, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1024, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0])),
    # TODO: UB overflow ("reduce_sum_test_30", 21, ([16, 24, 768], ), (torch.float32, ), ([16, 24, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (1, 21, 19, 1, 1, 1, 1, 384, 59136, 512, 72, 0, 0.0013020833721384406, [384, 768, 0, 0, 0, 0, 0, 0, 0], [768, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [384, 768, 0, 0, 0, 0, 0, 0, 0], [768, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_31", 69, ([8192, 16], ), (torch.float32, ), ([8192, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 69, 120, 1, 1, 1, 1, 8192, 57600, 3584, 72, 0, 0.0625, [8192, 16, 0, 0, 0, 0, 0, 0, 0], [16, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [8192, 16, 0, 0, 0, 0, 0, 0, 0], [16, 1, 0, 0, 0, 0, 0, 0, 0])),
    # TODO: UB overflow ("reduce_sum_test_32", 7, ([128, 768], ), (torch.float32, ), ([128, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 7, 19, 1, 1, 1, 1, 128, 59136, 512, 72, 0, 0.0013020833721384406, [128, 768, 0, 0, 0, 0, 0, 0, 0], [768, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [128, 768, 0, 0, 0, 0, 0, 0, 0], [768, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_33", 32, ([2048, 32], ), (torch.float32, ), ([2048, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 32, 64, 1, 1, 1, 1, 2048, 58368, 1792, 72, 0, 0.03125, [2048, 32, 0, 0, 0, 0, 0, 0, 0], [32, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [2048, 32, 0, 0, 0, 0, 0, 0, 0], [32, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_34", 69, ([2048, 8, 32], ), (torch.float32, ), ([2048, 8, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (1, 69, 240, 1, 1, 1, 1, 16384, 58368, 1792, 72, 0, 0.03125, [16384, 32, 0, 0, 0, 0, 0, 0, 0], [32, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [16384, 32, 0, 0, 0, 0, 0, 0, 0], [32, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_35", 42, ([128, 13, 768], ), (torch.float32, ), ([128, 13, 1], ), (torch.float32, ), (True, ), (2, [2], True), 136215, (2, 84, 20, 5, 5, 184, 1, 1664, 59136, 512, 72, 0, 0.0013020833721384406, [1664, 768, 0, 0, 0, 0, 0, 0, 0], [768, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1664, 768, 0, 0, 0, 0, 0, 0, 0], [768, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_36", 69, ([8192, 60], ), (torch.float32, ), ([8192, 1], ), (torch.float32, ), (True, ), (2, [1], True), 5143, (1, 69, 120, 1, 1, 1, 1, 8192, 58880, 768, 72, 0, 0.01666666753590107, [8192, 60, 0, 0, 0, 0, 0, 0, 0], [60, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [8192, 60, 0, 0, 0, 0, 0, 0, 0], [60, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_37", 40, ([2500, 1, 120], ), (torch.float32, ), ([2500, 1], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (1, 40, 64, 1, 1, 1, 1, 2500, 59136, 512, 72, 0, 0.008333333767950535, [2500, 120, 0, 0, 0, 0, 0, 0, 0], [120, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [2500, 120, 0, 0, 0, 0, 0, 0, 0], [120, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_38", 40, ([2500, 1, 88], ), (torch.float32, ), ([2500, 1], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (1, 40, 64, 1, 1, 1, 1, 2500, 59136, 512, 72, 0, 0.011363636702299118, [2500, 88, 0, 0, 0, 0, 0, 0, 0], [88, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [2500, 88, 0, 0, 0, 0, 0, 0, 0], [88, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_39", 69, ([1024, 16, 64], ), (torch.float32, ), ([1024, 16, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (2, 137, 120, 1, 1, 1, 1, 16384, 58880, 768, 72, 0, 0.015625, [16384, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [16384, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_40", 69, ([1024, 30, 64], ), (torch.float32, ), ([1024, 30, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (3, 207, 149, 1, 1, 1, 1, 30720, 58880, 768, 72, 0, 0.015625, [30720, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [30720, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_41", 70, ([2500, 100, 88], ), (torch.float32, ), ([2500, 100], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (28, 1954, 128, 1, 1, 1, 1, 250000, 59136, 512, 72, 0, 0.011363636702299118, [250000, 88, 0, 0, 0, 0, 0, 0, 0], [88, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [250000, 88, 0, 0, 0, 0, 0, 0, 0], [88, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_42", 71, ([2500, 150, 140], ), (torch.float32, ), ([2500, 150], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (52, 3677, 102, 1, 1, 1, 1, 375000, 59136, 512, 72, 0, 0.0071428571827709675, [375000, 140, 0, 0, 0, 0, 0, 0, 0], [140, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [375000, 140, 0, 0, 0, 0, 0, 0, 0], [140, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_43", 69, ([7000, 8, 72], ), (torch.float32, ), ([7000, 8], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (5, 342, 164, 1, 1, 1, 1, 56000, 58880, 768, 72, 0, 0.013888888992369175, [56000, 72, 0, 0, 0, 0, 0, 0, 0], [72, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [56000, 72, 0, 0, 0, 0, 0, 0, 0], [72, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_44", 72, ([1024, 8, 128], ), (torch.float32, ), ([1024, 8], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (1, 72, 115, 1, 1, 1, 1, 8192, 59136, 512, 72, 0, 0.0078125, [8192, 128, 0, 0, 0, 0, 0, 0, 0], [128, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [8192, 128, 0, 0, 0, 0, 0, 0, 0], [128, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_45", 72, ([16352, 39, 48], ), (torch.float32, ), ([16352, 39, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (35, 2492, 256, 1, 1, 1, 1, 637728, 58880, 1024, 72, 0, 0.02083333395421505, [637728, 48, 0, 0, 0, 0, 0, 0, 0], [48, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [637728, 48, 0, 0, 0, 0, 0, 0, 0], [48, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_46", 69, ([1024, 50, 64], ), (torch.float32, ), ([1024, 50, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (4, 274, 187, 1, 1, 1, 1, 51200, 58880, 768, 72, 0, 0.015625, [51200, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [51200, 64, 0, 0, 0, 0, 0, 0, 0], [64, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_47", 71, ([16352, 31, 48], ), (torch.float32, ), ([16352, 31, 1], ), (torch.float32, ), (True, ), (2, [2], True), 5143, (28, 1981, 256, 1, 1, 1, 1, 506912, 58880, 1024, 72, 0, 0.02083333395421505, [506912, 48, 0, 0, 0, 0, 0, 0, 0], [48, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [506912, 48, 0, 0, 0, 0, 0, 0, 0], [48, 1, 0, 0, 0, 0, 0, 0, 0])),
    # TODO: UB overflow ("reduce_sum_test_48", 72, ([2400, 300, 256], ), (torch.float32, ), ([2400, 300], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (176, 12632, 57, 1, 1, 1, 1, 720000, 59136, 512, 72, 0, 0.00390625, [720000, 256, 0, 0, 0, 0, 0, 0, 0], [256, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [720000, 256, 0, 0, 0, 0, 0, 0, 0], [256, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_49", 72, ([2500, 150, 148], ), (torch.float32, ), ([2500, 150], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (54, 3866, 97, 1, 1, 1, 1, 375000, 59136, 512, 72, 0, 0.006756756920367479, [375000, 148, 0, 0, 0, 0, 0, 0, 0], [148, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [375000, 148, 0, 0, 0, 0, 0, 0, 0], [148, 1, 0, 0, 0, 0, 0, 0, 0])),
    ("reduce_sum_test_50", 72, ([1024, 1000, 80], ), (torch.float32, ), ([1024, 1000], ), (torch.float32, ), (False, ), (2, [2], False), 5143, (112, 8000, 128, 1, 1, 1, 1, 1024000, 59136, 512, 72, 0, 0.012500000186264515, [1024000, 80, 0, 0, 0, 0, 0, 0, 0], [80, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1024000, 80, 0, 0, 0, 0, 0, 0, 0], [80, 1, 0, 0, 0, 0, 0, 0, 0])),
# PYASC_TESTS_END
])
# yapf: enable
def test_reduce_sum(profiler, runs, kernel_type, test_name, block_num, input_shapes, input_dtypes, output_shapes,
                    output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):
    input_shape = input_shapes[0]
    output_shape = output_shapes[0]
    dtype = input_dtypes[0]
    ub_factor_a = tiling_params[2]
    ub_factor_r = tiling_params[5]
    unroll_factor = runtime_params[0]
    axis = runtime_params[1][0]

    keep_dims = (len(input_shape) == len(output_shape))
    input_shape_2d = input_shape

    if len(input_shape) == 1:
        axis = 1
        input_shape_2d = [1, input_shape[0]]
    elif axis == 0:
        num_cols = math.prod(input_shape[1:])
        input_shape_2d = [input_shape[0], num_cols]
    elif axis == len(input_shape) - 1:
        axis = 1
        num_rows = math.prod(input_shape[:-1])
        input_shape_2d = [num_rows, input_shape[-1]]
    else:
        raise NotImplementedError("ReduceSum for middle dimension(s) is not implemented yet")

    length_a = input_shape_2d[1 - axis] if ub_factor_a == 1 else ub_factor_a
    length_r = input_shape_2d[axis] if ub_factor_r == 1 else ub_factor_r
    tile_shape = [length_a, length_r] if axis == 1 else [length_r, length_a]

    # Alignment for tile_shape
    if input_shape_2d[axis] == 1:
        kernel_impl = reduce_none
    else:
        ALIGNMENT_ELEMENTS = 32 // dtype.itemsize
        tile_shape = tile_shape[0], asc2.ceildiv(tile_shape[1], ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS
        if axis == 1:
            tile_shape = asc2.ceildiv(tile_shape[0], ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS, tile_shape[1]
        kernel_impl = reduce_sum_rows if axis == 1 else reduce_sum_cols

    num_rows, num_cols = input_shape_2d
    output_shape_1d = [num_rows] if axis == 1 else [num_cols]

    if dtype.is_floating_point:
        in_tensor = torch.randn(input_shape_2d, dtype=dtype)
    else:
        in_tensor = torch.randint(-10, 10, input_shape_2d, dtype=dtype)
    out_tensor = torch.zeros(output_shape_1d, dtype=dtype)

    params = [in_tensor, out_tensor]
    if kernel_type == STATIC:
        params.extend(
            [asc2.ConstExpr(input_shape_2d[0]),
             asc2.ConstExpr(input_shape_2d[1]),
             asc2.ConstExpr(output_shape_1d[0])])
    else:
        params.extend([input_shape_2d[0], input_shape_2d[1], output_shape_1d[0]])
    params.extend([tile_shape, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            kernel_impl[block_num](*params)
    if keep_dims:
        target_shape = [output_shape_1d[0], 1] if axis == 1 else [1, output_shape_1d[0]]
        out_tensor = out_tensor.reshape(target_shape)

    expected = torch.sum(in_tensor, axis, keepdim=keep_dims, dtype=dtype)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
