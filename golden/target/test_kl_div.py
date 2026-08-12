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


@asc2.jit(reuse_alloc=1, vf_fusion=True)
def kl_div(input_x_ptr: asc2.GlobalAddress, input_target_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress,
           input_size, tile_length: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    loop_count = asc2.ceildiv(input_size, tile_length)
    x_gm = asc2.global_tensor(input_x_ptr, [input_size])
    target_gm = asc2.global_tensor(input_target_ptr, [input_size])
    output_gm = asc2.global_tensor(output_ptr, [input_size])
    # Epsilon values for numerical stability
    # For float16: machine epsilon ~1.18e-07
    # For float32: smallest normalized value ~1.18e-38
    epsilon = 1.18e-07 if input_x_ptr.dtype == asc2.float16 else 1.1799999457746311e-38
    acc_block = asc2.zeros([tile_length], dtype=input_x_ptr.dtype)
    for i in asc2.range(loop_count, unroll_factor=unroll_factor):
        current_offset = i * tile_length
        tail_length = input_size - tile_length
        real_offset = max(current_offset, tail_length)
        target_block = asc2.copy_in(target_gm, [current_offset], [tile_length], real_shape=[input_size - real_offset],
                                    pad_value=0)
        positive_target = asc2.maximum(target_block, 0)
        if input_x_ptr.dtype == asc2.float16:
            positive_target = positive_target.to(asc2.float32) * 1024
        mask = positive_target // (positive_target + epsilon)
        if input_x_ptr.dtype == asc2.float16:
            mask = mask.to(asc2.float16)
        log_target = asc2.log(asc2.maximum(target_block, epsilon))
        x_block = asc2.copy_in(x_gm, [current_offset], [tile_length], real_shape=[input_size - real_offset],
                               pad_value=0)
        block_result = mask * (target_block * (log_target - x_block))
        acc_block = acc_block + block_result
    final_result = asc2.reduce_sum(acc_block)
    asc2.copy_out(final_result, output_gm, [0])


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("unroll_factor, input_shape, input_dtype, tile_length", [
    (2, [1, 427], torch.float16, 427),
    (2, [427, 2], torch.float16, 854),
    (2, [3762], torch.float16, 3762),
    (2, [1000, 16], torch.float16, 5632),
    (2, [31, 11], torch.float32, 341),
    (2, [768, 16, 2], torch.float32, 3328),
    (2, [2508, 16, 8], torch.float32, 3328),
    (2, [32, 3501, 5], torch.float32, 3328),
    (2, [7, 6016, 7, 3], torch.float32, 3328),
    (2, [427, 2, 1000], torch.float32, 3328),
    (2, [8689, 1000, 32], torch.float32, 3328),
    (2, [1000, 997, 1000, 2, 1], torch.float32, 3328),
])
def test_kl_div(profiler, runs, kernel_type, unroll_factor, input_shape, input_dtype, tile_length):
    input_shape_1d = [math.prod(input_shape)]

    in_tensor_x = torch.rand(input_shape_1d, dtype=input_dtype)
    in_tensor_target = torch.rand_like(in_tensor_x)
    out_tensor = torch.empty(1, dtype=in_tensor_x.dtype)

    params = [in_tensor_x, in_tensor_target, out_tensor]
    if kernel_type == STATIC:
        params.append(asc2.ConstExpr(input_shape_1d[0]))
    else:
        params.append(input_shape_1d[0])
    params.extend([tile_length, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            kl_div[1](*params)

    expected = torch.nn.functional.kl_div(in_tensor_x, in_tensor_target, reduction='sum', log_target=False)
    torch.testing.assert_close(out_tensor[0], expected, atol=1e-3, rtol=1e-3)
