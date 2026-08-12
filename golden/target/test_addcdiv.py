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

from .helpers import DYNAMIC, STATIC, select_elementwise_tile


@asc2.jit(reuse_alloc=1)
def addcdiv(input_ptr: asc2.GlobalAddress, x1_ptr: asc2.GlobalAddress, x2_ptr: asc2.GlobalAddress,
            output_ptr: asc2.GlobalAddress, input_length, tile_length: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    input_gm = asc2.global_tensor(input_ptr, [input_length])
    x1_gm = asc2.global_tensor(x1_ptr, [input_length])
    x2_gm = asc2.global_tensor(x2_ptr, [input_length])
    output_gm = asc2.global_tensor(output_ptr, [input_length])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        input_t = asc2.copy_in(input_gm, [current_offset], [tile_length])
        x1_t = asc2.copy_in(x1_gm, [current_offset], [tile_length])
        x2_t = asc2.copy_in(x2_gm, [current_offset], [tile_length])
        div_t = x1_t / x2_t
        scaled_t = div_t * 0.5
        zt = input_t + scaled_t
        asc2.copy_out(zt, output_gm, [current_offset])


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("test_name, input_shape, input_dtype, tiling", [
    ("addcdiv_test_1", [11734, 16], torch.float32, select_elementwise_tile([11734, 16], 4, 4)),
    ("addcdiv_test_2", [152], torch.float32, select_elementwise_tile([152], 4, 4)),
    ("addcdiv_test_3", [152, 456], torch.float32, select_elementwise_tile([152, 456], 4, 4)),
    ("addcdiv_test_4", [1, 168], torch.float32, select_elementwise_tile([1, 168], 4, 4)),
    ("addcdiv_test_5", [7, 10], torch.float32, select_elementwise_tile([7, 10], 4, 4)),
    ("addcdiv_test_6", [8], torch.float32, select_elementwise_tile([8], 4, 4)),
    ("addcdiv_test_7", [80], torch.float32, select_elementwise_tile([80], 4, 4)),
    ("addcdiv_test_8", [98166, 16], torch.float32, select_elementwise_tile([98166, 16], 4, 4)),
    ("addcdiv_test_9", [1024], torch.float32, select_elementwise_tile([1024], 4, 4)),
    ("addcdiv_test_10", [1, 14, 1], torch.float32, select_elementwise_tile([1, 14, 1], 4, 4)),
    ("addcdiv_test_11", [1024, 152], torch.float32, select_elementwise_tile([1024, 152], 4, 4)),
    ("addcdiv_test_12", [421], torch.float32, select_elementwise_tile([421], 4, 4)),
    ("addcdiv_test_13", [256, 320], torch.float32, select_elementwise_tile([256, 320], 4, 4)),
    ("addcdiv_test_14", [8, 64], torch.float32, select_elementwise_tile([8, 64], 4, 4)),
    ("addcdiv_test_15", [1, 40], torch.float32, select_elementwise_tile([1, 40], 4, 4)),
    ("addcdiv_test_16", [64, 121], torch.float32, select_elementwise_tile([64, 121], 4, 4)),
    ("addcdiv_test_17", [48], torch.float32, select_elementwise_tile([48], 4, 4)),
    ("addcdiv_test_18", [1024, 1024], torch.float32, select_elementwise_tile([1024, 1024], 4, 4)),
    ("addcdiv_test_19", [64, 225, 1], torch.float32, select_elementwise_tile([64, 225, 1], 4, 4)),
    ("addcdiv_test_20", [16, 16, 1], torch.float32, select_elementwise_tile([16, 16, 1], 4, 4)),
    ("addcdiv_test_21", [1820039, 16], torch.float32, select_elementwise_tile([1820039, 16], 4, 4)),
    ("addcdiv_test_22", [315511, 16], torch.float32, select_elementwise_tile([315511, 16], 4, 4)),
    ("addcdiv_test_23", [98166, 128], torch.float32, select_elementwise_tile([98166, 128], 4, 4)),
    ("addcdiv_test_24", [1024, 1024], torch.float16, select_elementwise_tile([1024, 1024], 2, 4)),
    ("addcdiv_test_25", [98166, 128], torch.float16, select_elementwise_tile([98166, 128], 2, 4)),
])
def test_addcdiv(profiler, runs, kernel_type, test_name, input_shape, input_dtype, tiling):
    length, tile_length, block_num, unroll_factor = tiling

    in_tensor_input = torch.randn([length], dtype=input_dtype)
    in_tensor_x1 = torch.randn([length], dtype=input_dtype)
    # Bound the divisor away from zero: a near-zero tensor2 makes x1/x2 blow up,
    # which overflows the fp16 range and diverges from the CPU golden. |x2| >= 1
    # keeps the division well-conditioned across all dtypes.
    in_tensor_x2 = torch.randn([length], dtype=input_dtype).abs() + 1.0
    out_tensor = torch.zeros([length], dtype=input_dtype)

    params = [in_tensor_input, in_tensor_x1, in_tensor_x2, out_tensor]
    if kernel_type == STATIC:
        params.append(asc2.ConstExpr(length))
    else:
        params.append(length)
    params.extend([tile_length, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            addcdiv[block_num](*params)

    # addcdiv divides by tensor2 (values near 0 amplify error); widen tolerance
    # for the lower-precision fp16 dtype.
    tol = {torch.float32: 1e-3, torch.float16: 4e-3}[input_dtype]
    expected = torch.addcdiv(in_tensor_input, in_tensor_x1, in_tensor_x2, value=0.5)
    torch.testing.assert_close(out_tensor, expected, atol=tol, rtol=tol)
