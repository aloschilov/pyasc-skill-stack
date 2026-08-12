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
def reciprocal(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, tile_length: asc2.ConstExpr,
               unroll_factor: asc2.ConstExpr):
    x = asc2.global_tensor(input_ptr, [input_length])
    z = asc2.global_tensor(output_ptr, [input_length])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        xt = asc2.copy_in(x, [current_offset], [tile_length])
        zt = asc2.div(1.0, xt)
        asc2.copy_out(zt, z, [current_offset])


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("test_name, input_shape, input_dtype, tiling", [
    ("reciprocal_test_1", [1024], torch.float32, select_elementwise_tile([1024], 4, 2)),
    ("reciprocal_test_2", [2400], torch.float32, select_elementwise_tile([2400], 4, 2)),
    ("reciprocal_test_3", [16, 5, 1, 64], torch.float32, select_elementwise_tile([16, 5, 1, 64], 4, 2)),
    ("reciprocal_test_4", [16, 256], torch.float32, select_elementwise_tile([16, 256], 4, 2)),
    ("reciprocal_test_5", [16, 320], torch.float32, select_elementwise_tile([16, 320], 4, 2)),
    ("reciprocal_test_6", [16, 24, 768], torch.float32, select_elementwise_tile([16, 24, 768], 4, 2)),
    ("reciprocal_test_7", [128, 1, 2304], torch.float32, select_elementwise_tile([128, 1, 2304], 4, 2)),
    ("reciprocal_test_8", [2500], torch.float32, select_elementwise_tile([2500], 4, 2)),
    ("reciprocal_test_9", [1200], torch.float32, select_elementwise_tile([1200], 4, 2)),
    ("reciprocal_test_10", [2048], torch.float32, select_elementwise_tile([2048], 4, 2)),
    ("reciprocal_test_11", [1500], torch.float32, select_elementwise_tile([1500], 4, 2)),
    ("reciprocal_test_12", [1024, 1, 20], torch.float32, select_elementwise_tile([1024, 1, 20], 4, 2)),
    ("reciprocal_test_13", [1024, 1, 50], torch.float32, select_elementwise_tile([1024, 1, 50], 4, 2)),
    ("reciprocal_test_14", [1024, 1, 1000], torch.float32, select_elementwise_tile([1024, 1, 1000], 4, 2)),
    ("reciprocal_test_15", [256, 1], torch.float32, select_elementwise_tile([256, 1], 4, 2)),
    ("reciprocal_test_16", [100, 14, 10], torch.float32, select_elementwise_tile([100, 14, 10], 4, 2)),
    ("reciprocal_test_17", [2048, 1], torch.float32, select_elementwise_tile([2048, 1], 4, 2)),
    ("reciprocal_test_18", [1024, 6144], torch.float32, select_elementwise_tile([1024, 6144], 4, 2)),
    ("reciprocal_test_19", [8192, 1024], torch.float32, select_elementwise_tile([8192, 1024], 4, 2)),
    ("reciprocal_test_20", [2048, 8192], torch.float32, select_elementwise_tile([2048, 8192], 4, 2)),
    ("reciprocal_test_21", [128, 2, 512], torch.float16, select_elementwise_tile([128, 2, 512], 2, 2)),
    ("reciprocal_test_22", [1024, 6144], torch.float16, select_elementwise_tile([1024, 6144], 2, 2)),
])
def test_reciprocal(profiler, runs, kernel_type, test_name, input_shape, input_dtype, tiling):
    length, tile_length, block_num, unroll_factor = tiling

    # For low-precision dtypes bound |x| away from zero: a near-zero x makes 1/x
    # blow past the fp16 range (inf), which diverges from the CPU golden. fp32 has
    # the range to keep randn well-defined.
    if input_dtype == torch.float32:
        in_tensor = torch.randn([length], dtype=input_dtype)
    else:
        in_tensor = torch.randn([length], dtype=input_dtype).abs() + 1.0
    out_tensor = torch.zeros([length], dtype=input_dtype)

    params = [in_tensor, out_tensor]
    if kernel_type == STATIC:
        params.append(asc2.ConstExpr(length))
    else:
        params.append(length)
    params.extend([tile_length, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            reciprocal[block_num](*params)

    # Low-precision dtypes carry more rounding error; reciprocal (1/x) amplifies
    # it further, so widen the tolerance for fp16.
    tol = {torch.float32: 1e-3, torch.float16: 4e-3}[input_dtype]
    expected = torch.reciprocal(in_tensor)
    torch.testing.assert_close(out_tensor, expected, atol=tol, rtol=tol)
