# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math

from asc.experimental import asctile
import pytest
import torch

from .helpers import parametrize_is_static


@asctile.jit(reuse_alloc=1)
def gather_simple(input_ptr, index_ptr, result_ptr, row_length: asctile.ConstExpr, ub_row: asctile.ConstExpr,
                  step: asctile.ConstExpr, repeats, data_size, index_size, unroll_factor: asctile.ConstExpr):
    input_tensor = asctile.global_tensor(input_ptr, [data_size, row_length])
    index_tensor = asctile.global_tensor(index_ptr, [index_size])
    result_tensor = asctile.global_tensor(result_ptr, [index_size, row_length])

    for i in range(asctile.block_idx(), repeats, asctile.block_num(), unroll_factor=unroll_factor):
        read_count = min(step, index_size - i * step)
        index = asctile.copy_in(index_tensor, [i * step], [step])
        data = asctile.gather(input_tensor, [0], 0, index, check_bounds=False, num_indices=read_count)
        asctile.copy_out(data, result_tensor, [i * step, 0])


# yapf: disable
@parametrize_is_static()
@pytest.mark.parametrize(
    "test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params",
    [
# PYASC_TESTS_BEGIN
        pytest.param("test8", 72, ([30522, 768], [65536]), (torch.float32, torch.int32), ([65536, 768], ), (torch.float32, ), (0, ), (0, 0, 0.1), 1000000299, (72, 0, 2048, 8, [0, 0, 0, 0], 30522, 65536, 384, 910, 16, 15360), id="test8"),
        pytest.param("test10", 72, ([32000, 5120], [4, 1024]), (torch.float16, torch.int64), ([4, 1024, 5120], ), (torch.float16, ), (0, ), (0, 0, 0.1), 1000000299, (72, 0, 1024, 8, [0, 0, 0, 0], 3200, 4096, 1280, 56, 64, 15360), id="test10"),
        pytest.param("test12", 72, ([2, 768], [65536]), (torch.float32, torch.int32), ([65536, 768], ), (torch.float32, ), (0, ), (0, 0, 0.1), 1000000299, (72, 0, 2048, 8, [0, 0, 0, 0], 2, 65536, 384, 910, 16, 15360), id="test12"),
        pytest.param("test13", 72, ([32000, 2048], [5064]), (torch.bfloat16, torch.int64), ([5064, 2048], ), (torch.bfloat16, ), (0, ), (0, 0, 0.1), 1000000299, (72, 0, 1024, 8, [0, 0, 0, 0], 32000, 5064, 512, 70, 24, 15360), id="test13"),
        pytest.param("test14", 72, ([3953, 4096], [1, 2059]), (torch.float32, torch.int64), ([1, 2059, 4096], ), (torch.float32, ), (0, ), (0, 0, 0.1), 1000000299, (72, 0, 1024, 8, [0, 0, 0, 0], 3953, 2059, 2048, 28, 43, 15360), id="test14"),
        pytest.param("test15", 72, ([2059, 4096], [1, 2059]), (torch.float32, torch.int64), ([1, 2059, 4096], ), (torch.float32, ), (0, ), (0, 0, 0.1), 1000000299, (72, 0, 1024, 8, [0, 0, 0, 0], 2059, 2059, 2048, 28, 43, 15360), id="test15"),
        pytest.param("test16", 72, ([2, 1024], [65536]), (torch.float32, torch.int32), ([65536, 1024], ), (torch.float32, ), (0, ), (0, 0, 0.1), 1000000299, (72, 0, 2048, 8, [0, 0, 0, 0], 2, 65536, 512, 910, 16, 15360), id="test16"),
# PYASC_TESTS_END
    ])
# yapf: enable
def test_gather(profiler, runs, is_static, test_name, block_num, input_shapes, input_dtypes, output_shapes,
                output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):

    input_shape = input_shapes[0]
    index_shape = input_shapes[1]
    input_dtype = input_dtypes[0]
    index_dtype = input_dtypes[1]
    unroll_factor = 2
    if len(runtime_params) > 0 and runtime_params[0] > 0:
        unroll_factor = runtime_params[0]

    row_length = input_shape[-1]
    index_count = math.prod(index_shape)
    data_count = math.prod(input_shape[:-1])
    element_size = input_dtype.itemsize
    index_size = index_dtype.itemsize
    elements_in_block = 32 // element_size
    index_in_block = 32 // index_size
    ub_row = asctile.ceildiv(row_length, elements_in_block) * elements_in_block

    input = torch.randn([data_count, row_length]).to(input_dtype)
    index = torch.randint(0, data_count - 1, [index_count]).to(index_dtype)

    ub_size = 220000  # workaround to fit in UB
    step = ub_size // (ub_row * element_size + index_size)
    step = step // unroll_factor
    step = max(1, min(step, index_count // block_num))
    step = step // index_in_block * index_in_block

    repeats = asctile.ceildiv(index_count, step)

    result = torch.zeros([index_count, row_length], dtype=input_dtype)

    with profiler.profile():
        for _ in range(runs):
            if is_static:
                gather_simple[block_num](input, index, result, row_length, ub_row, step, asctile.ConstExpr(repeats),
                                         data_count, index_count, unroll_factor)
            else:
                gather_simple[block_num](input, index, result, row_length, ub_row, step, asctile.ConstExpr(repeats),
                                         asctile.ConstExpr(data_count), asctile.ConstExpr(index_count), unroll_factor)

    golden = torch.zeros([index_count, row_length], dtype=input_dtype)
    for row in range(0, index_count):
        golden[row] = input[index[row]]

    torch.testing.assert_close(result, golden)
