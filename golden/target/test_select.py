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
def select(cond_ptr: asc2.GlobalAddress, input_x_ptr: asc2.GlobalAddress, input_y_ptr: asc2.GlobalAddress,
           output_ptr: asc2.GlobalAddress, input_length, tile_length: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    c = asc2.global_tensor(cond_ptr, [input_length])
    x = asc2.global_tensor(input_x_ptr, [input_length])
    y = asc2.global_tensor(input_y_ptr, [input_length])
    z = asc2.global_tensor(output_ptr, [input_length])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        ct = asc2.copy_in(c, [current_offset], [tile_length])
        xt = asc2.copy_in(x, [current_offset], [tile_length])
        yt = asc2.copy_in(y, [current_offset], [tile_length])
        zt = asc2.where(ct != 0, xt, yt)
        asc2.copy_out(zt, z, [current_offset])


# yapf: disable
@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params", [
# PYASC_TESTS_BEGIN
    ("select_test_1", 40, ([2048, 1, 30], [2048, 1, 30], [2048, 1, 30]), (torch.int8, torch.float32, torch.float32), ([2048, 1, 30], ), (torch.float32, ), None, (2, ), 8, (61440, 768, 40, 0)),
    ("select_test_2", 2, ([2048], [2048], [2048]), (torch.int8, torch.float32, torch.float32), ([2048], ), (torch.float32, ), None, (2, ), 8, (2048, 512, 2, 0)),
    ("select_test_3", 5, ([4608], [4608], [4608]), (torch.int8, torch.float32, torch.float32), ([4608], ), (torch.float32, ), None, (2, ), 8, (4608, 512, 5, 0)),
    ("select_test_4", 36, ([3072, 12], [3072, 12], [3072, 12]), (torch.int8, torch.int32, torch.int32), ([3072, 12], ), (torch.int32, ), None, (2, ), 8, (36864, 512, 36, 0)),
    ("select_test_5", 32, ([1024, 4, 8], [1024, 4, 8], [1024, 4, 8]), (torch.int8, torch.float32, torch.float32), ([1024, 4, 8], ), (torch.float32, ), None, (2, ), 8, (32768, 512, 32, 0)),
    ("select_test_6", 10, ([9600], [9600], [9600]), (torch.int8, torch.int32, torch.int32), ([9600], ), (torch.int32, ), None, (2, ), 8, (9600, 512, 10, 0)),
    ("select_test_7", 3, ([3000], [3000], [3000]), (torch.int8, torch.float32, torch.float32), ([3000], ), (torch.float32, ), None, (2, ), 8, (3000, 512, 3, 0)),
    ("select_test_8", 36, ([4608, 12], [4608, 12], [4608, 12]), (torch.int8, torch.int32, torch.int32), ([4608, 12], ), (torch.int32, ), None, (2, ), 8, (55296, 768, 36, 0)),
    ("select_test_9", 5, ([4800], [4800], [4800]), (torch.int8, torch.float32, torch.float32), ([4800], ), (torch.float32, ), None, (2, ), 8, (4800, 512, 5, 0)),
    ("select_test_10", 3, ([3072], [3072, 1], [3072, 1]), (torch.int8, torch.bfloat16, torch.bfloat16), ([3072, 1], ), (torch.bfloat16, ), None, (2, ), 8, (3072, 512, 3, 0)),
    ("select_test_11", 16, ([128, 1, 128], [128, 1, 128], [128, 1, 128]), (torch.int8, torch.float32, torch.float32), ([128, 1, 128], ), (torch.float32, ), None, (2, ), 8, (16384, 512, 16, 0)),
    ("select_test_12", 7, ([7000, 1], [7000, 1], [7000, 1]), (torch.int8, torch.int32, torch.int32), ([7000, 1], ), (torch.int32, ), None, (2, ), 8, (7000, 512, 7, 0)),
    ("select_test_13", 1, ([16, 64], [16, 64], [16, 64]), (torch.int8, torch.float32, torch.float32), ([16, 64], ), (torch.float32, ), None, (2, ), 8, (1024, 512, 1, 0)),
    ("select_test_14", 2, ([16, 128], [16, 128], [16, 128]), (torch.int8, torch.float32, torch.float32), ([16, 128], ), (torch.float32, ), None, (2, ), 8, (2048, 512, 2, 0)),
    ("select_test_15", 2, ([1500], [1500], [1500]), (torch.int8, torch.float32, torch.float32), ([1500], ), (torch.float32, ), None, (2, ), 8, (1500, 512, 2, 0)),
    ("select_test_16", 40, ([1024, 50], [1024, 50], [1024, 50]), (torch.int8, torch.float32, torch.float32), ([1024, 50], ), (torch.float32, ), None, (2, ), 8, (51200, 640, 40, 0)),
    ("select_test_17", 25, ([256, 1, 100], [256, 1, 100], [256, 1, 100]), (torch.int8, torch.float32, torch.float32), ([256, 1, 100], ), (torch.float32, ), None, (2, ), 8, (25600, 512, 25, 0)),
    ("select_test_18", 2, ([1200], [1200, 1], [1200, 1]), (torch.int8, torch.float32, torch.float32), ([1200, 1], ), (torch.float32, ), None, (2, ), 8, (1200, 512, 2, 0)),
    ("select_test_19", 69, ([4800, 1, 300], [4800, 1, 300], [4800, 1, 300]), (torch.int8, torch.float32, torch.float32), ([4800, 1, 300], ), (torch.float32, ), None, (2, ), 8, (1440000, 7040, 69, 0)),
    ("select_test_20", 43, ([128, 512], [128, 512], [128, 512]), (torch.int8, torch.float16, torch.float16), ([128, 512], ), (torch.float16, ), None, (2, ), 8, (65536, 768, 43, 0)),
    ("select_test_21", 40, ([7000, 1, 10], [7000, 1, 10], [7000, 1, 10]), (torch.int8, torch.float32, torch.float32), ([7000, 1, 10], ), (torch.float32, ), None, (2, ), 8, (70000, 896, 40, 0)),
    ("select_test_22", 59, ([128, 100, 64], [128, 100, 64], [128, 100, 64]), (torch.int8, torch.float32, torch.float32), ([128, 100, 64], ), (torch.float32, ), None, (2, ), 8, (819200, 7040, 59, 0)),
    ("select_test_23", 40, ([2048, 1, 60], [2048, 1, 60], [2048, 1, 60]), (torch.int8, torch.bfloat16, torch.bfloat16), ([2048, 1, 60], ), (torch.bfloat16, ), None, (2, ), 8, (122880, 1536, 40, 0)),
    ("select_test_24", 38, ([1024, 4, 40], [1024, 4, 40], [1024, 4, 40]), (torch.int8, torch.float32, torch.float32), ([1024, 4, 40], ), (torch.float32, ), None, (2, ), 8, (163840, 2176, 38, 0)),
    ("select_test_25", 59, ([1024, 4, 300], [1024, 4, 300], [1024, 4, 300]), (torch.int8, torch.float32, torch.float32), ([1024, 4, 300], ), (torch.float32, ), None, (2, ), 8, (1228800, 7040, 59, 0)),
    ("select_test_26", 38, ([2400, 4, 8], [2400, 4, 8], [2400, 4, 8]), (torch.int8, torch.float32, torch.float32), ([2400, 4, 8], ), (torch.float32, ), None, (2, ), 8, (76800, 1024, 38, 0)),
    ("select_test_27", 37, ([1024, 100], [1024, 100], [1024, 100]), (torch.int8, torch.float32, torch.float32), ([1024, 100], ), (torch.float32, ), None, (2, ), 8, (102400, 1408, 37, 0)),
    ("select_test_28", 37, ([2400, 4, 40], [2400, 4, 40], [2400, 4, 40]), (torch.int8, torch.float32, torch.float32), ([2400, 4, 40], ), (torch.float32, ), None, (2, ), 8, (384000, 5248, 37, 0)),
    ("select_test_29", 72, ([1024, 100, 64], [1024, 100, 64], [1024, 100, 64]), (torch.int8, torch.float32, torch.float32), ([1024, 100, 64], ), (torch.float32, ), None, (2, ), 8, (6553600, 7040, 72, 0)),
    ("select_test_30", 72, ([1500, 1000], [1500, 1000], [1500, 1000]), (torch.int8, torch.float32, torch.float32), ([1500, 1000], ), (torch.float32, ), None, (2, ), 8, (1500000, 7040, 72, 0)),
    ("select_test_31", 71, ([512, 200, 200], [512, 200, 200], [512, 200, 200]), (torch.int8, torch.float32, torch.float32), ([512, 200, 200], ), (torch.float32, ), None, (2, ), 8, (20480000, 7040, 71, 0)),
    ("select_test_32", 70, ([256, 200, 200], [256, 200, 200], [256, 200, 200]), (torch.int8, torch.float32, torch.float32), ([256, 200, 200], ), (torch.float32, ), None, (2, ), 8, (10240000, 7040, 70, 0)),
# PYASC_TESTS_END
])
# yapf: enable
def test_select(profiler, runs, kernel_type, test_name, block_num, input_shapes, input_dtypes, output_shapes,
                output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):
    input_shape = input_shapes[1]
    dtype = input_dtypes[1]
    tile_length = tiling_params[1]
    block_num = tiling_params[2]
    unroll_factor = runtime_params[0]

    input_shape_1d = [math.prod(input_shape)]
    cond_dtype = input_dtypes[0]
    in_tensor_c = torch.randint(0, 2, input_shape_1d, dtype=cond_dtype)
    in_tensor_x = torch.randn(input_shape_1d).to(dtype)
    in_tensor_y = torch.randn(input_shape_1d).to(dtype)
    out_tensor = torch.zeros(input_shape_1d, dtype=dtype)

    params = [in_tensor_c, in_tensor_x, in_tensor_y, out_tensor]
    if kernel_type == STATIC:
        params.append(asc2.ConstExpr(input_shape_1d[0]))
    else:
        params.append(input_shape_1d[0])
    params.extend([tile_length, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            select[block_num](*params)

    expected = torch.where(in_tensor_c.bool(), in_tensor_x, in_tensor_y)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
