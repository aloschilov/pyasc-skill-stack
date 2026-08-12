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
def add(input_x_ptr: asc2.GlobalAddress, input_y_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length,
        tile_length: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    x = asc2.global_tensor(input_x_ptr, [input_length])
    y = asc2.global_tensor(input_y_ptr, [input_length])
    z = asc2.global_tensor(output_ptr, [input_length])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        xt = asc2.copy_in(x, [current_offset], [tile_length])
        yt = asc2.copy_in(y, [current_offset], [tile_length])
        zt = xt + yt
        asc2.copy_out(zt, z, [current_offset])


# yapf: disable
@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params", [
# PYASC_TESTS_BEGIN
    ("vadd_test_1", 43, ([128, 2, 1, 128], [128, 2, 1, 128]), (torch.float16, torch.float16), ([128, 2, 1, 128], ), (torch.float16, ), None, (2, ), 8, (32768, 384, 43, 0)),
    ("vadd_test_2", 1, ([199], [199]), (torch.float32, torch.float32), ([199], ), (torch.float32, ), None, (2, ), 8, (199, 128, 1, 0)),
    ("vadd_test_3", 43, ([128, 256], [128, 256]), (torch.float16, torch.float16), ([128, 256], ), (torch.float16, ), None, (2, ), 8, (32768, 384, 43, 0)),
    ("vadd_test_4", 4, ([1024], [1024]), (torch.float32, torch.float32), ([1024], ), (torch.float32, ), None, (2, ), 8, (1024, 128, 4, 0)),
    ("vadd_test_5", 8, ([16, 128], [16, 128]), (torch.float32, torch.float32), ([16, 128], ), (torch.float32, ), None, (2, ), 8, (2048, 128, 8, 0)),
    ("vadd_test_6", 24, ([16, 380], [16, 380]), (torch.float32, torch.float32), ([16, 380], ), (torch.float32, ), None, (2, ), 8, (6080, 128, 24, 0)),
    ("vadd_test_7", 1, ([16, 2], [16, 2]), (torch.float32, torch.float32), ([16, 2], ), (torch.float32, ), None, (2, ), 8, (32, 128, 1, 0)),
    ("vadd_test_8", 44, ([128, 1, 88], [128, 1, 88]), (torch.float32, torch.float32), ([128, 1, 88], ), (torch.float32, ), None, (2, ), 8, (11264, 128, 44, 0)),
    ("vadd_test_9", 40, ([2500, 24], [2500, 24]), (torch.float32, torch.float32), ([2500, 24], ), (torch.float32, ), None, (2, ), 8, (60000, 768, 40, 0)),
    ("vadd_test_10", 43, ([16, 1360], [16, 1360]), (torch.float32, torch.float32), ([16, 1360], ), (torch.float32, ), None, (2, ), 8, (21760, 256, 43, 0)),
    ("vadd_test_11", 6, ([3072], [3072]), (torch.bfloat16, torch.bfloat16), ([3072], ), (torch.bfloat16, ), None, (2, ), 8, (3072, 256, 6, 0)),
    ("vadd_test_12", 16, ([16, 256], [16, 256]), (torch.float32, torch.float32), ([16, 256], ), (torch.float32, ), None, (2, ), 8, (4096, 128, 16, 0)),
    ("vadd_test_13", 1, ([16], [16]), (torch.float32, torch.float32), ([16], ), (torch.float32, ), None, (2, ), 8, (16, 128, 1, 0)),
    ("vadd_test_14", 64, ([16, 32, 32], [16, 32, 32]), (torch.float32, torch.float32), ([16, 32, 32], ), (torch.float32, ), None, (2, ), 8, (16384, 128, 64, 0)),
    ("vadd_test_15", 1, ([16, 3], [16, 3]), (torch.float32, torch.float32), ([16, 3], ), (torch.float32, ), None, (2, ), 8, (48, 128, 1, 0)),
    ("vadd_test_16", 1, ([128], [128]), (torch.float32, torch.float32), ([128], ), (torch.float32, ), None, (2, ), 8, (128, 128, 1, 0)),
    ("vadd_test_17", 1, ([256, 1], [256, 1]), (torch.float32, torch.float32), ([256, 1], ), (torch.float32, ), None, (2, ), 8, (256, 128, 1, 0)),
    ("vadd_test_18", 40, ([128, 80], [128, 80]), (torch.float32, torch.float32), ([128, 80], ), (torch.float32, ), None, (2, ), 8, (10240, 128, 40, 0)),
    ("vadd_test_19", 44, ([128, 352], [128, 352]), (torch.float16, torch.float16), ([128, 352], ), (torch.float16, ), None, (2, ), 8, (45056, 512, 44, 0)),
    ("vadd_test_20", 55, ([7000, 2], [7000, 2]), (torch.float32, torch.float32), ([7000, 2], ), (torch.float32, ), None, (2, ), 8, (14000, 128, 55, 0)),
    ("vadd_test_21", 38, ([100, 1623, 1], [100, 1623, 1]), (torch.float32, torch.float32), ([100, 1623, 1], ), (torch.float32, ), None, (2, ), 8, (162300, 2176, 38, 0)),
    ("vadd_test_22", 37, ([200, 1338, 1], [200, 1338, 1]), (torch.float32, torch.float32), ([200, 1338, 1], ), (torch.float32, ), None, (2, ), 8, (267600, 3712, 37, 0)),
    ("vadd_test_23", 37, ([100, 5905, 1], [100, 5905, 1]), (torch.float32, torch.float32), ([100, 5905, 1], ), (torch.float32, ), None, (2, ), 8, (590500, 8192, 37, 0)),
    ("vadd_test_24", 51, ([750, 2132, 1], [750, 2132, 1]), (torch.float32, torch.float32), ([750, 2132, 1], ), (torch.float32, ), None, (2, ), 8, (1599000, 10496, 51, 0)),
    ("vadd_test_25", 43, ([7000, 1, 128], [7000, 1, 128]), (torch.float32, torch.float32), ([7000, 1, 128], ), (torch.float32, ), None, (2, ), 8, (896000, 10496, 43, 0)),
    ("vadd_test_26", 37, ([100, 5840, 1], [100, 5840, 1]), (torch.float32, torch.float32), ([100, 5840, 1], ), (torch.float32, ), None, (2, ), 8, (584000, 8064, 37, 0)),
    ("vadd_test_27", 48, ([1024, 976], [1024, 976]), (torch.float32, torch.float32), ([1024, 976], ), (torch.float32, ), None, (2, ), 8, (999424, 10496, 48, 0)),
    ("vadd_test_28", 39, ([15991, 8], [15991, 8]), (torch.float32, torch.float32), ([15991, 8], ), (torch.float32, ), None, (2, ), 8, (127928, 1664, 39, 0)),
    ("vadd_test_29", 37, ([2048, 60, 1], [1, 1, 1]), (torch.bfloat16, torch.bfloat16), ([2048, 60, 1], ), (torch.bfloat16, ), None, (2, ), 8, (122880, 1664, 37, 2)),
    ("vadd_test_30", 37, ([128, 2, 512], [128, 2, 512]), (torch.float16, torch.float16), ([128, 2, 512], ), (torch.float16, ), None, (2, ), 8, (131072, 1792, 37, 0)),
    ("vadd_test_31", 38, ([100, 51, 32], [100, 51, 32]), (torch.float32, torch.float32), ([100, 51, 32], ), (torch.float32, ), None, (2, ), 8, (163200, 2176, 38, 0)),
    ("vadd_test_32", 51, ([400, 4000], [400, 4000]), (torch.float32, torch.float32), ([400, 4000], ), (torch.float32, ), None, (2, ), 8, (1600000, 10496, 51, 0)),
    ("vadd_test_33", 55, ([800, 2135, 1], [800, 2135, 1]), (torch.float32, torch.float32), ([800, 2135, 1], ), (torch.float32, ), None, (2, ), 8, (1708000, 10496, 55, 0)),
    ("vadd_test_34", 37, ([800, 768, 1], [800, 768, 1]), (torch.float32, torch.float32), ([800, 768, 1], ), (torch.float32, ), None, (2, ), 8, (614400, 8448, 37, 0)),
    ("vadd_test_35", 39, ([8678, 8], [8678, 8]), (torch.float32, torch.float32), ([8678, 8], ), (torch.float32, ), None, (2, ), 8, (69424, 896, 39, 0)),
    ("vadd_test_36", 37, ([39698, 8], [39698, 8]), (torch.float32, torch.float32), ([39698, 8], ), (torch.float32, ), None, (2, ), 8, (317584, 4352, 37, 0)),
    ("vadd_test_37", 62, ([650, 4000], [650, 4000]), (torch.float32, torch.float32), ([650, 4000], ), (torch.float32, ), None, (2, ), 8, (2600000, 10496, 62, 0)),
    ("vadd_test_38", 39, ([16031, 8], [16031, 8]), (torch.float32, torch.float32), ([16031, 8], ), (torch.float32, ), None, (2, ), 8, (128248, 1664, 39, 0)),
    ("vadd_test_39", 37, ([29172, 8], [29172, 8]), (torch.float32, torch.float32), ([29172, 8], ), (torch.float32, ), None, (2, ), 8, (233376, 3200, 37, 0)),
    ("vadd_test_40", 37, ([1024, 1, 128], [1024, 1, 128]), (torch.float32, torch.float32), ([1024, 1, 128], ), (torch.float32, ), None, (2, ), 8, (131072, 1792, 37, 0)),
    ("vadd_test_41", 20, ([16, 2, 160], [16, 2, 160]), (torch.float32, torch.float32), ([16, 2, 160], ), (torch.float32, ), None, (2, ), 8, (5120, 128, 20, 0)),
    ("vadd_test_42", 72, ([2400, 300, 256], [2400, 300, 256]), (torch.float32, torch.float32), ([2400, 300, 256], ), (torch.float32, ), None, (2, ), 8, (184320000, 10496, 72, 0)),
    ("vadd_test_43", 71, ([700, 20000], [700, 20000]), (torch.float32, torch.float32), ([700, 20000], ), (torch.float32, ), None, (2, ), 8, (14000000, 10496, 71, 0)),
    ("vadd_test_44", 65, ([189420, 32], [189420, 32]), (torch.float32, torch.float32), ([189420, 32], ), (torch.float32, ), None, (2, ), 8, (6061440, 10496, 65, 0)),
    ("vadd_test_45", 71, ([833950, 24], [833950, 24]), (torch.float32, torch.float32), ([833950, 24], ), (torch.float32, ), None, (2, ), 8, (20014800, 10496, 71, 0)),
    ("vadd_test_46", 70, ([298294, 32], [298294, 32]), (torch.float32, torch.float32), ([298294, 32], ), (torch.float32, ), None, (2, ), 8, (9545408, 10496, 70, 0)),
    ("vadd_test_47", 66, ([1500, 18, 256], [1500, 18, 256]), (torch.float32, torch.float32), ([1500, 18, 256], ), (torch.float32, ), None, (2, ), 8, (6912000, 10496, 66, 0)),
    ("vadd_test_48", 64, ([1500, 30, 104], [1500, 30, 104]), (torch.float32, torch.float32), ([1500, 30, 104], ), (torch.float32, ), None, (2, ), 8, (4680000, 10496, 64, 0)),
    ("vadd_test_49", 70, ([342312, 32], [342312, 32]), (torch.float32, torch.float32), ([342312, 32], ), (torch.float32, ), None, (2, ), 8, (10953984, 10496, 70, 0)),
    ("vadd_test_50", 72, ([2400, 300, 128], [2400, 300, 128]), (torch.float32, torch.float32), ([2400, 300, 128], ), (torch.float32, ), None, (2, ), 8, (92160000, 10496, 72, 0)),
    ("vadd_test_51", 72, ([2500, 100, 145], [2500, 100, 145]), (torch.float32, torch.float32), ([2500, 100, 145], ), (torch.float32, ), None, (2, ), 8, (36250000, 10496, 72, 0)),
    ("vadd_test_52", 70, ([512, 25000], [512, 25000]), (torch.bfloat16, torch.bfloat16), ([512, 25000], ), (torch.bfloat16, ), None, (2, ), 8, (12800000, 7040, 70, 0)),
    ("vadd_test_53", 72, ([1024, 25000], [1024, 25000]), (torch.bfloat16, torch.bfloat16), ([1024, 25000], ), (torch.bfloat16, ), None, (2, ), 8, (25600000, 7040, 72, 0)),
    ("vadd_test_54", 72, ([1536, 25000], [1536, 25000]), (torch.bfloat16, torch.bfloat16), ([1536, 25000], ), (torch.bfloat16, ), None, (2, ), 8, (38400000, 7040, 72, 0)),
# PYASC_TESTS_END
])
# yapf: enable
def test_add(profiler, runs, kernel_type, test_name, block_num, input_shapes, input_dtypes, output_shapes,
             output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):
    input_shape = input_shapes[0]
    dtype = input_dtypes[0]
    tile_length = tiling_params[1]
    unroll_factor = runtime_params[0]

    input_shape_1d = [math.prod(input_shape)]
    in_tensor_x = torch.randn(input_shape_1d, dtype=dtype)
    in_tensor_y = torch.randn(input_shape_1d, dtype=dtype)
    out_tensor = torch.zeros(input_shape_1d, dtype=dtype)

    params = [in_tensor_x, in_tensor_y, out_tensor]
    if kernel_type == STATIC:
        params.append(asc2.ConstExpr(input_shape_1d[0]))
    else:
        params.append(input_shape_1d[0])
    params.extend([tile_length, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            add[block_num](*params)

    expected = torch.add(in_tensor_x, in_tensor_y)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
