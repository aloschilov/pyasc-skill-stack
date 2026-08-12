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
def cast_direct(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, block_loop_num,
                block_loop_num_tail, block_length, tile_length: asc2.ConstExpr, dst_dtype: asc2.ConstExpr,
                unroll_factor: asc2.ConstExpr):
    x_gm = asc2.global_tensor(input_ptr, [input_length])
    out_gm = asc2.global_tensor(output_ptr, [input_length])

    # block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    # block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        xt = asc2.copy_in(x_gm, [current_offset], [tile_length])
        zt = xt.to(dst_dtype)
        asc2.copy_out(zt, out_gm, [current_offset])


@asc2.jit(reuse_alloc=1)
def cast_two(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, block_loop_num,
             block_loop_num_tail, block_length, tile_length: asc2.ConstExpr, intermediate_dtype: asc2.ConstExpr,
             dst_dtype: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    x_gm = asc2.global_tensor(input_ptr, [input_length])
    out_gm = asc2.global_tensor(output_ptr, [input_length])

    # block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    # block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        xt = asc2.copy_in(x_gm, [current_offset], [tile_length])
        middle_tile = xt.to(intermediate_dtype)
        zt = middle_tile.to(dst_dtype)
        asc2.copy_out(zt, out_gm, [current_offset])


# DYNAMIC [2, 5, 7, 42767] only supports unroll_factor = 1
# yapf: disable
@pytest.mark.parametrize("kernel_type", [STATIC])
@pytest.mark.parametrize("test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params", [
# PYASC_TESTS_BEGIN
    ("cast_test_1", 8, ([128, 64], ), (torch.int32, ), ([128, 64], ), (torch.float16, ), (1, ), (2, 1), 0, (8, 21120, 1024, 1, 1, 1024, 1024, 0, 0, 0, 0, 0)),
    ("cast_test_2", 4, ([128, 32], ), (torch.int32, ), ([128, 32], ), (torch.float16, ), (1, ), (2, 1), 0, (4, 21120, 1024, 1, 1, 1024, 1024, 0, 0, 0, 0, 0)),
    ("cast_test_3", 10, ([128, 80], ), (torch.int32, ), ([128, 80], ), (torch.float16, ), (1, ), (2, 1), 0, (10, 21120, 1024, 1, 1, 1024, 1024, 0, 0, 0, 0, 0)),
    ("cast_test_4", 40, ([400, 1, 50], ), (torch.int64, ), ([400, 1, 50], ), (torch.float32, ), (0, ), (2, 0), 0, (40, 10560, 504, 1, 1, 504, 344, 0, 0, 0, 0, 0)),
    ("cast_test_5", 1, ([128], ), (torch.int64, ), ([128], ), (torch.int32, ), (3, ), (2, 3), 0, (1, 10560, 128, 1, 1, 128, 128, 32, 16, 330, 4, 4)),
    # BUG: int8 -> int64 ("cast_test_6", 4, ([128, 128], ), (torch.int8, ), ([128, 128], ), (torch.int64, ), (9, ), (2, 9), 0, (4, 9728, 4096, 1, 1, 4096, 4096, 0, 0, 0, 0, 0)),
    # BUG: int8 -> int64 ("cast_test_7", 1, ([1, 256], ), (torch.int8, ), ([1, 256], ), (torch.int64, ), (9, ), (2, 9), 0, (1, 9728, 256, 1, 1, 256, 256, 0, 0, 0, 0, 0)),
    # BUG: int8 -> int64 ("cast_test_8", 1, ([1, 300], ), (torch.int8, ), ([1, 300], ), (torch.int64, ), (9, ), (2, 9), 0, (1, 9728, 304, 1, 1, 304, 300, 0, 0, 0, 0, 0)),
    ("cast_test_9", 2, ([7376, 1], ), (torch.int8, ), ([7376, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (2, 17920, 3688, 1, 1, 3688, 3688, 0, 0, 0, 0, 0)),
    ("cast_test_10", 8, ([100, 1, 302], ), (torch.int8, ), ([100, 1, 302], ), (torch.float32, ), (0, ), (2, 0), 0, (8, 17920, 3776, 1, 1, 3776, 3768, 0, 0, 0, 0, 0)),
    ("cast_test_11", 64, ([650, 100], ), (torch.int32, ), ([650, 100], ), (torch.float32, ), (0, ), (2, 0), 0, (64, 15808, 1016, 1, 1, 1016, 992, 0, 0, 0, 0, 0)),
    ("cast_test_12", 2, ([800, 1], ), (torch.int64, ), ([800, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (2, 10560, 400, 1, 1, 400, 400, 0, 0, 0, 0, 0)),
    ("cast_test_13", 10, ([100, 100], ), (torch.int32, ), ([100, 100], ), (torch.float32, ), (0, ), (2, 0), 0, (10, 15808, 1000, 1, 1, 1000, 1000, 0, 0, 0, 0, 0)),
    ("cast_test_14", 1, ([100, 1], ), (torch.int64, ), ([100, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (1, 10560, 104, 1, 1, 104, 100, 0, 0, 0, 0, 0)),
    ("cast_test_15", 8, ([1, 1, 4000], ), (torch.int64, ), ([1, 1, 4000], ), (torch.float32, ), (0, ), (2, 0), 0, (8, 10560, 504, 1, 1, 504, 472, 0, 0, 0, 0, 0)),
    ("cast_test_16", 19, ([982, 19], ), (torch.float32, ), ([982, 19], ), (torch.bfloat16, ), (27, ), (2, 27), 0, (19, 21120, 984, 1, 1, 984, 946, 0, 0, 0, 0, 0)),
    ("cast_test_17", 16, ([822, 19], ), (torch.float32, ), ([822, 19], ), (torch.bfloat16, ), (27, ), (2, 27), 0, (16, 21120, 984, 1, 1, 984, 858, 0, 0, 0, 0, 0)),
    ("cast_test_18", 4, ([417, 9], ), (torch.float32, ), ([417, 9], ), (torch.bfloat16, ), (27, ), (2, 27), 0, (4, 21120, 944, 1, 1, 944, 921, 0, 0, 0, 0, 0)),
    ("cast_test_19", 2, ([178, 9], ), (torch.float32, ), ([178, 9], ), (torch.bfloat16, ), (27, ), (2, 27), 0, (2, 21120, 808, 1, 1, 808, 794, 0, 0, 0, 0, 0)),
    ("cast_test_20", 1, ([2, 512], ), (torch.int32, ), ([2, 512], ), (torch.int64, ), (9, ), (2, 9), 0, (1, 10560, 1024, 1, 1, 1024, 1024, 0, 0, 0, 0, 0)),
    ("cast_test_21", 2, ([2, 8, 16, 16], ), (torch.float16, ), ([2, 8, 16, 16], ), (torch.int8, ), (12, ), (2, 12), 0, (2, 34944, 2048, 1, 1, 2048, 2048, 0, 0, 0, 0, 0)),
    ("cast_test_22", 72, ([700, 192], ), (torch.float32, ), ([700, 192], ), (torch.int32, ), (3, ), (2, 3), 0, (72, 15808, 1872, 1, 1, 1872, 1488, 0, 0, 0, 0, 0)),
    ("cast_test_23", 72, ([1500, 60, 9], ), (torch.float32, ), ([1500, 60, 9], ), (torch.bfloat16, ), (27, ), (2, 27), 0, (72, 21120, 11256, 1, 1, 11256, 10824, 0, 0, 0, 0, 0)),
    ("cast_test_24", 64, ([259230, 1], ), (torch.int8, ), ([259230, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (64, 17920, 4056, 1, 1, 4056, 3702, 0, 0, 0, 0, 0)),
    ("cast_test_25", 72, ([700, 1, 192], ), (torch.int64, ), ([700, 1, 192], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 10560, 1872, 1, 1, 1872, 1488, 0, 0, 0, 0, 0)),
    ("cast_test_26", 72, ([16, 64, 160], ), (torch.float32, ), ([16, 64, 160], ), (torch.float16, ), (1, ), (2, 1), 0, (72, 21120, 2280, 1, 1, 2280, 1960, 0, 0, 0, 0, 0)),
    ("cast_test_27", 72, ([16, 5, 64, 64], ), (torch.float16, ), ([16, 5, 64, 64], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 21120, 4552, 1, 1, 4552, 4488, 0, 0, 0, 0, 0)),
    ("cast_test_28", 72, ([298294, 1], ), (torch.int8, ), ([298294, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 17920, 4144, 1, 1, 4144, 4070, 0, 0, 0, 0, 0)),
    ("cast_test_29", 72, ([854096, 1], ), (torch.int8, ), ([854096, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 17920, 11864, 1, 1, 11864, 11752, 0, 0, 0, 0, 0)),
    ("cast_test_30", 72, ([883357, 1], ), (torch.int8, ), ([883357, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 17920, 12272, 1, 1, 12272, 12045, 0, 0, 0, 0, 0)),
    ("cast_test_31", 72, ([1500, 50, 9], ), (torch.bfloat16, ), ([1500, 50, 9], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 21120, 9376, 1, 1, 9376, 9304, 0, 0, 0, 0, 0)),
    ("cast_test_32", 37, ([150834, 1], ), (torch.int8, ), ([150834, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (37, 17920, 4080, 1, 1, 4080, 3954, 0, 0, 0, 0, 0)),
    ("cast_test_33", 34, ([135678, 1], ), (torch.int8, ), ([135678, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (34, 17920, 3992, 1, 1, 3992, 3942, 0, 0, 0, 0, 0)),
    ("cast_test_34", 72, ([589824], ), (torch.float32, ), ([589824], ), (torch.float16, ), (1, ), (2, 1), 0, (72, 21120, 8192, 1, 1, 8192, 8192, 0, 0, 0, 0, 0)),
    ("cast_test_35", 72, ([160, 640], ), (torch.float32, ), ([160, 640], ), (torch.float16, ), (1, ), (2, 1), 0, (72, 21120, 1424, 1, 1, 1424, 1296, 0, 0, 0, 0, 0)),
    ("cast_test_36", 72, ([335872], ), (torch.float32, ), ([335872], ), (torch.bfloat16, ), (27, ), (2, 27), 0, (72, 21120, 4672, 1, 1, 4672, 4160, 0, 0, 0, 0, 0)),
    ("cast_test_37", 64, ([131072], ), (torch.bfloat16, ), ([131072], ), (torch.float32, ), (0, ), (2, 0), 0, (64, 21120, 2048, 1, 1, 2048, 2048, 0, 0, 0, 0, 0)),
    ("cast_test_38", 72, ([317062, 2], ), (torch.int64, ), ([317062, 2], ), (torch.int32, ), (3, ), (2, 3), 0, (72, 10560, 8808, 1, 1, 8808, 8756, 32, 16, 330, 276, 274)),
    ("cast_test_39", 72, ([182366, 2], ), (torch.int64, ), ([182366, 2], ), (torch.int32, ), (3, ), (2, 3), 0, (72, 10560, 5072, 1, 1, 5072, 4620, 32, 16, 330, 159, 145)),
    ("cast_test_40", 35, ([141939, 1], ), (torch.int8, ), ([141939, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (35, 17920, 4056, 1, 1, 4056, 4035, 0, 0, 0, 0, 0)),
    ("cast_test_41", 34, ([139107, 1], ), (torch.int8, ), ([139107, 1], ), (torch.float32, ), (0, ), (2, 0), 0, (34, 17920, 4096, 1, 1, 4096, 3939, 0, 0, 0, 0, 0)),
    ("cast_test_42", 37, ([3, 25000], ), (torch.bfloat16, ), ([3, 25000], ), (torch.float32, ), (0, ), (2, 0), 0, (37, 21120, 2032, 1, 1, 2032, 1848, 0, 0, 0, 0, 0)),
    ("cast_test_43", 72, ([700, 20000], ), (torch.int32, ), ([700, 20000], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 15808, 194448, 13, 13, 4752, 4496, 0, 0, 0, 0, 0)),
    ("cast_test_44", 72, ([7000, 20, 32], ), (torch.int64, ), ([7000, 20, 32], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 10560, 62224, 6, 6, 9424, 9296, 0, 0, 0, 0, 0)),
    ("cast_test_45", 72, ([2048, 4506], ), (torch.bfloat16, ), ([2048, 4506], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 21120, 128176, 7, 7, 1456, 1072, 0, 0, 0, 0, 0)),
    ("cast_test_46", 72, ([213897504], ), (torch.float32, ), ([213897504], ), (torch.bfloat16, ), (27, ), (2, 27), 0, (72, 21120, 2970800, 141, 141, 14000, 13904, 0, 0, 0, 0, 0)),
    ("cast_test_47", 72, ([1500, 14144], ), (torch.float32, ), ([1500, 14144], ), (torch.bfloat16, ), (27, ), (2, 27), 0, (72, 21120, 294672, 14, 14, 20112, 19728, 0, 0, 0, 0, 0)),
    ("cast_test_48", 72, ([276233, 97], ), (torch.float32, ), ([276233, 97], ), (torch.bfloat16, ), (27, ), (2, 27), 0, (72, 21120, 372152, 18, 18, 13112, 12769, 0, 0, 0, 0, 0)),
    ("cast_test_49", 72, ([1200, 5, 4, 4, 16, 16], ), (torch.float32, ), ([1200, 5, 4, 4, 16, 16], ), (torch.float16, ), (1, ), (2, 1), 0, (72, 21120, 341336, 17, 17, 3416, 3224, 0, 0, 0, 0, 0)),
    ("cast_test_50", 72, ([4608, 115, 12], ), (torch.int8, ), ([4608, 115, 12], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 17920, 88320, 5, 5, 16640, 16640, 0, 0, 0, 0, 0)),
    ("cast_test_51", 72, ([2059, 2059], ), (torch.int8, ), ([2059, 2059], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 17920, 58888, 4, 4, 5128, 4673, 0, 0, 0, 0, 0)),
    ("cast_test_52", 72, ([1536, 25000], ), (torch.bfloat16, ), ([1536, 25000], ), (torch.float32, ), (0, ), (2, 0), 0, (72, 21120, 533336, 26, 26, 5336, 5144, 0, 0, 0, 0, 0)),
# PYASC_TESTS_END
])
# yapf: enable
def test_cast(profiler, runs, kernel_type, test_name, block_num, input_shapes, input_dtypes, output_shapes,
              output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):
    input_shape = input_shapes[0]
    input_dtype = input_dtypes[0]
    output_dtype = output_dtypes[0]
    tile_length = tiling_params[1]
    block_length = tiling_params[2]
    block_loop_num = tiling_params[3]
    block_loop_num_tail = tiling_params[4]
    unroll_factor = runtime_params[0]

    # There is no cast for bool in Ascend, use int8 instead
    input_dtype = torch.int8 if input_dtype == torch.bool else input_dtype
    output_dtype = torch.int8 if output_dtype == torch.bool else output_dtype

    input_shape_1d = [math.prod(input_shape)]

    if block_loop_num == 1 and block_loop_num_tail == 1:
        tile_length = block_length

    # Fix range to correct test int8 cast
    low, high = -127, 128

    if input_dtype.is_floating_point:
        if input_dtype == torch.bfloat16:
            in_tensor_x = torch.empty(input_shape_1d, dtype=torch.float32).uniform_(float(low),
                                                                                    float(high)).to(input_dtype)
        else:
            in_tensor_x = torch.empty(input_shape_1d, dtype=input_dtype).uniform_(float(low), float(high))
    else:
        in_tensor_x = torch.randint(low, high, input_shape_1d, dtype=input_dtype)
    out_tensor = torch.zeros(input_shape_1d, dtype=output_dtype)

    dtype_map = {
        torch.int8: asc2.int8,
        torch.int16: asc2.int16,
        torch.int32: asc2.int32,
        torch.int64: asc2.int64,
        torch.float16: asc2.float16,
        torch.float32: asc2.float32,
        torch.bfloat16: asc2.bfloat16,
    }
    dst_dtype = dtype_map[output_dtype]

    params = [in_tensor_x, out_tensor]
    if kernel_type == STATIC:
        params.extend([
            asc2.ConstExpr(input_shape_1d[0]),
            asc2.ConstExpr(block_loop_num),
            asc2.ConstExpr(block_loop_num_tail),
            asc2.ConstExpr(block_length)
        ])
    else:
        params.extend([input_shape_1d[0], block_loop_num, block_loop_num_tail, block_length])

    if input_dtype == torch.int8 or output_dtype == torch.int8:
        intermediate_dtype = asc2.float16
        params.extend([tile_length, intermediate_dtype, dst_dtype, unroll_factor])

        with profiler.profile():
            for _ in range(runs):
                cast_two[block_num](*params)
    else:
        params.extend([tile_length, dst_dtype, unroll_factor])

        with profiler.profile():
            for _ in range(runs):
                cast_direct[block_num](*params)

    expected = in_tensor_x.to(output_dtype)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
