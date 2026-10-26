# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.experimental import asctile
import pytest
import torch

from .helpers import parametrize_is_static


@asctile.jit(reuse_alloc=2)
def rms_norm_kernel(x_ptr: asctile.GlobalAddress, gamma_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress,
                    num_row, num_col, num_col_align, block_factor, col_flod_factor, ub_factor, epsilon, avg_factor,
                    last_block_factor):
    x_gm = asctile.global_tensor(x_ptr, [num_row, num_col])
    gamma_gm = asctile.global_tensor(gamma_ptr, [1, num_col])
    y_gm = asctile.global_tensor(y_ptr, [num_row, num_col])
    cur_block_factor = last_block_factor if asctile.block_idx() == (asctile.block_num() - 1) else block_factor
    cur_block_loops = asctile.ceildiv(cur_block_factor, ub_factor)
    cur_ub_tails = cur_block_factor - (cur_block_loops - 1) * ub_factor
    base_offset = block_factor * asctile.block_idx()

    gamma = asctile.copy_in(gamma_gm, [0, 0], [1, num_col_align], real_shape=[1, num_col])
    for i in asctile.range(cur_block_loops, unroll_factor=1):  # TODO: fix uf
        cur_ub_factor = cur_ub_tails if i == (cur_block_loops - 1) else ub_factor
        offset = base_offset + i * ub_factor
        x = asctile.copy_in(x_gm, [offset, 0], [ub_factor, num_col_align], real_shape=[cur_ub_factor, num_col])
        y = asctile.rms_norm(x, gamma, epsilon)
        asctile.copy_out(y, y_gm, [offset, 0], real_shape=[cur_ub_factor, num_col])


def rms_golden(input_tensor, gamma, eps):
    rms_norm = torch.nn.RMSNorm(gamma.shape, eps)
    rms_norm.weight.data = gamma
    return rms_norm(input_tensor)


# yapf: disable
@parametrize_is_static()
@pytest.mark.parametrize("test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params", [
# PYASC_TESTS_BEGIN
    pytest.param("rms_norm_1", 2, ([2, 1024], [1024]), (torch.float16, torch.float16), ([2, 1024], [2, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (2, 1024, 1024, 1, 512, 1, 9.999999747378752e-06, 0.0009765625, 1)),
    pytest.param("rms_norm_2", 2, ([2, 2048], [2048]), (torch.float16, torch.float16), ([2, 2048], [2, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (2, 2048, 2048, 1, 1024, 1, 9.999999747378752e-06, 0.00048828125, 1)),
    pytest.param("rms_norm_3", 1, ([1, 2048], [2048]), (torch.float16, torch.float16), ([1, 2048], [1, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (1, 2048, 2048, 1, 1024, 1, 9.999999747378752e-06, 0.00048828125, 1)),
    pytest.param("rms_norm_4", 3, ([3, 2048], [2048]), (torch.float16, torch.float16), ([3, 2048], [3, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (3, 2048, 2048, 1, 1024, 1, 9.999999747378752e-06, 0.00048828125, 1)),
    pytest.param("rms_norm_5", 60, ([300, 1024], [1024]), (torch.float16, torch.float16), ([300, 1024], [300, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (300, 1024, 1024, 5, 512, 5, 9.999999747378752e-06, 0.0009765625, 5)),
    pytest.param("rms_norm_6", 60, ([300, 2048], [2048]), (torch.float16, torch.float16), ([300, 2048], [300, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (300, 2048, 2048, 5, 1024, 5, 9.999999747378752e-06, 0.00048828125, 5)),
    pytest.param("rms_norm_7", 71, ([2048, 2048], [2048]), (torch.float16, torch.float16), ([2048, 2048], [2048, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (2048, 2048, 2048, 29, 1024, 15, 9.999999747378752e-06, 0.00048828125, 18)),
    pytest.param("rms_norm_8", 72, ([3072, 2048], [2048]), (torch.float16, torch.float16), ([3072, 2048], [3072, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (3072, 2048, 2048, 43, 1024, 15, 9.999999747378752e-06, 0.00048828125, 19)),
    pytest.param("rms_norm_9", 70, ([900, 2048], [2048]), (torch.float16, torch.float16), ([900, 2048], [900, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (900, 2048, 2048, 13, 1024, 13, 9.999999747378752e-06, 0.00048828125, 3)),
    pytest.param("rms_norm_10", 70, ([1536, 2048], [2048]), (torch.float16, torch.float16), ([1536, 2048], [1536, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (1536, 2048, 2048, 22, 1024, 15, 9.999999747378752e-06, 0.00048828125, 18) ),
    pytest.param("rms_norm_11", 72, ([4608, 2048], [2048]), (torch.float16, torch.float16), ([4608, 2048], [4608, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (4608, 2048, 2048, 64, 1024, 15, 9.999999747378752e-06, 0.00048828125, 64) ),
    pytest.param("rms_norm_12", 69, ([1024, 2048], [2048]), (torch.float16, torch.float16), ([1024, 2048], [1024, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (1024, 2048, 2048, 15, 1024, 15, 9.999999747378752e-06, 0.00048828125, 4) ),
    pytest.param("rms_norm_13", 67, ([600, 2048], [2048]), (torch.float16, torch.float16), ([600, 2048], [600, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (600, 2048, 2048, 9, 1024, 9, 9.999999747378752e-06, 0.00048828125, 6) ),
    pytest.param("rms_norm_14", 70, ([1536, 1024], [1024]), (torch.float16, torch.float16), ([1536, 1024], [1536, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (1536, 1024, 1024, 22, 512, 22, 9.999999747378752e-06, 0.0009765625, 18) ),
    pytest.param("rms_norm_15", 71, ([2048, 1024], [1024]), (torch.float16, torch.float16), ([2048, 1024], [2048, 1]), (torch.float16, torch.float32), (1e-05), None, 5000, (2048, 1024, 1024, 29, 512, 29, 9.999999747378752e-06, 0.0009765625, 18) )
# PYASC_TESTS_END
])
# yapf: enable
def test_rms_norm(profiler, runs, is_static, test_name, block_num, input_shapes, input_dtypes, output_shapes,
                  output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):
    x_shape, gamma_shape = input_shapes
    y_shape = output_shapes[0]
    x_dtype, gamma_dtype = input_dtypes
    y_dtype = output_dtypes[0]

    x_tensor = torch.randn(x_shape, dtype=x_dtype)
    gamma_tensor = torch.randn(gamma_shape, dtype=gamma_dtype)
    y_tensor = torch.zeros(y_shape, dtype=y_dtype)
    num_row, num_col, num_col_align, block_factor, col_flod_factor, ub_factor, epsilon, avg_factor, last_block_factor = tiling_params

    params = [
        num_row, num_col,
        asctile.ConstExpr(num_col_align),
        asctile.ConstExpr(block_factor), col_flod_factor,
        asctile.ConstExpr(ub_factor), epsilon, avg_factor, last_block_factor
    ]
    if is_static:
        params = list(map(asctile.ConstExpr, params))

    with profiler.profile():
        for _ in range(runs):
            rms_norm_kernel[block_num](x_tensor, gamma_tensor, y_tensor, *params)
    expected = rms_golden(x_tensor, gamma_tensor, epsilon)
    torch.testing.assert_close(y_tensor, expected, atol=2e-3, rtol=2e-3)
