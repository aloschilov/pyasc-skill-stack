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
import torch.nn.functional as F

from .helpers import parametrize_is_static


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def layer_norm_batch(x_ptr: asctile.GlobalAddress, gamma_ptr: asctile.GlobalAddress, beta_ptr: asctile.GlobalAddress,
                     y_ptr: asctile.GlobalAddress, unroll_factor: asctile.ConstExpr[int], l2_cache: asctile.ConstExpr,
                     a, a_block_factor, a_ub_factor, r, r_align, former_block_ub_loops, tail_block_ub_loops, epsilon):
    l2_x, l2_gamma, l2_beta, l2_y = l2_cache
    x_gm = asctile.global_tensor(x_ptr, [a, r], l2_cache=l2_x)
    gamma_gm = asctile.global_tensor(gamma_ptr, [1, r], l2_cache=l2_gamma)
    beta_gm = asctile.global_tensor(beta_ptr, [1, r], l2_cache=l2_beta)
    y_gm = asctile.global_tensor(y_ptr, [a, r], l2_cache=l2_y)

    block_offset = asctile.block_idx() * a_block_factor
    ub_loop_num = tail_block_ub_loops if asctile.block_idx() == asctile.block_num() - 1 else former_block_ub_loops
    tail_a = a - a_block_factor * (asctile.block_num() - 1) - (
        tail_block_ub_loops - 1) * a_ub_factor if asctile.block_idx(
        ) == asctile.block_num() - 1 else a_block_factor - (former_block_ub_loops - 1) * a_ub_factor

    gamma = asctile.copy_in(gamma_gm, offsets=[0, 0], shape=[1, r_align], real_shape=[1, r])
    beta = asctile.copy_in(beta_gm, offsets=[0, 0], shape=[1, r_align], real_shape=[1, r])
    gamma_f32 = gamma.to(asctile.float32)
    beta_f32 = beta.to(asctile.float32)

    for ub_loop_idx in asctile.range(ub_loop_num, unroll_factor=unroll_factor):
        current_a = tail_a if ub_loop_num == ub_loop_idx - 1 else a_ub_factor
        a_offset = ub_loop_idx * a_ub_factor
        row_offset = block_offset + a_offset
        x = asctile.copy_in(x_gm, offsets=[row_offset, 0], shape=[a_ub_factor, r_align], real_shape=[current_a, r])
        x_f32 = x.to(asctile.float32)
        r1 = asctile.reduce_sum(x_f32, 1, keep_dims=True)
        mean = r1 / r
        r2 = asctile.reduce_sum(x_f32 * x_f32, 1, keep_dims=True)
        sq_mean = r2 / r
        var = sq_mean - mean * mean
        inv_std = asctile.rsqrt(var + epsilon)
        gamma_f32_ = asctile.broadcast_to(gamma_f32, [a_ub_factor, r_align])
        beta_f32_ = asctile.broadcast_to(beta_f32, [a_ub_factor, r_align])
        y = (x_f32 - mean) * inv_std * gamma_f32_ + beta_f32_
        asctile.copy_out(y.to(x.dtype), y_gm, offsets=[row_offset, 0], real_shape=[current_a, r])


def layer_norm_golden(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, norm_size: int, eps: float,
                      output_dtype: torch.dtype) -> torch.Tensor:
    return F.layer_norm(x.float(), [norm_size], gamma.float(), beta.float(), eps).to(output_dtype)


# yapf: disable
@parametrize_is_static()
@pytest.mark.parametrize("test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params", [
# PYASC_TESTS_BEGIN
    pytest.param("test_1", 72, ([100, 52, 10], [1], [10], [10]), (torch.float32, torch.int32, torch.float32, torch.float32), ([100, 52, 10], [100, 52, 1], [100, 52, 1]), (torch.float32, torch.float32, torch.float32), (1e-05, (True, True, True, True)), (1e-05, ), 500, (5200, 73, 559, 560, 10, 16, 1, 1, 16, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_2", 69, ([1024, 768], [768], [768]), (torch.float32, torch.float32, torch.float32), ([1024, 768], [1024, 1], [1024, 1]), (torch.float32, torch.float32, torch.float32), (-1, -1, (True, True, True, True)), None, 500, (1024, 15, 11, 16, 768, 768, 2, 1, 1024, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_3", 71, ([2048, 152], [152], [152]), (torch.float32, torch.float32, torch.float32), ([2048, 152], [2048, 1], [2048, 1]), (torch.float32, torch.float32, torch.float32), (-1, -1, (True, True, True, True)), None, 500, (2048, 29, 59, 64, 152, 152, 1, 1, 256, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_4", 71, ([2048, 256], [256], [256]), (torch.float32, torch.float32, torch.float32), ([2048, 256], [2048, 1], [2048, 1]), (torch.float32, torch.float32, torch.float32), (-1, -1, (True, True, True, True)), None, 500, (2048, 29, 35, 40, 256, 256, 1, 1, 256, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_5", 72, ([4096, 50, 32], [32], [32]), (torch.float32, torch.float32, torch.float32), ([4096, 50, 32], [4096, 50, 1], [4096, 50, 1]), (torch.float32, torch.float32, torch.float32), (-1, -1, (False, True, True, True)), None, 500, (204800, 2845, 281, 288, 32, 32, 11, 10, 32, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_6", 72, ([2000, 4096], [4096], [4096]), (torch.float32, torch.float32, torch.float32), ([2000, 4096], [2000, 1], [2000, 1]), (torch.float32, torch.float32, torch.float32), (-1, -1, (False, True, True, True)), None, 500, (2000, 28, 2, 8, 4096, 4096, 14, 6, 4096, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_7", 72, ([4096, 50, 16], [16], [16]), (torch.float32, torch.float32, torch.float32), ([4096, 50, 16], [4096, 50, 1], [4096, 50, 1]), (torch.float32, torch.float32, torch.float32), (-1, -1, (True, True, True, True)), None, 500, (204800, 2845, 559, 560, 16, 16, 6, 6, 16, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_8", 72, ([50, 4096, 16], [16], [16]), (torch.float32, torch.float32, torch.float32), ([50, 4096, 16], [50, 4096, 1], [50, 4096, 1]), (torch.float32, torch.float32, torch.float32), (-1, -1, (True, True, True, True)), None, 500, (204800, 2845, 559, 560, 16, 16, 6, 6, 16, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_9", 72, ([4096, 39, 64], [64], [64]), (torch.float32, torch.float32, torch.float32), ([4096, 39, 64], [4096, 39, 1], [4096, 39, 1]), (torch.float32, torch.float32, torch.float32), (-1, -1, (False, True, True, True)), None, 500, (159744, 2219, 140, 144, 64, 64, 16, 16, 64, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_10", 72, ([1024, 32, 128], [128], [128]), (torch.float32, torch.float32, torch.float32), ([1024, 32, 128], [1024, 32, 1], [1024, 32, 1]), (torch.float32, torch.float32, torch.float32), (-1, -1, (False, True, True, True)), None, 500, (32768, 456, 70, 72, 128, 128, 7, 6, 128, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_11", 72, ([2000, 4096], [1], [4096], [4096]), (torch.bfloat16, torch.int32, torch.bfloat16, torch.bfloat16), ([2000, 4096], [2000, 1], [2000, 1]), (torch.bfloat16, torch.float32,torch.float32), (1e-05, (True, True, True, True)), (1e-05, ), 522, (2000, 28, 2, 8, 4096, 4096, 14, 6, 4096, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
    pytest.param("test_12", 72, ([2000, 1024], [1], [1024], [1024]), (torch.bfloat16, torch.int32, torch.bfloat16, torch.bfloat16), ([2000, 1024], [2000, 1], [2000, 1]), (torch.bfloat16, torch.float32,torch.float32), (1e-05, (True, True, True, True)), (1e-05, ), 522, (2000, 28, 7, 16, 1024, 1024, 4, 2, 1024, 9.999999747378752e-06, 0, 0, [0, 0, 0, 0, 0, 0])),
# PYASC_TESTS_END
])
# yapf: enable
def test_layer_norm(profiler, runs, is_static, test_name, block_num, input_shapes, input_dtypes, output_shapes,
                    output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):
    assert tiling_key in [500, 522], "unsupported tiling_key"
    unroll_factor = 2
    (a, a_block_factor, a_ub_factor, a_ub_factor_align_b32, r, r_align, former_block_ub_loops, tail_block_ub_loops,
     power_of_two_for_r, epsilon, nullptr_gamma, nullptr_beta, tiling_data) = tiling_params
    l2_cache = compile_params[-1]

    input_dtype = input_dtypes[0]
    output_dtype = output_dtypes[0]
    params = [
        a,
        asctile.ConstExpr(a_block_factor),
        asctile.ConstExpr(a_ub_factor), r,
        asctile.ConstExpr(r_align), former_block_ub_loops, tail_block_ub_loops, epsilon
    ]
    if is_static:
        params = list(map(asctile.ConstExpr, params))

    x_tensor = torch.randn([a, r], dtype=input_dtype)
    y_tensor = torch.zeros([a, r], dtype=output_dtype)
    gamma_tensor = torch.randn([1, r], dtype=input_dtype)
    beta_tensor = torch.randn([1, r], dtype=input_dtype)

    with profiler.profile():
        for _ in range(runs):
            layer_norm_batch[block_num](x_tensor, gamma_tensor, beta_tensor, y_tensor, unroll_factor, l2_cache, *params)

    expected = layer_norm_golden(x_tensor, gamma_tensor.view(-1), beta_tensor.view(-1), r, epsilon, output_dtype)
    atol = rtol = 2e-2 if output_dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(y_tensor, expected, atol=atol, rtol=rtol)
