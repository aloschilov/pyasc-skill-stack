# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asctile
import pytest
import torch


@asctile.jit
def matmul_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                  a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr, single_core_m: asctile.ConstExpr,
                  single_core_n: asctile.ConstExpr, step_ka: asctile.ConstExpr, step_kb: asctile.ConstExpr,
                  base_k: asctile.ConstExpr, quant_type: asctile.ConstExpr, unroll_factor: asctile.ConstExpr):
    m, k = a_shape
    _, n = b_shape
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, [m, n])
    acc = asctile.zeros_acc([single_core_m, single_core_n], dtype=asctile.float32)
    block_idx = asctile.block_idx()
    n_blocks = asctile.ceildiv(n, single_core_n)
    m_off = single_core_m * (block_idx / n_blocks)
    n_off = single_core_n * (block_idx % n_blocks)
    for k_outer in range(asctile.ceildiv(k, step_kb), unroll_factor=unroll_factor):
        b_l1 = asctile.copy_in(b_gm, [k_outer * step_kb, n_off], [step_kb, single_core_n], asctile.TensorLocation.L1)
        for k_mid in range(asctile.ceildiv(step_kb, step_ka), unroll_factor=unroll_factor):
            k_off = k_outer * step_kb + k_mid * step_ka
            a_l1 = asctile.copy_in(a_gm, [m_off, k_off], [single_core_m, step_ka], asctile.TensorLocation.L1)
            for k_l0 in range(asctile.ceildiv(step_ka, base_k), unroll_factor=unroll_factor):
                a_l0 = asctile.copy(a_l1, [0, k_l0 * base_k], [single_core_m, base_k], asctile.TensorLocation.L0A)
                b_l0 = asctile.copy(b_l1, [k_mid * step_ka + k_l0 * base_k, 0], [base_k, single_core_n],
                                    asctile.TensorLocation.L0B)
                asctile.matmul_acc(acc, a_l0, b_l0)
    acc = acc.to(quant_type)
    asctile.copy_out(acc, c_gm, [m_off, n_off])


@pytest.mark.parametrize("block_num, unroll_factor, input_type, output_type, tiling_data", [
    (16, 2, torch.float16, torch.float16, (128, 784, 832, 32, 208, 16, 784, 16)),
    (16, 2, torch.float32, torch.float32, (1024, 64, 16, 64, 16, 16, 64, 16)),
])
def test_matmul_k_tiled(profiler, runs, block_num, unroll_factor, input_type, output_type, tiling_data):
    quant_type = asctile.float32
    if output_type == torch.float16:
        quant_type = asctile.float16
    elif output_type == torch.bfloat16:
        quant_type = asctile.bfloat16
    m, k, n, single_core_m, single_core_n, step_ka, step_kb, base_k = tiling_data
    a = (torch.rand((m, k), dtype=input_type))
    b = (torch.rand((k, n), dtype=input_type))
    c = torch.zeros((m, n), dtype=output_type)
    with profiler.profile():
        for _ in range(runs):
            matmul_kernel[block_num](a, b, c, a.shape, b.shape, single_core_m, single_core_n, step_ka, step_kb, base_k,
                                     quant_type, unroll_factor)
    c_ref = (a.to(torch.float32) @ b.to(torch.float32)).to(output_type)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)
