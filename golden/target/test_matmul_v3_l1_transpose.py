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

from .test_matmul_v3 import FullLoadMode, test_cases


@asc2.jit(reuse_alloc=0)
def matmul_v3_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, c_ptr: asc2.GlobalAddress,
                     bias_ptr: asc2.GlobalAddress, a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr,
                     m_L1: asc2.ConstExpr, n_L1: asc2.ConstExpr, k_L1: asc2.ConstExpr, base_m: asc2.ConstExpr,
                     base_n: asc2.ConstExpr, base_k: asc2.ConstExpr, is_a_transpose_l1: asc2.ConstExpr,
                     is_b_transpose_l1: asc2.ConstExpr, full_load_mode: asc2.ConstExpr, quant_type: asc2.ConstExpr,
                     enable_hf32_mode: asc2.ConstExpr, has_bias: asc2.ConstExpr, double_buffering: asc2.ConstExpr):
    m, k = a_shape
    if is_a_transpose_l1:
        k, m = a_shape
    if not is_b_transpose_l1:
        n = b_shape[1]
    else:
        n = b_shape[0]
    a_gm = asc2.global_tensor(a_ptr, a_shape)
    b_gm = asc2.global_tensor(b_ptr, b_shape)
    c_gm = asc2.global_tensor(c_ptr, [m, n])
    if has_bias:
        bias_gm = asc2.global_tensor(bias_ptr, [n])
    m_blocks = asc2.ceildiv(m, m_L1)
    n_blocks = asc2.ceildiv(n, n_L1)
    tiles_num = m_blocks * n_blocks
    is_A_full_load = full_load_mode == FullLoadMode.A
    is_B_full_load = full_load_mode == FullLoadMode.B
    if is_A_full_load:
        tile_m = asc2.ceildiv(m, base_m) * base_m
        tile_k = asc2.ceildiv(k, k_L1) * k_L1
        if not is_a_transpose_l1:
            shape = [tile_m, tile_k]
            a_l1 = asc2.copy_in(a_gm, [0, 0], shape, location=asc2.TensorLocation.L1)
        else:
            shape = [tile_k, tile_m]
            a_l1 = asc2.copy_in(a_gm, [0, 0], shape, location=asc2.TensorLocation.L1).T
    elif is_B_full_load:
        tile_k = asc2.ceildiv(k, k_L1) * k_L1
        tile_n = asc2.ceildiv(n, base_n) * base_n
        if not is_b_transpose_l1:
            shape = [tile_k, tile_n]
            b_l1 = asc2.copy_in(b_gm, [0, 0], shape, location=asc2.TensorLocation.L1)
        else:
            shape = [tile_n, tile_k]
            b_l1 = asc2.copy_in(b_gm, [0, 0], shape, location=asc2.TensorLocation.L1).T
    tile_uf, m_uf, n_uf, k_l1_uf, k_l0_uf = double_buffering
    group_size = 4
    main_group = min(group_size, m_blocks)
    main_row = (m_blocks // main_group - 1) if m_blocks >= main_group else 0
    tail_group = m_blocks - main_row * main_group
    for tile_id in range(asc2.block_idx(), tiles_num, asc2.block_num(), unroll_factor=tile_uf):
        tile_idx = tile_id % tiles_num
        row_idx = tile_idx // n_blocks // main_group
        m_idx = row_idx * main_group + tile_idx % main_group
        n_idx = (tile_idx // main_group) % n_blocks
        if row_idx >= main_row:
            tail_index = tile_idx - main_row * main_group * n_blocks
            m_idx = main_row * main_group + tail_index % tail_group
            n_idx = (tail_index // tail_group) % n_blocks
            row_idx = m_idx // main_group
        if row_idx % 2 != 0:
            n_idx = n_blocks - 1 - n_idx
        m_tile_off = m_L1 * m_idx
        n_tile_off = n_L1 * n_idx
        for i_aL1 in range(asc2.ceildiv(m_L1, base_m), unroll_factor=m_uf):
            m_gm_off = m_tile_off + i_aL1 * base_m
            m_l0_off = 0
            if is_A_full_load:
                m_l0_off = m_gm_off
            for j_bL1 in range(asc2.ceildiv(n_L1, base_n), unroll_factor=n_uf):
                n_gm_off = n_tile_off + j_bL1 * base_n
                n_l0_off = 0
                if is_B_full_load:
                    n_l0_off = n_gm_off
                if has_bias:
                    bias = asc2.copy_in(bias_gm, [n_gm_off], [base_n], asc2.TensorLocation.BT)
                    acc = asc2.zeros_acc([base_m, base_n], dtype=asc2.float32, bias=bias)
                else:
                    acc = asc2.zeros_acc([base_m, base_n], dtype=asc2.float32)
                for outer_k in range(asc2.ceildiv(k, k_L1), unroll_factor=k_l1_uf):
                    k_gm_off = outer_k * k_L1
                    if not is_A_full_load:
                        if not is_a_transpose_l1:
                            a_l1 = asc2.copy_in(a_gm, [m_gm_off, k_gm_off], [base_m, k_L1], asc2.TensorLocation.L1)
                        else:
                            a_l1 = asc2.copy_in(a_gm, [k_gm_off, m_gm_off], [k_L1, base_m], asc2.TensorLocation.L1).T
                    if not is_B_full_load:
                        if not is_b_transpose_l1:
                            b_l1 = asc2.copy_in(b_gm, [k_gm_off, n_gm_off], [k_L1, base_n], asc2.TensorLocation.L1)
                        else:
                            b_l1 = asc2.copy_in(b_gm, [n_gm_off, k_gm_off], [base_n, k_L1], asc2.TensorLocation.L1).T
                    for inner_k in range(asc2.ceildiv(k_L1, base_k), unroll_factor=k_l0_uf):
                        ka_off = inner_k * base_k
                        kb_off = inner_k * base_k
                        if is_A_full_load:
                            ka_off += k_gm_off
                        elif is_B_full_load:
                            kb_off += k_gm_off
                        a_l0 = asc2.copy(a_l1, [m_l0_off, ka_off], [base_m, base_k], asc2.TensorLocation.L0A)
                        b_l0 = asc2.copy(b_l1, [kb_off, n_l0_off], [base_k, base_n], asc2.TensorLocation.L0B)
                        asc2.matmul_acc(acc, a_l0, b_l0, hf32=enable_hf32_mode)
                asc2.copy_out(acc.to(quant_type), c_gm, offsets=[m_gm_off, n_gm_off])


@pytest.mark.parametrize(
    "core_num, tiling_data, dtype, is_a_transpose_l1, is_b_transpose_l1, full_load_mode, enable_hf32_mode, has_bias, double_buffering, input_range, accuracy",
    test_cases, ids=["_".join(map(str, tc[1][:3])) for tc in test_cases])
def test_matmul_v3(profiler, runs, core_num, tiling_data, dtype, is_a_transpose_l1, is_b_transpose_l1, full_load_mode,
                   enable_hf32_mode, has_bias, double_buffering, input_range, accuracy):
    quant_type = asc2.float32
    if dtype == torch.float16:
        quant_type = asc2.float16
    elif dtype == torch.bfloat16:
        quant_type = asc2.bfloat16
    m, n, k, m_L1, n_L1, k_L1, base_m, base_n, base_k = tiling_data
    a_shape = (m, k) if not is_a_transpose_l1 else (k, m)
    b_shape = (k, n) if not is_b_transpose_l1 else (n, k)
    low, high = input_range
    a = (high - low) * torch.rand(a_shape, dtype=dtype) + low
    b = (high - low) * torch.rand(b_shape, dtype=dtype) + low
    c = torch.zeros((m, n), dtype=dtype)
    bias = (high - low) * torch.rand([n], dtype=dtype) + low
    with profiler.profile():
        for _ in range(runs):
            matmul_v3_kernel[core_num](a, b, c, bias, a.shape, b.shape, m_L1, n_L1, k_L1, base_m, base_n, base_k,
                                       is_a_transpose_l1, is_b_transpose_l1, full_load_mode, quant_type,
                                       enable_hf32_mode, has_bias, double_buffering)
    if is_a_transpose_l1:
        a = a.T
    if is_b_transpose_l1:
        b = b.T
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    if has_bias:
        c_ref = c_ref + bias
    c_ref = c_ref.to(dtype)
    atol, rtol = accuracy
    torch.testing.assert_close(c, c_ref, atol=atol, rtol=rtol)
