# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from enum import IntEnum

from asc.experimental import asctile
import torch


class FullLoad(IntEnum):
    NONE = 0
    A = 1
    B = 2


class Transpose(IntEnum):
    NONE = 0
    L1 = 1
    L0 = 2


@asctile.jit(reuse_alloc=2)
def matmul_v3_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                     bias_ptr: asctile.GlobalAddress, m, n, k, m_L1: asctile.ConstExpr, n_L1: asctile.ConstExpr,
                     k_L1: asctile.ConstExpr, base_m: asctile.ConstExpr, base_n: asctile.ConstExpr,
                     base_k: asctile.ConstExpr, a_transp: asctile.ConstExpr, b_transp: asctile.ConstExpr,
                     full_load_mode: asctile.ConstExpr, quant_type: asctile.ConstExpr,
                     enable_hf32_mode: asctile.ConstExpr, has_bias: asctile.ConstExpr,
                     double_buffering: asctile.ConstExpr, l0c2ub: asctile.ConstExpr,
                     full_load_tile_m: asctile.ConstExpr, full_load_tile_k: asctile.ConstExpr,
                     full_load_tile_n: asctile.ConstExpr, distrib_mode: asctile.ConstExpr):
    if a_transp == Transpose.NONE:
        a_gm = asctile.global_tensor(a_ptr, [m, k])
    else:
        a_gm = asctile.global_tensor(a_ptr, [k, m])
    if b_transp == Transpose.NONE:
        b_gm = asctile.global_tensor(b_ptr, [k, n])
    else:
        b_gm = asctile.global_tensor(b_ptr, [n, k])
    c_gm = asctile.global_tensor(c_ptr, [m, n])
    if has_bias:
        bias_gm = asctile.global_tensor(bias_ptr, [n])
    m_blocks = asctile.ceildiv(m, m_L1)
    n_blocks = asctile.ceildiv(n, n_L1)
    tiles_num = m_blocks * n_blocks
    is_A_full_load = full_load_mode == FullLoad.A
    is_B_full_load = full_load_mode == FullLoad.B
    if is_A_full_load:
        tile_m = full_load_tile_m
        tile_k = full_load_tile_k
        if a_transp == Transpose.NONE:
            shape = [tile_m, tile_k]
        else:
            shape = [tile_k, tile_m]
        a_l1 = asctile.copy_in(a_gm, [0, 0], shape, location=asctile.TensorLocation.L1)
        if a_transp == Transpose.L1:
            a_l1 = a_l1.T
    elif is_B_full_load:
        tile_k = full_load_tile_k
        tile_n = full_load_tile_n
        if b_transp == Transpose.NONE:
            shape = [tile_k, tile_n]
        else:
            shape = [tile_n, tile_k]
        b_l1 = asctile.copy_in(b_gm, [0, 0], shape, location=asctile.TensorLocation.L1)
        if b_transp == Transpose.L1:
            b_l1 = b_l1.T
    tile_uf, m_uf, n_uf, k_l1_uf, k_l0_uf = double_buffering
    group_size = 4
    main_group = min(group_size, m_blocks)
    main_row = (m_blocks // main_group - 1) if m_blocks >= main_group else 0
    tail_group = m_blocks - main_row * main_group
    block_idx = asctile.block_idx()
    sub_block_num = asctile.sub_block_num()
    start_index = block_idx / sub_block_num
    for tile_idx in range(start_index, tiles_num, asctile.block_num(), unroll_factor=tile_uf):
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
        for i_aL1 in range(asctile.ceildiv(m_L1, base_m), unroll_factor=m_uf):
            m_gm_off = m_tile_off + i_aL1 * base_m
            m_l0_off = 0
            if is_A_full_load:
                m_l0_off = m_gm_off
            for j_bL1 in range(asctile.ceildiv(n_L1, base_n), unroll_factor=n_uf):
                n_gm_off = n_tile_off + j_bL1 * base_n
                n_l0_off = 0
                if is_B_full_load:
                    n_l0_off = n_gm_off
                if has_bias:
                    bias = asctile.copy_in(bias_gm, [n_gm_off], [base_n], asctile.TensorLocation.BT)
                    acc = asctile.zeros_acc([base_m, base_n], dtype=asctile.float32, bias=bias)
                else:
                    acc = asctile.zeros_acc([base_m, base_n], dtype=asctile.float32)
                for outer_k in range(asctile.ceildiv(k, k_L1), unroll_factor=k_l1_uf):
                    k_gm_off = outer_k * k_L1
                    if not is_A_full_load:
                        if a_transp == Transpose.NONE:
                            a_l1 = asctile.copy_in(a_gm, [m_gm_off, k_gm_off], [base_m, k_L1],
                                                   asctile.TensorLocation.L1)
                        else:
                            a_l1 = asctile.copy_in(a_gm, [k_gm_off, m_gm_off], [k_L1, base_m],
                                                   asctile.TensorLocation.L1)
                            if a_transp == Transpose.L1:
                                a_l1 = a_l1.T
                    if not is_B_full_load:
                        if b_transp == Transpose.NONE:
                            b_l1 = asctile.copy_in(b_gm, [k_gm_off, n_gm_off], [k_L1, base_n],
                                                   asctile.TensorLocation.L1)
                        else:
                            b_l1 = asctile.copy_in(b_gm, [n_gm_off, k_gm_off], [base_n, k_L1],
                                                   asctile.TensorLocation.L1)
                            if b_transp == Transpose.L1:
                                b_l1 = b_l1.T
                    for inner_k in range(asctile.ceildiv(k_L1, base_k), unroll_factor=k_l0_uf):
                        ka_off = inner_k * base_k
                        kb_off = inner_k * base_k
                        if is_A_full_load:
                            ka_off += k_gm_off
                        elif is_B_full_load:
                            kb_off += k_gm_off
                        if a_transp != Transpose.L0:
                            a_l0 = asctile.copy(a_l1, [m_l0_off, ka_off], [base_m, base_k], asctile.TensorLocation.L0A)
                        else:
                            a_l0 = asctile.copy(a_l1, [ka_off, m_l0_off], [base_k, base_m],
                                                asctile.TensorLocation.L0A).T
                        if b_transp != Transpose.L0:
                            b_l0 = asctile.copy(b_l1, [kb_off, n_l0_off], [base_k, base_n], asctile.TensorLocation.L0B)
                        else:
                            b_l0 = asctile.copy(b_l1, [n_l0_off, kb_off], [base_n, base_k],
                                                asctile.TensorLocation.L0B).T
                        asctile.matmul_acc(acc, a_l0, b_l0, hf32=enable_hf32_mode)
                if l0c2ub:
                    if distrib_mode is None:
                        # cast on Cube core
                        acc = acc.to(quant_type)
                    acc_ub = asctile.copy(acc, location=asctile.TensorLocation.UB, distrib=distrib_mode)
                    split_off_m = 0
                    split_off_n = 0
                    if distrib_mode is not None:
                        # cast on Vector core
                        acc_ub = acc_ub.to(quant_type)
                        if distrib_mode == asctile.DistribMode.SplitByM:
                            split_off_m = acc_ub.shape[0] * asctile.sub_block_idx()
                        if distrib_mode == asctile.DistribMode.SplitByN:
                            split_off_n = acc_ub.shape[1] * asctile.sub_block_idx()
                    asctile.copy_out(acc_ub, c_gm, offsets=[m_gm_off + split_off_m, n_gm_off + split_off_n])
                else:
                    asctile.copy_out(acc.to(quant_type), c_gm, offsets=[m_gm_off, n_gm_off])


def run_matmul_v3_test(profiler, runs, is_static, core_num, tiling_data, dtype, a_transp, b_transp, full_load_mode,
                       has_bias, double_buffering, distrib_mode, l0c2ub):
    quant_type = asctile.float32
    if dtype == torch.float16:
        quant_type = asctile.float16
    elif dtype == torch.bfloat16:
        quant_type = asctile.bfloat16
    m, n, k, m_L1, n_L1, k_L1, base_m, base_n, base_k = tiling_data
    if k_L1 > k:
        k_L1 = asctile.ceildiv(k, base_k) * base_k
    a_shape = (m, k) if a_transp == Transpose.NONE else (k, m)
    b_shape = (k, n) if b_transp == Transpose.NONE else (n, k)
    full_load_tile_m = asctile.ceildiv(m, base_m) * base_m
    full_load_tile_k = asctile.ceildiv(k, k_L1) * k_L1
    full_load_tile_n = asctile.ceildiv(n, base_n) * base_n
    enable_hf32_mode = dtype == torch.float32
    low, high = (-1, 1) if not enable_hf32_mode else (0, 1)
    a = (high - low) * torch.rand(a_shape, dtype=dtype) + low
    b = (high - low) * torch.rand(b_shape, dtype=dtype) + low
    c = torch.zeros((m, n), dtype=dtype)
    bias = (high - low) * torch.rand([n], dtype=dtype) + low
    with profiler.profile():
        for _ in range(runs):
            matmul_v3_kernel[core_num](a, b, c, bias, asctile.ConstExpr(m) if is_static else m,
                                       asctile.ConstExpr(n) if is_static else n,
                                       asctile.ConstExpr(k) if is_static else k, m_L1, n_L1, k_L1, base_m, base_n,
                                       base_k, a_transp, b_transp, full_load_mode, quant_type, enable_hf32_mode,
                                       has_bias, double_buffering, l0c2ub, full_load_tile_m, full_load_tile_k,
                                       full_load_tile_n, distrib_mode, cv_ratio=2 if distrib_mode else 1)
    if a_transp != Transpose.NONE:
        a = a.T
    if b_transp != Transpose.NONE:
        b = b.T
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    if has_bias:
        c_ref = c_ref + bias
    c_ref = c_ref.to(dtype)
    atol, rtol = (1e-3, 1e-3)
    torch.testing.assert_close(c, c_ref, atol=atol, rtol=rtol)
