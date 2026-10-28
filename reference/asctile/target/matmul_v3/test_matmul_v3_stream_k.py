# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.experimental import asctile
import pytest
import torch

from . import Transpose


@asctile.jit(reuse_alloc=2)
def mm3_streamk_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                       bias_ptr: asctile.GlobalAddress, workspace_ptr: asctile.GlobalAddress,
                       a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr, m_L1: asctile.ConstExpr,
                       n_L1: asctile.ConstExpr, k_L1: asctile.ConstExpr, base_m: asctile.ConstExpr,
                       base_n: asctile.ConstExpr, base_k: asctile.ConstExpr, sk_single_k: asctile.ConstExpr,
                       used_core_num: asctile.ConstExpr, quant_type: asctile.ConstExpr, enable_hf32: asctile.ConstExpr,
                       has_bias: asctile.ConstExpr, double_buffering: asctile.ConstExpr, a_transp: asctile.ConstExpr,
                       b_transp: asctile.ConstExpr):
    if a_transp == Transpose.NONE:
        m, k = a_shape
    else:
        k, m = a_shape
    if b_transp == Transpose.NONE:
        _, n = b_shape
    else:
        n, _ = b_shape

    m_tiles = asctile.ceildiv(m, m_L1)
    n_tiles = asctile.ceildiv(n, n_L1)
    mn_tiles = m_tiles * n_tiles
    sk_mn_tiles = mn_tiles % used_core_num
    sk_k_tiles = asctile.ceildiv(k, sk_single_k)
    dp_tiles = mn_tiles - sk_mn_tiles
    sk_total_tiles = sk_mn_tiles * sk_k_tiles

    c_gm = asctile.global_tensor(c_ptr, (m, n))
    ws_shape = (sk_total_tiles * m_L1, n_L1) if sk_total_tiles > 0 else (1, 1)
    workspace_gm = asctile.global_tensor(workspace_ptr, ws_shape)
    if has_bias:
        bias_gm = asctile.global_tensor(bias_ptr, [n])

    #############
    # Cube part #
    #############
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    tile_uf, m_uf, n_uf, k_l1_uf, k_l0_uf = double_buffering

    group_size = 4
    main_group = min(group_size, m_tiles)
    main_row = (m_tiles // main_group - 1) if m_tiles > main_group else 0
    tail_group = m_tiles - main_row * main_group

    total_tiles = dp_tiles + sk_total_tiles
    for tile_idx in range(asctile.block_idx(), total_tiles, asctile.block_num(), unroll_factor=tile_uf):
        # TODO: Switch last DP step and SK step (optimization improvement for large tensors)
        is_dp = dp_tiles > 0 and tile_idx < dp_tiles
        sk_mn_idx = 0 if is_dp else (tile_idx - dp_tiles) // sk_k_tiles
        sk_k_idx = 0 if is_dp else (tile_idx - dp_tiles) % sk_k_tiles
        mn_idx = tile_idx if is_dp else dp_tiles + sk_mn_idx

        row_idx = mn_idx // n_tiles // main_group
        m_idx = row_idx * main_group + mn_idx % main_group
        n_idx = (mn_idx // main_group) % n_tiles
        if row_idx >= main_row:
            tail_index = mn_idx - main_row * main_group * n_tiles
            m_idx = main_row * main_group + tail_index % tail_group
            n_idx = (tail_index // tail_group) % n_tiles
            row_idx = m_idx // main_group
        if row_idx % 2 != 0:
            n_idx = n_tiles - 1 - n_idx

        m_off = m_idx * m_L1
        n_off = n_idx * n_L1

        for i_aL1 in range(asctile.ceildiv(m_L1, base_m), unroll_factor=m_uf):
            for j_bL1 in range(asctile.ceildiv(n_L1, base_n), unroll_factor=n_uf):
                m_gm_off = m_off + i_aL1 * base_m
                n_gm_off = n_off + j_bL1 * base_n

                if is_dp:
                    # Data parallel (DP) tile: full K reduction, write to output tensor directly
                    if has_bias:
                        bias = asctile.copy_in(bias_gm, [n_gm_off], [base_n], asctile.TensorLocation.BT)
                        acc = asctile.zeros_acc([base_m, base_n], dtype=asctile.float32, bias=bias)
                    else:
                        acc = asctile.zeros_acc([base_m, base_n], dtype=asctile.float32)
                    for k_outer in range(asctile.ceildiv(k, k_L1), unroll_factor=k_l1_uf):
                        k_gm_off = k_outer * k_L1
                        if a_transp == Transpose.NONE:
                            a_l1 = asctile.copy_in(a_gm, [m_gm_off, k_gm_off], [base_m, k_L1], "L1")
                        else:
                            a_l1 = asctile.copy_in(a_gm, [k_gm_off, m_gm_off], [k_L1, base_m], "L1")
                        if b_transp == Transpose.NONE:
                            b_l1 = asctile.copy_in(b_gm, [k_gm_off, n_gm_off], [k_L1, base_n], "L1")
                        else:
                            b_l1 = asctile.copy_in(b_gm, [n_gm_off, k_gm_off], [base_n, k_L1], "L1")
                        for k_inner in range(asctile.ceildiv(k_L1, base_k), unroll_factor=k_l0_uf):
                            if a_transp == Transpose.NONE:
                                a_l0 = asctile.copy(a_l1, [0, k_inner * base_k], [base_m, base_k], "L0A")
                            else:
                                a_l0 = asctile.copy(a_l1, [k_inner * base_k, 0], [base_k, base_m], "L0A").T
                            if b_transp == Transpose.NONE:
                                b_l0 = asctile.copy(b_l1, [k_inner * base_k, 0], [base_k, base_n], "L0B")
                            else:
                                b_l0 = asctile.copy(b_l1, [0, k_inner * base_k], [base_n, base_k], "L0B").T
                            asctile.matmul_acc(acc, a_l0, b_l0, hf32=enable_hf32)
                    asctile.copy_out(acc.to(quant_type), c_gm, [m_gm_off, n_gm_off])
                else:
                    # StreamK (SK) tile: partial K reduction, write to temporary workspace tensor

                    # Initialize accumulator without bias. It will be added on Vector
                    acc = asctile.zeros_acc([base_m, base_n], dtype=asctile.float32)
                    k_start = sk_k_idx * sk_single_k
                    if k % sk_single_k == 0 or k - k_start >= sk_single_k:
                        k_steps = sk_single_k // base_k
                        for k_step in range(k_steps, unroll_factor=k_l0_uf):
                            k_off = k_start + k_step * base_k
                            if a_transp == Transpose.NONE:
                                a_l0 = asctile.copy_in(a_gm, [m_gm_off, k_off], [base_m, base_k], "L0A")
                            else:
                                a_l0 = asctile.copy_in(a_gm, [k_off, m_gm_off], [base_k, base_m], "L0A").T
                            if b_transp == Transpose.NONE:
                                b_l0 = asctile.copy_in(b_gm, [k_off, n_gm_off], [base_k, base_n], "L0B")
                            else:
                                b_l0 = asctile.copy_in(b_gm, [n_gm_off, k_off], [base_n, base_k], "L0B").T
                            asctile.matmul_acc(acc, a_l0, b_l0, hf32=enable_hf32)
                        tail_k = sk_single_k % base_k
                        if tail_k > 0:
                            k_off = k_start + k_steps * base_k
                            if a_transp == Transpose.NONE:
                                a_l0 = asctile.copy_in(a_gm, [m_gm_off, k_off], [base_m, tail_k], "L0A")
                            else:
                                a_l0 = asctile.copy_in(a_gm, [k_off, m_gm_off], [tail_k, base_m], "L0A").T
                            if b_transp == Transpose.NONE:
                                b_l0 = asctile.copy_in(b_gm, [k_off, n_gm_off], [tail_k, base_n], "L0B")
                            else:
                                b_l0 = asctile.copy_in(b_gm, [n_gm_off, k_off], [base_n, tail_k], "L0B").T
                            asctile.matmul_acc(acc, a_l0, b_l0, hf32=enable_hf32)
                    else:
                        sk_single_tail_k = k % sk_single_k
                        k_steps = sk_single_tail_k // base_k
                        for k_step in range(k_steps, unroll_factor=k_l0_uf):
                            k_off = k_start + k_step * base_k
                            if a_transp == Transpose.NONE:
                                a_l0 = asctile.copy_in(a_gm, [m_gm_off, k_off], [base_m, base_k], "L0A")
                            else:
                                a_l0 = asctile.copy_in(a_gm, [k_off, m_gm_off], [base_k, base_m], "L0A").T
                            if b_transp == Transpose.NONE:
                                b_l0 = asctile.copy_in(b_gm, [k_off, n_gm_off], [base_k, base_n], "L0B")
                            else:
                                b_l0 = asctile.copy_in(b_gm, [n_gm_off, k_off], [base_n, base_k], "L0B").T
                            asctile.matmul_acc(acc, a_l0, b_l0, hf32=enable_hf32)
                        tail_k = sk_single_tail_k % base_k
                        if tail_k > 0:
                            k_off = k_start + k_steps * base_k
                            if a_transp == Transpose.NONE:
                                a_l0 = asctile.copy_in(a_gm, [m_gm_off, k_off], [base_m, tail_k], "L0A")
                            else:
                                a_l0 = asctile.copy_in(a_gm, [k_off, m_gm_off], [tail_k, base_m], "L0A").T
                            if b_transp == Transpose.NONE:
                                b_l0 = asctile.copy_in(b_gm, [k_off, n_gm_off], [tail_k, base_n], "L0B")
                            else:
                                b_l0 = asctile.copy_in(b_gm, [n_gm_off, k_off], [base_n, tail_k], "L0B").T
                            asctile.matmul_acc(acc, a_l0, b_l0, hf32=enable_hf32)

                    ws_row = (tile_idx - dp_tiles) * m_L1 + i_aL1 * base_m
                    ws_col = j_bL1 * base_n
                    asctile.copy_out(acc.to(asctile.float32), workspace_gm, [ws_row, ws_col])

    ###############
    # Vector part #
    ###############
    task_ration = 2
    ub_src_gap_unit = 32
    if sk_mn_tiles > 0:
        asc.sync_all()
        # TODO: Process tail m_L1, n_L1 (optimization improvement)
        # TODO: Align n_L1 for 32b if copy L0C->UB->GM was used

        v_tile_idx = asctile.block_idx() // (sk_k_tiles * task_ration)
        v_m_burst_idx = asctile.block_idx() % (sk_k_tiles * task_ration)
        v_mn_idx = v_tile_idx + dp_tiles
        v_row_idx = v_mn_idx // n_tiles // main_group
        v_m_idx = v_row_idx * main_group + v_mn_idx % main_group
        v_n_idx = (v_mn_idx // main_group) % n_tiles
        if v_row_idx >= main_row:
            tail_index = v_mn_idx - main_row * main_group * n_tiles
            v_m_idx = main_row * main_group + tail_index % tail_group
            v_n_idx = (tail_index // tail_group) % n_tiles
            v_row_idx = v_m_idx // main_group
        if v_row_idx % 2 != 0:
            v_n_idx = n_tiles - 1 - v_n_idx
        m_per_vec_unit = asctile.ceildiv(m_L1, sk_k_tiles * task_ration)
        n_aligned = asctile.ceildiv(ub_src_gap_unit, n_L1)
        m_burst_base = asctile.ceildiv(m_per_vec_unit, n_aligned) * n_aligned
        m_burst_count = asctile.ceildiv(m_L1, m_burst_base)
        m_burst_tail = m_L1 - (m_burst_count - 1) * m_burst_base
        v_m_gm_off = v_m_idx * m_L1 + v_m_burst_idx * m_burst_base
        v_n_gm_off = v_n_idx * n_L1
        v_ws_row_off = v_tile_idx * sk_k_tiles * m_L1 + v_m_burst_idx * m_burst_base
        if v_tile_idx < sk_mn_tiles and v_m_burst_idx < m_burst_count:
            if m_burst_tail == m_burst_base or v_m_burst_idx < m_burst_count - 1:
                sum_tile = asctile.copy_in(workspace_gm, [v_ws_row_off, 0], [m_burst_base, n_L1])
                for k_idx in range(1, sk_k_tiles):
                    partial = asctile.copy_in(workspace_gm, [v_ws_row_off + k_idx * m_L1, 0], [m_burst_base, n_L1])
                    sum_tile = sum_tile + partial
                if has_bias:
                    bias = asctile.copy_in(bias_gm, [v_n_gm_off], [n_L1])
                    bias_2d = asctile.broadcast_to(bias, [m_burst_base, n_L1])
                    sum_tile = sum_tile + bias_2d
                asctile.copy_out(sum_tile.to(quant_type), c_gm, [v_m_gm_off, v_n_gm_off])
            else:
                sum_tile = asctile.copy_in(workspace_gm, [v_ws_row_off, 0], [m_burst_tail, n_L1])
                for k_idx in range(1, sk_k_tiles):
                    partial = asctile.copy_in(workspace_gm, [v_ws_row_off + k_idx * m_L1, 0], [m_burst_tail, n_L1])
                    sum_tile = sum_tile + partial
                if has_bias:
                    bias = asctile.copy_in(bias_gm, [v_n_gm_off], [n_L1])
                    bias_2d = asctile.broadcast_to(bias, [m_burst_tail, n_L1])
                    sum_tile = sum_tile + bias_2d
                asctile.copy_out(sum_tile.to(quant_type), c_gm, [v_m_gm_off, v_n_gm_off])


test_param_str = "core_num, tiling_data, dtype, a_transp, b_transp, has_bias, double_buffering"
test_cases = [
    (None, (4, 128, 8192, 16, 128, 256, 16, 128, 128, 228), torch.float16, Transpose.NONE, Transpose.L0, False,
     (1, 1, 1, 2, 2)),
    (None, (1, 1152, 6912, 16, 240, 256, 16, 240, 64, 988), torch.float16, Transpose.NONE, Transpose.NONE, False,
     (1, 1, 1, 2, 2)),
    (None, (231, 1280, 5120, 240, 256, 256, 240, 256, 64, 732), torch.float16, Transpose.NONE, Transpose.L0, True,
     (1, 1, 1, 2, 2)),
    (None, (16, 640, 12288, 16, 224, 256, 16, 224, 64, 1024), torch.float16, Transpose.NONE, Transpose.L0, False,
     (1, 1, 1, 2, 2)),
    (None, (640, 16, 12288, 224, 16, 256, 224, 16, 64, 1024), torch.float16, Transpose.NONE, Transpose.L0, False,
     (1, 1, 1, 2, 2)),
    (None, (4, 1024, 8192, 16, 256, 256, 16, 256, 64, 911), torch.float16, Transpose.NONE, Transpose.L0, False,
     (1, 1, 1, 2, 2)),
    (None, (32, 1152, 7680, 32, 240, 128, 32, 240, 64, 1098), torch.float16, True, Transpose.NONE, Transpose.NONE,
     (1, 1, 1, 2, 2)),
    (None, (32, 2112, 7168, 32, 240, 256, 32, 240, 64, 1792), torch.float16, False, Transpose.NONE, Transpose.L0,
     (1, 1, 1, 2, 2)),
    (None, (64, 2112, 7168, 64, 240, 256, 64, 240, 64, 1792), torch.float16, False, Transpose.NONE, Transpose.L0,
     (1, 1, 1, 2, 2)),
    (None, (16, 2304, 7168, 16, 256, 256, 16, 256, 64, 1792), torch.float16, False, Transpose.NONE, Transpose.L0,
     (1, 1, 1, 2, 2)),
    (None, (4, 3584, 8192, 16, 208, 256, 16, 208, 64, 4096), torch.float16, Transpose.NONE, Transpose.L0, False,
     (1, 1, 1, 2, 2)),
    (None, (4096, 256, 7168, 240, 256, 256, 240, 256, 64, 3584), torch.bfloat16, Transpose.NONE, Transpose.L0, False,
     (1, 1, 1, 2, 2)),
    (None, (4096, 128, 8192, 240, 128, 256, 240, 128, 64, 4096), torch.float16, Transpose.NONE, Transpose.L0, False,
     (1, 1, 1, 2, 2)),
    (None, (2048, 1280, 10240, 256, 256, 256, 256, 256, 64, 1138), torch.float16, Transpose.NONE, Transpose.NONE, False,
     (1, 1, 1, 2, 2)),
    (None, (120, 4096, 10240, 128, 240, 256, 128, 240, 64, 5120), torch.float16, Transpose.NONE, Transpose.L0, False,
     (1, 1, 1, 2, 2)),
    (None, (4096, 1280, 8192, 256, 256, 256, 256, 256, 64, 2048), torch.bfloat16, Transpose.NONE, Transpose.L0, False,
     (1, 1, 1, 2, 2)),
    (None, (4096, 1280, 10240, 256, 256, 256, 256, 256, 64, 2560), torch.float16, Transpose.NONE, Transpose.NONE, False,
     (1, 1, 1, 2, 2)),
    (None, (448, 256, 384000, 224, 256, 64, 224, 256, 32, 21334), torch.float32, True, Transpose.NONE, Transpose.NONE,
     (1, 1, 1, 2, 2)),
    (None, (16640, 10240, 4096, 256, 256, 256, 256, 256, 64, 1024), torch.float16, Transpose.NONE, Transpose.NONE,
     False, (1, 1, 1, 2, 2)),
    (None, (4096, 129280, 7168, 256, 256, 256, 256, 256, 64, 3584), torch.bfloat16, Transpose.NONE, Transpose.L0, False,
     (1, 1, 1, 2, 2)),
]


@pytest.mark.parametrize(test_param_str, test_cases,
                         ids=["_".join(map(str,
                                           getattr(tc, "values", tc)[1][:3])) for tc in test_cases])
def test_streamk_matmul(profiler, runs, core_num, tiling_data, dtype, a_transp, b_transp, has_bias, double_buffering):
    quant_type = asctile.float32
    if dtype == torch.float16:
        quant_type = asctile.float16
    elif dtype == torch.bfloat16:
        quant_type = asctile.bfloat16
    m, n, k, m_L1, n_L1, k_L1, base_m, base_n, base_k, sk_single_k = tiling_data
    enable_hf32_mode = dtype == torch.float32
    low, high = (-1, 1) if not enable_hf32_mode else (0, 1)
    a_shape = (m, k) if a_transp == Transpose.NONE else (k, m)
    b_shape = (k, n) if b_transp == Transpose.NONE else (n, k)
    a = (high - low) * torch.rand(a_shape, dtype=dtype) + low
    b = (high - low) * torch.rand(b_shape, dtype=dtype) + low
    c = torch.zeros((m, n), dtype=dtype)
    bias = (high - low) * torch.rand([n], dtype=dtype) + low

    if core_num is None:
        core_num = 36
    m_tiles = asctile.ceildiv(m, m_L1)
    n_tiles = asctile.ceildiv(n, n_L1)
    total_mn_tiles = m_tiles * n_tiles
    sk_mn_tiles = total_mn_tiles % core_num
    sk_k_tiles = asctile.ceildiv(k, sk_single_k)
    ws_shape = (sk_mn_tiles * sk_k_tiles * m_L1, n_L1) if sk_mn_tiles > 0 else (1, 1)
    workspace = 3 * torch.ones(ws_shape, dtype=torch.float32)

    with profiler.profile():
        for _ in range(runs):
            mm3_streamk_kernel[core_num](a, b, c, bias, workspace, a.shape, b.shape, m_L1, n_L1, k_L1, base_m, base_n,
                                         base_k, sk_single_k, core_num, quant_type, enable_hf32_mode, has_bias,
                                         double_buffering, a_transp, b_transp)
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
