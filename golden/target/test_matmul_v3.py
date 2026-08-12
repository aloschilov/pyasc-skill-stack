# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from enum import IntEnum

import asc2
import pytest
import torch


class FullLoadMode(IntEnum):
    NONE = 0
    A = 1
    B = 2


@asc2.jit(reuse_alloc=0)
def matmul_v3_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, c_ptr: asc2.GlobalAddress,
                     bias_ptr: asc2.GlobalAddress, a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr,
                     m_L1: asc2.ConstExpr, n_L1: asc2.ConstExpr, k_L1: asc2.ConstExpr, base_m: asc2.ConstExpr,
                     base_n: asc2.ConstExpr, base_k: asc2.ConstExpr, is_a_transpose_l0: asc2.ConstExpr,
                     is_b_transpose_l0: asc2.ConstExpr, full_load_mode: asc2.ConstExpr, quant_type: asc2.ConstExpr,
                     enable_hf32_mode: asc2.ConstExpr, has_bias: asc2.ConstExpr, double_buffering: asc2.ConstExpr):
    m, k = a_shape
    if is_a_transpose_l0:
        k, m = a_shape
    if not is_b_transpose_l0:
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
        if not is_a_transpose_l0:
            shape = [tile_m, tile_k]
        else:
            shape = [tile_k, tile_m]
        a_l1 = asc2.copy_in(a_gm, [0, 0], shape, location=asc2.TensorLocation.L1)
    elif is_B_full_load:
        tile_k = asc2.ceildiv(k, k_L1) * k_L1
        tile_n = asc2.ceildiv(n, base_n) * base_n
        if not is_b_transpose_l0:
            shape = [tile_k, tile_n]
        else:
            shape = [tile_n, tile_k]
        b_l1 = asc2.copy_in(b_gm, [0, 0], shape, location=asc2.TensorLocation.L1)
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
                        if not is_a_transpose_l0:
                            a_l1 = asc2.copy_in(a_gm, [m_gm_off, k_gm_off], [base_m, k_L1], asc2.TensorLocation.L1)
                        else:
                            a_l1 = asc2.copy_in(a_gm, [k_gm_off, m_gm_off], [k_L1, base_m], asc2.TensorLocation.L1)
                    if not is_B_full_load:
                        if not is_b_transpose_l0:
                            b_l1 = asc2.copy_in(b_gm, [k_gm_off, n_gm_off], [k_L1, base_n], asc2.TensorLocation.L1)
                        else:
                            b_l1 = asc2.copy_in(b_gm, [n_gm_off, k_gm_off], [base_n, k_L1], asc2.TensorLocation.L1)
                    for inner_k in range(asc2.ceildiv(k_L1, base_k), unroll_factor=k_l0_uf):
                        ka_off = inner_k * base_k
                        kb_off = inner_k * base_k
                        if is_A_full_load:
                            ka_off += k_gm_off
                        elif is_B_full_load:
                            kb_off += k_gm_off
                        if not is_a_transpose_l0:
                            a_l0 = asc2.copy(a_l1, [m_l0_off, ka_off], [base_m, base_k], asc2.TensorLocation.L0A)
                        else:
                            a_l0 = asc2.copy(a_l1, [ka_off, m_l0_off], [base_k, base_m], asc2.TensorLocation.L0A).T
                        if not is_b_transpose_l0:
                            b_l0 = asc2.copy(b_l1, [kb_off, n_l0_off], [base_k, base_n], asc2.TensorLocation.L0B)
                        else:
                            b_l0 = asc2.copy(b_l1, [n_l0_off, kb_off], [base_n, base_k], asc2.TensorLocation.L0B).T
                        asc2.matmul_acc(acc, a_l0, b_l0, hf32=enable_hf32_mode)
                asc2.copy_out(acc.to(quant_type), c_gm, offsets=[m_gm_off, n_gm_off])


test_cases = [
    (1, (16, 16, 64, 16, 16, 64, 16, 16, 64), torch.float32, False, True, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (16, (256, 1, 13, 16, 16, 16, 16, 16, 16), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (4, (32, 64, 64, 16, 32, 64, 16, 32, 64), torch.float32, False, False, FullLoadMode.NONE, True, True,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (15, (231, 128, 4, 16, 128, 16, 16, 128, 16), torch.float16, False, True, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),
    (16, (1024, 16, 16, 64, 16, 16, 64, 16, 16), torch.bfloat16, True, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),
    (20, (32, 160, 160, 16, 16, 160, 16, 16, 160), torch.float16, False, True, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),
    (32, (256, 128, 32, 32, 32, 32, 32, 32, 32), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 2, 1, 2), (0, 1), (1e-3, 1e-3)),
    (32, (256, 255, 32, 32, 64, 32, 32, 64, 32), torch.float32, False, True, FullLoadMode.NONE, True, True,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (2, (32, 16, 2048, 16, 16, 2048, 16, 16, 1024), torch.float16, False, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (26, (784, 64, 64, 64, 32, 64, 64, 32, 64), torch.bfloat16, True, True, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),
    (4, (16, 128, 800, 16, 32, 512, 16, 32, 128), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (16, 128, 800, 16, 32, 512, 16, 32, 256)
    (20, (32, 640, 160, 32, 32, 160, 32, 32, 160), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (26, (96, 784, 128, 48, 64, 128, 48, 64, 128), torch.float16, False, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),
    (16, (256, 1, 1024, 16, 16, 1024, 16, 16, 512), torch.float32, False, False, FullLoadMode.NONE, True, True,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (35, (160, 640, 240, 32, 96, 256, 32, 96, 128), torch.bfloat16, False, True, FullLoadMode.NONE, False, True,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),
    (8, (16, 256, 1168, 16, 32, 320, 16, 32, 80), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (16, 256, 1168, 16, 32, 512, 16, 32, 256)
    (4, (240, 16, 1280, 64, 16, 256, 64, 16, 64), torch.float16, True, True, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (240, 16, 1280, 64, 16, 512, 64, 16, 256)
    (32, (1024, 160, 160, 64, 80, 160, 64, 80, 160), torch.float16, False, True, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),
    (35, (6656, 64, 4, 192, 64, 16, 192, 64, 16), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (24, (16, 768, 640, 16, 32, 512, 16, 32, 128), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (16, 768, 640, 16, 32, 512, 16, 32, 256)
    (27, (5120, 40, 80, 192, 48, 80, 192, 48, 80), torch.bfloat16, True, False, FullLoadMode.NONE, False, True,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (35, (6050, 1, 128, 176, 16, 128, 176, 16, 32), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (32, (256, 512, 1024, 64, 64, 256, 64, 64, 64), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (256, 512, 1024, 64, 64, 256, 64, 64, 128)
    (33, (1024, 160, 640, 96, 64, 160, 96, 64, 80), torch.float16, False, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (1024, 160, 640, 96, 64, 256, 96, 64, 128)
    (None, (384, 768, 768, 128, 64, 128, 128, 64, 64), torch.float16, False, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (384, 768, 768, 128, 64, 256, 128, 64, 128)
    (None, (6400, 128, 96, 160, 128, 96, 160, 128, 32), torch.float32, False, False, FullLoadMode.B, True, False,
     (1, 1, 1, 1, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (2, 1, 1, 1, 2)
    (None, (120, 1152, 1152, 64, 64, 256, 64, 64, 64), torch.float16, False, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (120, 1152, 1152, 64, 64, 512, 64, 64, 256)
    (None, (1, 16384, 100, 16, 256, 64, 16, 64, 32), torch.float32, False, False, FullLoadMode.A, True, True,
     (1, 1, 2, 1, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (1, 16384, 100, 16, 352, 48, 16, 352, 16)
    (32, (128, 2048, 1024, 128, 64, 256, 128, 64, 64), torch.float16, False, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (128, 2048, 1024, 128, 64, 256, 128, 64, 128)
    (None, (1536, 512, 1024, 176, 128, 256, 176, 128, 32), torch.float16, False, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (1536, 512, 1024, 176, 128, 256, 176, 128, 64)
    (24, (3072, 16, 1280, 128, 16, 384, 128, 16, 64), torch.float16, True, True, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (3072, 16, 1280, 128, 16, 256, 128, 16, 128)
    (None, (12288, 4, 640, 256, 16, 192, 128, 16, 32), torch.float16, False, True, FullLoadMode.B, False, False,
     (1, 1, 1, 4, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (12288, 4, 640, 256, 16, 128, 256, 16, 64)
    (None, (12288, 16, 640, 256, 16, 192, 128, 16, 32), torch.float16, True, True, FullLoadMode.B, False, False,
     (1, 1, 1, 4, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (12288, 16, 640, 256, 16, 128, 256, 16, 64)
    (None, (65536, 128, 128, 256, 128, 32, 128, 128, 16), torch.float32, False, True, FullLoadMode.B, True, True,
     (1, 1, 1, 4, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (65536, 128, 128, 256, 128, 64, 256, 128, 16)
    (None, (5064, 2048, 2048, 256, 256, 256, 256, 256, 32), torch.bfloat16, False, True, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (5064, 2048, 2048, 256, 256, 256, 256, 256, 64)
    (None, (4608, 7382, 384, 256, 256, 192, 128, 128, 32), torch.float32, False, True, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (4608, 7382, 384, 256, 256, 128, 256, 256, 32)
    (None, (5064, 5632, 2048, 256, 256, 256, 256, 256, 32), torch.bfloat16, False, True, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (5064, 5632, 2048, 256, 256, 256, 256, 256, 64)
    (None, (5064, 2048, 5632, 224, 256, 256, 224, 256, 32), torch.bfloat16, False, True, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (5064, 2048, 5632, 224, 256, 256, 224, 256, 64)
    (None, (65536, 2, 1024, 512, 16, 64, 128, 16, 16), torch.float32, False, True, FullLoadMode.B, True, True,
     (1, 1, 1, 4, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (65536, 2, 1024, 432, 16, 64, 432, 16, 16)
    (None, (4608, 12476, 2048, 256, 256, 128, 256, 256, 16), torch.float32, False, True, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (4608, 12476, 2048, 256, 256, 128, 256, 256, 32)
    (None, (4608, 2048, 12476, 256, 256, 96, 256, 256, 16), torch.float32, False, False, FullLoadMode.NONE, True, True,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (4608, 2048, 12476, 256, 256, 64, 256, 256, 32)
    (None, (16384, 1536, 4096, 256, 256, 128, 256, 256, 32), torch.bfloat16, True, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (16384, 1536, 4096, 256, 256, 128, 256, 256, 64)
    (None, (65536, 768, 768, 256, 256, 128, 256, 256, 32), torch.float16, False, False, FullLoadMode.NONE, False, True,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (65536, 768, 768, 256, 256, 128, 256, 256, 64)
    (None, (4096, 24576, 1536, 256, 256, 256, 256, 256, 32), torch.bfloat16, False, True, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 24576, 1536, 256, 256, 256, 256, 256, 64)
    (None, (4, 32000, 8192, 16, 256, 128, 16, 256, 16), torch.float16, False, True, FullLoadMode.A, False, False,
     (1, 1, 1, 4, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4, 32000, 8192, 16, 256, 128, 16, 256, 64)
    (None, (65536, 4096, 1024, 256, 256, 64, 256, 256, 16), torch.float32, False, False, FullLoadMode.NONE, True, True,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (65536, 4096, 1024, 256, 256, 64, 256, 256, 32)
    (None, (128, 2304, 768, 128, 64, 128, 128, 64, 32), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (128, 2304, 768, 128, 64, 128, 128, 64, 64)
    (27, (3, 1280, 2816, 16, 48, 512, 16, 48, 128), torch.float16, False, True, FullLoadMode.NONE, False, True,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (3, 1280, 2816, 16, 48, 512, 16, 48, 256)
    (None, (1494, 750, 2048, 256, 128, 64, 256, 128, 16), torch.float32, True, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (1494, 750, 2048, 256, 128, 64, 256, 128, 32)
    (None, (2048, 750, 1494, 240, 192, 128, 240, 192, 16), torch.float32, False, False, FullLoadMode.NONE, True, True,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (2048, 750, 1494, 240, 192, 128, 240, 192, 32)
    (None, (1, 6912, 1152, 16, 192, 256, 16, 192, 16), torch.float16, False, True, FullLoadMode.NONE, False, True,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (1, 6912, 1152, 16, 192, 256, 16, 192, 64)
    (None, (7680, 32, 1152, 160, 32, 256, 160, 32, 16), torch.float16, False, True, FullLoadMode.B, False, True,
     (1, 1, 1, 4, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (7680, 32, 1152, 160, 32, 256, 160, 32, 64)
    (None, (3072, 1280, 1280, 224, 256, 160, 224, 256, 16), torch.float16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (3072, 1280, 1280, 224, 256, 256, 224, 256, 64)
    (None, (64, 6144, 1536, 64, 128, 64, 64, 128, 32), torch.float16, False, False, FullLoadMode.A, False, True,
     (1, 1, 1, 4, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (64, 6144, 1536, 64, 128, 256, 64, 128, 128)
    (None, (1024, 7680, 256, 256, 256, 128, 256, 256, 32), torch.float16, False, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (1024, 7680, 256, 256, 256, 256, 256, 256, 64)
    (None, (128, 6781, 1500, 128, 192, 64, 128, 192, 16), torch.float32, True, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (128, 6781, 1500, 128, 192, 64, 128, 192, 32)
    (None, (4, 8192, 3584, 16, 240, 256, 16, 240, 32), torch.float16, False, True, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4, 8192, 3584, 16, 240, 256, 16, 240, 64)
    (None, (120, 10240, 4096, 128, 288, 128, 128, 288, 16), torch.float16, False, True, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (120, 10240, 4096, 128, 288, 128, 128, 288, 32)
    (34, (1, 8080, 7168, 16, 240, 256, 16, 240, 32), torch.float16, False, True, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (1, 8080, 7168, 16, 240, 256, 16, 240, 64)
    (None, (4608, 10556, 1024, 256, 256, 128, 256, 256, 16), torch.float32, False, True, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (4608, 10556, 1024, 256, 256, 128, 256, 256, 32)
    (None, (3072, 12318, 2048, 256, 256, 128, 256, 256, 16), torch.float32, False, True, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (3072, 12318, 2048, 256, 256, 128, 256, 256, 32)
    (None, (12288, 5120, 640, 256, 256, 64, 256, 256, 32), torch.float16, False, True, FullLoadMode.NONE, False, True,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (12288, 5120, 640, 256, 256, 128, 256, 256, 64)
    (None, (8192, 7680, 2048, 256, 256, 256, 256, 256, 32), torch.float16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (8192, 7680, 2048, 256, 256, 256, 256, 256, 64)
    (None, (4, 25088, 4096, 16, 352, 64, 16, 176, 16), torch.float32, False, False, FullLoadMode.A, False, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),  # TODO: (4, 25088, 4096, 16, 352, 48, 16, 352, 16)
    (None, (1024, 15360, 7680, 256, 256, 256, 256, 256, 32), torch.float16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (1024, 15360, 7680, 256, 256, 256, 256, 256, 64)
    (None, (2752, 4096, 32768, 256, 256, 256, 256, 256, 32), torch.float16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (2752, 4096, 32768, 256, 256, 256, 256, 256, 64)
    (None, (7168, 18432, 4096, 256, 256, 128, 256, 256, 32), torch.bfloat16, True, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (7168, 18432, 4096, 256, 256, 128, 256, 256, 64)
    (None, (5064, 32000, 2048, 256, 256, 256, 256, 256, 32), torch.bfloat16, False, True, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (5064, 32000, 2048, 256, 256, 256, 256, 256, 64)
    (None, (36864, 7168, 4096, 256, 256, 128, 256, 256, 32), torch.bfloat16, True, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (36864, 7168, 4096, 256, 256, 128, 256, 256, 64)
    (None, (4096, 36864, 7168, 256, 256, 256, 256, 256, 32), torch.bfloat16, False, True, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 36864, 7168, 256, 256, 256, 256, 256, 64)
    (None, (4096, 64640, 7168, 256, 256, 256, 256, 256, 32), torch.float16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 64640, 7168, 256, 256, 256, 256, 256, 64)
    (None, (129280, 7168, 4096, 256, 256, 64, 256, 256, 32), torch.bfloat16, True, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (129280, 7168, 4096, 256, 256, 128, 256, 256, 64)
    (None, (16384, 4096, 1376, 256, 256, 256, 256, 256, 32), torch.float16, False, True, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (16384, 4096, 1376, 256, 256, 256, 256, 256, 64)
    (None, (4096, 1376, 32768, 256, 256, 256, 256, 256, 32), torch.float16, True, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 1376, 32768, 256, 256, 256, 256, 256, 64)
    (None, (4096, 512, 32768, 240, 256, 256, 240, 256, 32), torch.float16, True, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 512, 32768, 240, 256, 256, 240, 256, 64)
    (None, (12928, 7168, 4096, 256, 256, 256, 256, 256, 32), torch.bfloat16, True, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (12928, 7168, 4096, 256, 256, 256, 256, 256, 64)
    (None, (7680, 3456, 1152, 256, 256, 192, 256, 256, 32), torch.bfloat16, False, True, FullLoadMode.NONE, False, True,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (7680, 3456, 1152, 256, 256, 192, 256, 256, 64)
    (None, (8192, 3072, 1536, 256, 256, 256, 256, 256, 32), torch.float16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (8192, 3072, 1536, 256, 256, 256, 256, 256, 64)
    (None, (32768, 4096, 1536, 256, 256, 256, 256, 256, 32), torch.float16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (32768, 4096, 1536, 256, 256, 256, 256, 256, 64)
    (None, (1536, 4096, 32768, 256, 256, 256, 256, 256, 32), torch.float16, True, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (1536, 4096, 32768, 256, 256, 256, 256, 256, 64)
    (None, (7168, 8192, 4096, 256, 256, 256, 256, 256, 32), torch.bfloat16, True, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (7168, 8192, 4096, 256, 256, 256, 256, 256, 64)
    (None, (4096, 14336, 3584, 256, 256, 256, 256, 256, 32), torch.bfloat16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 14336, 3584, 256, 256, 256, 256, 256, 64)
    (None, (4096, 8192, 3584, 256, 256, 256, 256, 256, 32), torch.bfloat16, False, True, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 8192, 3584, 256, 256, 256, 256, 256, 64)
    (None, (4096, 4000, 8192, 256, 256, 256, 256, 256, 32), torch.bfloat16, False, True, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 4000, 8192, 256, 256, 256, 256, 256, 64)
    (None, (4096, 8192, 4000, 256, 256, 256, 256, 256, 32), torch.bfloat16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 8192, 4000, 256, 256, 256, 256, 256, 64)
    (None, (7680, 4608, 1152, 256, 256, 192, 256, 256, 32), torch.float16, False, True, FullLoadMode.NONE, False, True,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (7680, 4608, 1152, 256, 256, 192, 256, 256, 64)
    (None, (12288, 640, 5120, 256, 256, 256, 256, 256, 32), torch.float16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (12288, 640, 5120, 256, 256, 192, 256, 256, 64)
    (None, (4096, 10240, 5120, 256, 256, 256, 256, 256, 32), torch.float16, False, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 10240, 5120, 256, 256, 256, 256, 256, 64)
    (None, (4096, 1024, 8192, 256, 256, 256, 256, 256, 32), torch.float16, False, True, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (4096, 1024, 8192, 256, 256, 256, 256, 256, 64)
    (None, (17, 2304, 1152, 32, 64, 1024, 32, 64, 128), torch.float16, False, True, FullLoadMode.NONE, False, True,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (17, 2304, 1152, 32, 64, 1024, 32, 64, 256)
    (None, (24576, 1536, 4096, 256, 256, 256, 256, 256, 32), torch.bfloat16, True, False, FullLoadMode.NONE, False,
     False, (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (24576, 1536, 4096, 256, 256, 256, 256, 256, 64)
    (None, (1152, 4608, 7680, 256, 256, 256, 256, 256, 32), torch.float16, True, False, FullLoadMode.NONE, False, False,
     (1, 1, 1, 2, 2), (-1, 1), (1e-3, 1e-3)),  # TODO: (1152, 4608, 7680, 256, 256, 256, 256, 256, 64)
]


@pytest.mark.parametrize(
    "core_num, tiling_data, dtype, is_a_transpose_l0, is_b_transpose_l0, full_load_mode, enable_hf32_mode, has_bias, double_buffering, input_range, accuracy",
    test_cases, ids=["_".join(map(str, tc[1][:3])) for tc in test_cases])
def test_matmul_v3(profiler, runs, core_num, tiling_data, dtype, is_a_transpose_l0, is_b_transpose_l0, full_load_mode,
                   enable_hf32_mode, has_bias, double_buffering, input_range, accuracy):
    quant_type = asc2.float32
    if dtype == torch.float16:
        quant_type = asc2.float16
    elif dtype == torch.bfloat16:
        quant_type = asc2.bfloat16
    m, n, k, m_L1, n_L1, k_L1, base_m, base_n, base_k = tiling_data
    a_shape = (m, k) if not is_a_transpose_l0 else (k, m)
    b_shape = (k, n) if not is_b_transpose_l0 else (n, k)
    low, high = input_range
    a = (high - low) * torch.rand(a_shape, dtype=dtype) + low
    b = (high - low) * torch.rand(b_shape, dtype=dtype) + low
    c = torch.zeros((m, n), dtype=dtype)
    bias = (high - low) * torch.rand([n], dtype=dtype) + low
    with profiler.profile():
        for _ in range(runs):
            matmul_v3_kernel[core_num](a, b, c, bias, a.shape, b.shape, m_L1, n_L1, k_L1, base_m, base_n, base_k,
                                       is_a_transpose_l0, is_b_transpose_l0, full_load_mode, quant_type,
                                       enable_hf32_mode, has_bias, double_buffering)
    if is_a_transpose_l0:
        a = a.T
    if is_b_transpose_l0:
        b = b.T
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    if has_bias:
        c_ref = c_ref + bias
    c_ref = c_ref.to(dtype)
    atol, rtol = accuracy
    torch.testing.assert_close(c, c_ref, atol=atol, rtol=rtol)
