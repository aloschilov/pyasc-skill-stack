# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reference implementation: https://github.com/sgl-project/sgl-kernel-npu/blob/87153b4f6e19d68fc11ab1bc705cc35f1d1eca3e/python/sgl_kernel_npu/sgl_kernel_npu/fla/kda_chunk_delta_h.py

from asc.experimental import asctile
import pytest
import torch
import torch.nn.functional as F


@asctile.jit(cv_ratio=2, vf_fusion=True)
def kda_chunk_delta_h_kernel(k_ptr: asctile.GlobalAddress, v_ptr: asctile.GlobalAddress, w_ptr: asctile.GlobalAddress,
                             v_new_ptr: asctile.GlobalAddress, g_ptr: asctile.GlobalAddress,
                             gk_ptr: asctile.GlobalAddress, h_ptr: asctile.GlobalAddress,
                             state_ptr: asctile.GlobalAddress, cu_seqlens_ptr: asctile.GlobalAddress,
                             chunk_offsets_ptr: asctile.GlobalAddress, shapes: asctile.ConstExpr, bt: asctile.ConstExpr,
                             bv: asctile.ConstExpr, base_k: asctile.ConstExpr, unroll_factor_k: asctile.ConstExpr,
                             use_g: asctile.ConstExpr, use_gk: asctile.ConstExpr, use_initial_state: asctile.ConstExpr,
                             save_new_value: asctile.ConstExpr, is_varlen: asctile.ConstExpr,
                             use_exp2: asctile.ConstExpr, quant_type: asctile.ConstExpr):
    """
    KDA Chunk Delta H: computes recurrent states for Kernel Delta Attention
    Algorithm: for each chunk, save state, compute delta = v - w@state,
    apply gating (decay), update state += k^T @ delta
    Supports: scalar gating (use_g), per-channel gating (use_gk),
    initial state loading, variable-length sequences, and delta output.
    """
    b, t_max, h, hg, k, v = shapes
    block_idx = asctile.block_idx()
    sub_block_num = asctile.sub_block_num()
    sb = asctile.sub_block_idx()
    start_index = block_idx / sub_block_num
    i_v = start_index / (b * h)
    nh_idx = start_index % (b * h)
    i_n = nh_idx / h
    i_h = nh_idx % h
    i_hg = i_h / (h / hg)
    bos_v = (i_n * h * t_max) + (i_h * t_max)
    bos_k = (i_n * hg * t_max) + (i_hg * t_max)
    nt_max = asctile.ceildiv(t_max, bt)
    if is_varlen:
        cu_seqlens_gm = asctile.global_tensor(cu_seqlens_ptr, [b + 1])
        bos_seq = asctile.copy_in(cu_seqlens_gm, [i_n])
        eos_seq = asctile.copy_in(cu_seqlens_gm, [i_n + 1])
        t = eos_seq - bos_seq
        nt = asctile.ceildiv(t, bt)
        chunk_offsets_gm = asctile.global_tensor(chunk_offsets_ptr, [b + 1])
        boh = asctile.copy_in(chunk_offsets_gm, [i_n])
    else:
        t = t_max
        nt = nt_max
        boh = i_n * nt_max
    k_gm = asctile.global_tensor(k_ptr, [b * hg * t_max, k])
    v_gm = asctile.global_tensor(v_ptr, [b * h * t_max, v])
    w_gm = asctile.global_tensor(w_ptr, [b * h * t_max, k])
    h_gm = asctile.global_tensor(h_ptr, [b * h * nt_max * k, v])
    state_gm = asctile.global_tensor(state_ptr, [b * h * k, v])
    if use_g:
        g_gm = asctile.global_tensor(g_ptr, [b * h * t_max])
    if use_gk:
        gk_gm = asctile.global_tensor(gk_ptr, [b * h * t_max, k])
    if save_new_value:
        v_new_gm = asctile.global_tensor(v_new_ptr, [b * h * t_max, v])
    k1 = min(k, 64)
    k2 = k - k1
    bt_half = bt // 2
    k1_half = k1 // 2
    k2_half = k2 // 2
    sb_off = sb * k1_half
    sb_off2 = sb * k2_half
    b_h1 = asctile.zeros([k1_half, bv], dtype=asctile.float32)
    if k2 > 0:
        b_h2 = asctile.zeros([k2_half, bv], dtype=asctile.float32)
    if use_initial_state:
        state_row_offset = (i_n * h * k) + (i_h * k)
        b_h1 = b_h1 + asctile.copy_in(state_gm, [state_row_offset + sb_off, i_v * bv], [k1_half, bv]).to(
            asctile.float32)
        if k2 > 0:
            b_h2 = b_h2 + asctile.copy_in(state_gm, [state_row_offset + k1 + sb_off2, i_v * bv], [k2_half, bv]).to(
                asctile.float32)
    for i_t in asctile.range(nt, gm_barrier=True):
        h_row_offset = ((boh + i_t) * h + i_h) * k
        asctile.copy_out(b_h1.to(quant_type), h_gm, [h_row_offset + sb_off, i_v * bv])
        if k2 > 0:
            asctile.copy_out(b_h2.to(quant_type), h_gm, [h_row_offset + k1 + sb_off2, i_v * bv])
        v_row_offset = bos_v + (i_t * bt)
        v_chunk = asctile.copy_in(v_gm, [v_row_offset + bt_half * sb, i_v * bv], [bt_half, bv])
        b_h1_compute = b_h1.to(quant_type)
        acc1 = asctile.zeros_acc([bt, bv], dtype=asctile.float32)
        w_row_offset = bos_v + (i_t * bt)
        w_l0 = asctile.copy_in(w_gm, [w_row_offset, 0], [bt, k1])
        h_l1 = asctile.copy(b_h1_compute, [0, 0], [k1, bv], location=asctile.TensorLocation.L1,
                            distrib=asctile.DistribMode.JoinByM)
        asctile.matmul_acc(acc1, w_l0, h_l1)
        if k2 > 0:
            b_h2_compute = b_h2.to(quant_type)
            w_l0 = asctile.copy_in(w_gm, [w_row_offset, k1], [bt, k2])
            h_l1 = asctile.copy(b_h2_compute, [0, 0], [k2, bv], location=asctile.TensorLocation.L1,
                                distrib=asctile.DistribMode.JoinByM)
            asctile.matmul_acc(acc1, w_l0, h_l1)
        acc1_ub = asctile.copy(acc1, location=asctile.TensorLocation.UB, distrib=asctile.DistribMode.SplitByM)
        b_v = v_chunk.to(asctile.float32) - acc1_ub.to(asctile.float32)
        if save_new_value:
            v_new_row_offset = bos_v + (i_t * bt)
            asctile.copy_out(b_v.to(quant_type), v_new_gm, [v_new_row_offset + bt_half * sb, i_v * bv])
        if use_g:
            last_idx = min((i_t + 1) * bt, t) - 1
            g_last_offset = bos_v + last_idx
            g_last_tensor = asctile.copy_in(g_gm, [g_last_offset], [1])
            g_chunk_offset = bos_v + (i_t * bt)
            g_chunk = asctile.copy_in(g_gm, [g_chunk_offset + bt_half * sb], [bt_half])
            g_diff = g_last_tensor - g_chunk
            g_diff_safe = asctile.where(g_diff <= 0, g_diff, asctile.full([bt_half], -1e10, dtype=asctile.float32))
            g_safe_exp = asctile.exp(g_diff_safe)
            g_safe_exp_col = g_safe_exp.expand_dims(1)
            b_v = b_v * g_safe_exp_col
            g_last_exp = asctile.exp(g_last_tensor)
            b_h1 = b_h1 * g_last_exp
            if k2 > 0:
                b_h2 = b_h2 * g_last_exp
        if use_gk:
            last_idx = min((i_t + 1) * bt, t) - 1
            gk_row_offset = bos_v + last_idx
            gk_last1 = asctile.copy_in(gk_gm, [gk_row_offset, sb_off], [k1_half])
            if use_exp2:
                gk_exp1 = asctile.exp2(gk_last1)
            else:
                gk_exp1 = asctile.exp(gk_last1)
            gk_exp1_col = gk_exp1.expand_dims(1)
            b_h1 = b_h1 * gk_exp1_col
            if k2 > 0:
                gk_last2 = asctile.copy_in(gk_gm, [gk_row_offset, k1 + sb_off2], [k2_half])
                if use_exp2:
                    gk_exp2 = asctile.exp2(gk_last2)
                else:
                    gk_exp2 = asctile.exp(gk_last2)
                gk_exp2_col = gk_exp2.expand_dims(1)
                b_h2 = b_h2 * gk_exp2_col
        b_v_compute = b_v.to(quant_type)
        k_row_offset = bos_k + (i_t * bt)
        acc2 = asctile.zeros_acc([k1, bv], dtype=asctile.float32)
        v_l1 = asctile.copy(b_v_compute, location=asctile.TensorLocation.L1, distrib=asctile.DistribMode.JoinByM)
        for bt_step in asctile.range(asctile.ceildiv(bt, base_k), unroll_factor=unroll_factor_k):
            bt_off = bt_step * base_k
            k_l0 = asctile.copy_in(k_gm, [k_row_offset + bt_off, 0], [base_k, k1]).T
            v_l0 = asctile.copy(v_l1, [bt_off, 0], [base_k, bv])
            asctile.matmul_acc(acc2, k_l0, v_l0)
        acc2_ub = asctile.copy(acc2, location=asctile.TensorLocation.UB, distrib=asctile.DistribMode.SplitByM)
        b_h1 = b_h1 + acc2_ub
        if k2 > 0:
            acc3 = asctile.zeros_acc([k2, bv], dtype=asctile.float32)
            for bt_step in asctile.range(asctile.ceildiv(bt, base_k), unroll_factor=unroll_factor_k):
                bt_off = bt_step * base_k
                k_l0 = asctile.copy_in(k_gm, [k_row_offset + bt_off, k1], [base_k, k2]).T
                v_l0 = asctile.copy(v_l1, [bt_off, 0], [base_k, bv])
                asctile.matmul_acc(acc3, k_l0, v_l0)
            acc3_ub = asctile.copy(acc3, location=asctile.TensorLocation.UB, distrib=asctile.DistribMode.SplitByM)
            b_h2 = b_h2 + acc3_ub
    state_row_offset = (i_n * h * k) + (i_h * k)
    asctile.copy_out(b_h1.to(asctile.float32), state_gm, [state_row_offset + sb_off, i_v * bv])
    if k2 > 0:
        asctile.copy_out(b_h2.to(asctile.float32), state_gm, [state_row_offset + k1 + sb_off2, i_v * bv])


def chunk_delta_h_ref(k, v, w, g=None, gk=None, initial_state=None, cu_seqlens=None, bt=64, use_exp2=False,
                      save_new_value=False):
    if cu_seqlens is None:
        b, hg, t, k_dim = k.shape
        h = v.shape[1]
        v_dim = v.shape[3]
        nt = (t + bt - 1) // bt
        h_out = torch.zeros(b, h, nt, k_dim, v_dim, dtype=torch.float32)
        state = torch.zeros(b, h, k_dim, v_dim, dtype=torch.float32)
        v_new = torch.zeros(b, h, t, v_dim, dtype=torch.float32) if save_new_value else None
        if initial_state is not None:
            state = initial_state.clone().float()
        for bi in range(b):
            for head in range(h):
                hg_idx = head // (h // hg)
                for t_idx in range(nt):
                    h_out[bi, head, t_idx] = state[bi, head].clone()
                    t_start = t_idx * bt
                    t_end = min(t_start + bt, t)
                    w_chunk = w[bi, head, t_start:t_end, :].float()
                    v_chunk = v[bi, head, t_start:t_end, :].float()
                    k_chunk = k[bi, hg_idx, t_start:t_end, :].float()
                    b_v = v_chunk - w_chunk @ state[bi, head]
                    if save_new_value:
                        v_new[bi, head, t_start:t_end, :] = b_v
                    if g is not None:
                        last_idx = t_end - 1
                        g_last = g[bi, head, last_idx].float()
                        g_chunk = g[bi, head, t_start:t_end].float()
                        g_diff = g_last - g_chunk
                        g_safe = torch.where(g_diff <= 0, g_diff, torch.tensor(float('-inf')))
                        b_v = b_v * torch.exp(g_safe).unsqueeze(-1)
                        state[bi, head] = state[bi, head] * torch.exp(g_last)
                    if gk is not None:
                        last_idx = t_end - 1
                        gk_last = gk[bi, head, last_idx, :].float()
                        if use_exp2:
                            decay = torch.exp2(gk_last)
                        else:
                            decay = torch.exp(gk_last)
                        state[bi, head] = state[bi, head] * decay.unsqueeze(-1)
                    state[bi, head] += k_chunk.t() @ b_v
        return h_out, state, v_new
    else:
        b = len(cu_seqlens) - 1
        h = v.shape[1]
        v_dim = v.shape[3]
        hg = k.shape[1]
        k_dim = k.shape[3]
        t_max = v.shape[2]
        chunk_counts = []
        chunk_offsets_list = [0]
        for bi in range(b):
            seq_len = cu_seqlens[bi + 1] - cu_seqlens[bi]
            n_chunks = (seq_len + bt - 1) // bt
            chunk_counts.append(n_chunks)
            chunk_offsets_list.append(chunk_offsets_list[-1] + n_chunks)
        total_chunks = chunk_offsets_list[-1]
        h_out = torch.zeros(total_chunks, h, k_dim, v_dim, dtype=torch.float32)
        state = torch.zeros(b, h, k_dim, v_dim, dtype=torch.float32)
        v_new = torch.zeros(b, h, t_max, v_dim, dtype=torch.float32) if save_new_value else None
        if initial_state is not None:
            state = initial_state.clone().float()
        for bi in range(b):
            bos = cu_seqlens[bi]
            eos = cu_seqlens[bi + 1]
            seq_len = eos - bos
            boh = chunk_offsets_list[bi]
            for head in range(h):
                hg_idx = head // (h // hg)
                for t_idx in range(chunk_counts[bi]):
                    h_out[boh + t_idx, head] = state[bi, head].clone()
                    t_start = t_idx * bt
                    t_end = min(t_start + bt, seq_len)
                    w_chunk = w[bi, head, t_start:t_end, :].float()
                    v_chunk = v[bi, head, t_start:t_end, :].float()
                    k_chunk = k[bi, hg_idx, t_start:t_end, :].float()
                    b_v = v_chunk - w_chunk @ state[bi, head]
                    if save_new_value:
                        v_new[bi, head, t_start:t_end, :] = b_v
                    if g is not None:
                        last_idx = t_end - 1
                        g_last = g[bi, head, last_idx].float()
                        g_chunk = g[bi, head, t_start:t_end].float()
                        g_diff = g_last - g_chunk
                        g_safe = torch.where(g_diff <= 0, g_diff, torch.tensor(float('-inf')))
                        b_v = b_v * torch.exp(g_safe).unsqueeze(-1)
                        state[bi, head] = state[bi, head] * torch.exp(g_last)
                    if gk is not None:
                        last_idx = t_end - 1
                        gk_last = gk[bi, head, last_idx, :].float()
                        if use_exp2:
                            decay = torch.exp2(gk_last)
                        else:
                            decay = torch.exp(gk_last)
                        state[bi, head] = state[bi, head] * decay.unsqueeze(-1)
                    state[bi, head] += k_chunk.t() @ b_v
        return h_out, state, v_new


def run_kda_chunk_delta_h_test(profiler, runs, b, t, h, hg, k, v, bt, bv, dtype, use_g, use_gk, use_initial_state,
                               save_new_value, is_varlen, use_exp2, base_k, unroll_factor_k, reuse_alloc,
                               always_compile=False, w_scale=0.01):
    assert k <= 128, f"k must be <= 128, got {k}"
    quant_type = asctile.float16
    if dtype == torch.bfloat16:
        quant_type = asctile.bfloat16
    elif dtype == torch.float32:
        quant_type = asctile.float32
    t_max = t
    t_padded = ((t + bt - 1) // bt) * bt
    k_tensor = F.normalize(torch.randn(b, hg, t_padded, k, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v_tensor = torch.randn(b, h, t_padded, v, dtype=dtype)
    w_tensor = torch.randn(b, h, t_padded, k, dtype=dtype) * w_scale
    k_tensor[:, :, t:] = 0
    v_tensor[:, :, t:] = 0
    w_tensor[:, :, t:] = 0
    g_tensor = F.logsigmoid(torch.rand(b, h, t_padded, dtype=torch.float32)) if use_g else None
    gk_tensor = F.logsigmoid(torch.rand(b, h, t_padded, k, dtype=torch.float32)) if use_gk else None
    if use_g:
        g_tensor[:, :, t:] = 0
    if use_gk:
        gk_tensor[:, :, t:] = 0
    initial_state = torch.randn(b, h, k, v, dtype=torch.float32) if use_initial_state else None
    if is_varlen:
        seq_lens = [t_max]
        cu_seqlens = torch.tensor([0] + [sum(seq_lens[:i + 1]) for i in range(b)], dtype=torch.int32)
        chunk_offsets = [0]
        for seq_len in seq_lens:
            n_chunks = (seq_len + bt - 1) // bt
            chunk_offsets.append(chunk_offsets[-1] + n_chunks)
        chunk_offsets = torch.tensor(chunk_offsets, dtype=torch.int32)
        total_chunks = chunk_offsets[-1].item()
    else:
        cu_seqlens = None
        chunk_offsets = None
        total_chunks = b * ((t_max + bt - 1) // bt)
    nt = (t_max + bt - 1) // bt
    h_out = torch.zeros(total_chunks, h, k, v, dtype=dtype)
    state = torch.zeros(b, h, k, v, dtype=torch.float32)
    v_new = torch.zeros(b, h, t_padded, v, dtype=dtype) if save_new_value else None
    v_tiles = (v + bv - 1) // bv
    block_num = v_tiles * b * h
    g_arg = g_tensor if use_g else torch.zeros(1)
    gk_arg = gk_tensor if use_gk else torch.zeros(1)
    v_new_arg = v_new if save_new_value else torch.zeros(1)
    cu_seqlens_arg = cu_seqlens if is_varlen else torch.zeros(1)
    chunk_offsets_arg = chunk_offsets if is_varlen else torch.zeros(1)
    shapes = (b, t_padded, h, hg, k, v)
    if use_initial_state:
        state = initial_state.clone()
    state_backup = state.clone()
    with profiler.profile():
        for _ in range(runs):
            state.copy_(state_backup)
            kda_chunk_delta_h_kernel[block_num](k_tensor, v_tensor, w_tensor, v_new_arg, g_arg, gk_arg, h_out, state,
                                                cu_seqlens_arg, chunk_offsets_arg, shapes, bt, bv, base_k,
                                                unroll_factor_k, use_g, use_gk, use_initial_state, save_new_value,
                                                is_varlen, use_exp2, quant_type, reuse_alloc=reuse_alloc,
                                                always_compile=always_compile)
    h_ref, state_ref, v_new_ref = chunk_delta_h_ref(k_tensor, v_tensor, w_tensor, g=g_tensor, gk=gk_tensor,
                                                    initial_state=initial_state, cu_seqlens=cu_seqlens, bt=bt,
                                                    use_exp2=use_exp2, save_new_value=save_new_value)
    if is_varlen:
        torch.testing.assert_close(h_out.float(), h_ref, atol=5e-2, rtol=5e-2)
    else:
        h_compare = h_out.view(b, nt, h, k, v).permute(0, 2, 1, 3, 4)
        torch.testing.assert_close(h_compare.float(), h_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(state.float(), state_ref, atol=5e-2, rtol=5e-2)
    if save_new_value:
        torch.testing.assert_close(v_new[:, :, :t].float(), v_new_ref[:, :, :t], atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "b, t, h, hg, k, v, bt, bv, dtype, use_g, use_gk, use_initial_state, save_new_value, is_varlen, use_exp2, "
    "base_k, unroll_factor_k, reuse_alloc", [
        # --- pypto-gym target shapes (HV=4, K=V=BT=128) ---
        (1, 8192, 4, 4, 128, 128, 128, 128, torch.float16, False, True, False, False, False, False, 64, 1, 2),
        (1, 4117, 4, 4, 128, 128, 128, 128, torch.float16, False, True, False, False, False, False, 64, 1, 2),
        (1, 1024, 4, 4, 128, 128, 128, 128, torch.float16, False, True, False, False, True, False, 64, 1, 2),
        (1, 3118, 4, 4, 128, 128, 128, 128, torch.float16, False, True, False, False, True, False, 64, 1, 2),
        (1, 768, 4, 4, 128, 128, 128, 128, torch.float16, False, True, False, False, True, False, 64, 1, 2),
        (1, 156, 4, 4, 128, 128, 128, 128, torch.float16, False, True, False, False, True, False, 64, 1, 2),
        # --- sgl-kernel-npu target shapes (D=K=V=128, BT=64, BV=32) ---
        (1, 6, 8, 8, 128, 128, 64, 32, torch.float16, True, False, False, True, True, False, 64, 1, 0),
        (1, 64, 8, 8, 128, 128, 64, 32, torch.float16, True, False, False, True, True, False, 64, 1, 0),
        (1, 127, 8, 8, 128, 128, 64, 32, torch.float16, True, False, False, True, True, False, 64, 1, 0),
        (1, 7168, 8, 8, 128, 128, 64, 32, torch.float16, True, False, False, True, True, False, 64, 1, 0),
        (1, 2000, 8, 8, 128, 128, 64, 32, torch.float16, True, False, False, True, True, False, 64, 1, 0),
        (1, 129, 16, 4, 128, 128, 64, 32, torch.float16, True, False, False, True, False, False, 64, 1, 0),
        (1, 1333, 16, 4, 128, 128, 64, 32, torch.float16, True, False, True, True, True, False, 64, 1, 0),
        (1, 1333, 64, 16, 128, 128, 64, 32, torch.float16, True, False, True, True, True, False, 64, 1, 0),
    ])
def test_kda_chunk_delta_h(profiler, runs, b, t, h, hg, k, v, bt, bv, dtype, use_g, use_gk, use_initial_state,
                           save_new_value, is_varlen, use_exp2, base_k, unroll_factor_k, reuse_alloc):
    run_kda_chunk_delta_h_test(profiler, runs, b, t, h, hg, k, v, bt, bv, dtype, use_g, use_gk, use_initial_state,
                               save_new_value, is_varlen, use_exp2, base_k, unroll_factor_k, reuse_alloc)
