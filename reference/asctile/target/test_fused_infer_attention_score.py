# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.experimental import asctile
import pytest
import torch


@asctile.jit
def bmm1(q_part, k_part, D: asctile.ConstExpr, s1_batch: asctile.ConstExpr, step2: asctile.ConstExpr,
         mm1_K_step: asctile.ConstExpr):
    mm_acc = asctile.zeros_acc([s1_batch, step2], dtype=asctile.float32)
    for k in asctile.range(asctile.ceildiv(D, mm1_K_step), unroll_factor=2):
        l0q = asctile.copy(q_part, [0, k * mm1_K_step], [s1_batch, mm1_K_step], location=asctile.TensorLocation.L0A)
        l0k = asctile.copy(k_part, [k * mm1_K_step, 0], [mm1_K_step, step2], location=asctile.TensorLocation.L0B)
        asctile.matmul_acc(mm_acc, l0q, l0k)
    return mm_acc


@asctile.jit
def online_softmax(attn_weight, softmax_max, softmax_d, scale_factor: asctile.ConstExpr, s1_batch: asctile.ConstExpr,
                   step2: asctile.ConstExpr, dtype: asctile.ConstExpr):
    # Online softmax on this sub-block's rows only (SPMD, no guards).
    scaled_weight = attn_weight * scale_factor
    max_tmp = asctile.reduce_max(scaled_weight, 1)
    x_max = asctile.maximum(softmax_max, max_tmp)
    beta = asctile.exp(max_tmp - x_max)
    exp_max = asctile.exp(softmax_max - x_max)
    x_exp = asctile.exp(scaled_weight - max_tmp.reshape(s1_batch, 1).broadcast_to(s1_batch, step2))
    sum_tmp = asctile.reduce_sum(x_exp, 1)
    softmax_d = exp_max * softmax_d + beta * sum_tmp
    beta2 = x_exp * beta.reshape(s1_batch, 1).broadcast_to(s1_batch, step2)
    p16 = beta2.to(dtype)
    l1p = asctile.copy(p16, location=asctile.TensorLocation.L1)
    return exp_max, x_max, softmax_d, l1p


@asctile.jit
def bmm2(l1p, l1v, D: asctile.ConstExpr, s1_batch: asctile.ConstExpr, step2: asctile.ConstExpr,
         mm2_K_step: asctile.ConstExpr):
    # BMM2 fragmented over K: [s1_batch, step2] @ [step2, D]
    mm_acc = asctile.zeros_acc([s1_batch, D], dtype=asctile.float32)
    for k2 in asctile.range(asctile.ceildiv(step2, mm2_K_step), unroll_factor=2):
        l0p = asctile.copy(l1p, [0, k2 * mm2_K_step], [s1_batch, mm2_K_step], location=asctile.TensorLocation.L0A)
        l0v = asctile.copy(l1v, [k2 * mm2_K_step, 0], [mm2_K_step, D], location=asctile.TensorLocation.L0B)
        asctile.matmul_acc(mm_acc, l0p, l0v)
    return mm_acc


@asctile.jit(cv_ratio=2, reuse_alloc=2, vf_fusion=True)
def flash_attention_pyasc(q_ptr: asctile.GlobalAddress, k_ptr: asctile.GlobalAddress, v_ptr: asctile.GlobalAddress,
                          o_ptr: asctile.GlobalAddress, ws_o_ptr: asctile.GlobalAddress, B: asctile.ConstExpr,
                          S1: asctile.ConstExpr, S2: asctile.ConstExpr, N: asctile.ConstExpr, N_kv: asctile.ConstExpr,
                          D: asctile.ConstExpr, dtype: asctile.ConstExpr, scale_factor: asctile.ConstExpr,
                          step1: asctile.ConstExpr, step2: asctile.ConstExpr, g_group: asctile.ConstExpr,
                          s2_unroll: asctile.ConstExpr, mm1_K_step: asctile.ConstExpr, mm2_K_step: asctile.ConstExpr,
                          kv_cache: asctile.ConstExpr):
    q_gm = asctile.global_tensor(q_ptr, [B * N * S1, D], l2_cache=False)
    k_gm = asctile.global_tensor(k_ptr, [B * N_kv * S2, D], l2_cache=kv_cache)
    v_gm = asctile.global_tensor(v_ptr, [B * N_kv * S2, D], l2_cache=kv_cache)
    o_gm = asctile.global_tensor(o_ptr, [B * N * S1, D])

    N_repeats = N // N_kv
    g_chunks = N_repeats // g_group
    s1_batch = g_group * step1

    # This is a w/a to prevent hoisting of initialization buffer for o before task loop.
    # This helps to reuse that buffer inside S2 loop and thus increase g_group size
    # fitting into UB.
    ws_o_gm = asctile.global_tensor(ws_o_ptr, [s1_batch, D])

    # On the AIVs block_idx() already contains the sub-block offset
    # (block_idx = block * sub_block_num + sub_block_idx), so dividing by
    # sub_block_num maps both sub-blocks of a block onto the same task; on the
    # AIC sub_block_num() == 1 and block_idx() is the plain block index.
    start = asctile.block_idx() / asctile.sub_block_num()
    total_tasks = asctile.ceildiv(B * N, g_group)
    step = asctile.block_num()

    # One task = one (batch, kv-head, g-chunk); both AIV sub-blocks of a block
    # work on the same task, each owning half of the M-tile rows.
    for task in asctile.range(start, total_tasks, step, unroll_factor=1):
        g_chunk = task % g_chunks
        kv_task = task // g_chunks
        n_kv_local = kv_task % N_kv
        b_idx = kv_task // N_kv
        n_kv = b_idx * N_kv + n_kv_local
        g_start = g_chunk * g_group
        q_row = (b_idx * N + n_kv_local * N_repeats + g_start) * S1

        # Q tile: L1-resident for the whole task, reused by every S2 step.
        q_part = asctile.copy_in(q_gm, [q_row, 0], [s1_batch, D], location=asctile.TensorLocation.L1)

        # Online-softmax row statistics for this sub-block's half of the rows and
        # the O accumulator, all UB-resident for the whole task.
        with asctile.cv_strategy(asctile.DistribMode.SplitByM):
            softmax_max = asctile.full([s1_batch], -1e10, dtype=asctile.float32)
            softmax_d = asctile.zeros([s1_batch], dtype=asctile.float32)
            o = asctile.copy_in(ws_o_gm, [0, 0], [s1_batch, D], location=asctile.TensorLocation.UB)

        # S2 innermost loop: stream K/V tiles GM->L1.
        for j in asctile.range(asctile.ceildiv(S2, step2), unroll_factor=s2_unroll):
            s2 = n_kv * S2 + j * step2
            k_part = asctile.copy_in(k_gm, [s2, 0], [step2, D], location=asctile.TensorLocation.L1).transpose()
            l1v = asctile.copy_in(v_gm, [s2, 0], [step2, D], location=asctile.TensorLocation.L1)
            qk = bmm1(q_part, k_part, D, s1_batch, step2, mm1_K_step)
            with asctile.cv_strategy(asctile.DistribMode.SplitByM):
                # BMM1 fixpipe, split by M: each AIV sub-block receives its row half of the attention tile.
                attn_weight = asctile.copy(qk, location=asctile.TensorLocation.UB)
                exp_max, softmax_max, softmax_d, l1p = online_softmax(attn_weight, softmax_max, softmax_d, scale_factor,
                                                                      s1_batch, step2, dtype)
            pv = bmm2(l1p, l1v, D, s1_batch, step2, mm2_K_step)
            with asctile.cv_strategy(asctile.DistribMode.SplitByM):
                # One fixpipe per S2 step, split by M: each sub-block receives its [s1_batch, D] row half and
                # accumulates it into its UB-resident O tensor, rescaling the previous state by exp_max.
                mm_res = asctile.copy(pv, location=asctile.TensorLocation.UB)
                o = o * exp_max.reshape(s1_batch, 1).broadcast_to(s1_batch, D) + mm_res

        # Final normalize: each sub-block owns the rows [q_row + s1_batch * sb, q_row + s1_batch * (sb + 1) // 2];
        # The fp16 cast is done after the fixpipe copies, on the UB tensor.
        with asctile.cv_strategy(asctile.DistribMode.SplitByM):
            update = o / softmax_d.reshape(s1_batch, 1).broadcast_to(s1_batch, D)
            asctile.copy_out(update.to(dtype), o_gm, [q_row, 0])


def flash_attention_torch_reference(q, k, v, N, N_kv, scale):
    q_ref = q.to(torch.float32)  # [B, N, S1, D]
    k_ref = k.to(torch.float32).repeat_interleave(N // N_kv, dim=1)  # [B, N_kv, S2, D]
    v_ref = v.to(torch.float32).repeat_interleave(N // N_kv, dim=1)  # [B, N_kv, S2, D]
    return torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=scale, is_causal=False,
                                                            dropout_p=0.0)


# Q: [B, S1, N, D]
# V,K: [B, S2, N_k, D]
# yapf: disable
test_cases = [
    # PYASC_TESTS_BEGIN
    pytest.param(None, ([8, 1, 16, 512], [8, 4096, 1, 512], "BSND"), torch.bfloat16, 16, 2, id="8-1-16-512-4096"),
    pytest.param(None, ([8, 1, 128, 512], [8, 4096, 1, 512], "BSND"), torch.bfloat16, 32, 2, id="8-1-128-512-4096"),
    pytest.param(None, ([64, 64, 1, 512], [64, 1, 4096, 512], "BNSD"), torch.bfloat16, 64, 1, id="64-64-1-512-4096"),
    pytest.param(None, ([64, 32, 1, 512], [64, 1, 4096, 512], "BNSD"), torch.bfloat16, 32, 2, id="64-32-1-512-4096"),
    pytest.param(None, ([64, 16, 1, 512], [64, 1, 4096, 512], "BNSD"), torch.bfloat16, 16, 2, id="64-16-1-512-4096"),
    pytest.param(None, ([64, 128, 2, 512], [64, 1, 4096, 512], "BNSD"), torch.bfloat16, 32, 1, id="64-128-2-512-4096"),
    pytest.param(None, ([64, 128, 1, 512], [64, 1, 4096, 512], "BNSD"), torch.bfloat16, 64, 1, id="64-128-1-512-4096"),
    pytest.param(None, ([64, 1, 16, 512], [64, 4096, 1, 512], "BSND"), torch.bfloat16, 16, 2, id="64-1-16-512-4096"),
    pytest.param(None, ([256, 64, 1, 512], [256, 1, 3584, 512], "BNSD"), torch.bfloat16, 64, 1, id="256-64-1-512-3584"),
    pytest.param(None, ([192, 1, 64, 512], [192, 4096, 1, 512], "BSND"), torch.bfloat16, 64, 1, id="192-1-64-512-4096"),
    pytest.param(None, ([192, 1, 64, 512], [192, 3584, 1, 512], "BSND"), torch.bfloat16, 64, 1, id="192-1-64-512-3584"),
    pytest.param(None, ([192, 1, 64, 512], [192, 3072, 1, 512], "BSND"), torch.bfloat16, 64, 1, id="192-1-64-512-3072"),
    # PYASC_TESTS_END
]
# yapf: enable


@pytest.mark.parametrize("block_num, input_shapes, torch_dtype, g_group, s2_unroll", test_cases)
def test_flash_attention_score(profiler, runs, block_num, input_shapes, torch_dtype, g_group, s2_unroll, compile_only):
    q_shape, v_shape, layout = input_shapes
    if layout == "BNSD":
        B, N, S1, D = q_shape
        N_kv, S2 = v_shape[1:3]
    elif layout == "BSND":
        B, S1, N, D = q_shape
        S2, N_kv = v_shape[1:3]
    else:
        raise RuntimeError("Wrong layout: " + layout)
    assert v_shape[0] == B and v_shape[3] == D

    assert torch_dtype in (torch.float16, torch.bfloat16)
    asctile_dtype = asctile.bfloat16 if torch_dtype == torch.bfloat16 else asctile.float16
    scale_factor = 0.041666666666666664

    # Now a separate L1 buffer is allocated for each K/V tile. So, amount of memeory utilized
    # by K/V matrices is 2 * (step2 * D * torch.finfo(dtype).bits // 8) * s2_unroll.
    # So, if double buffering is used, step2 should be decreased to fit into L1 memory.
    step1, step2 = S1, min(128 // s2_unroll, S2)

    # Bigger S2 tile (CANN s2BaseSize=128). L0B (64KB) is double-buffered by the unrolled
    # bmm2 loop, so the B2 fragment area is bounded: 2 * mm2_K_step * D * 2B <= 64KB, i.e.
    # mm2_K_step * D <= 16384 elements.
    mm1_K_step = min(256, 16384 // step2, D)
    mm2_K_step = min(16384 // D, step2)

    # L2 caching startegy:
    #  - Q tensor isn't cached (each tile is loaded once per kernel execution)
    #  - K/V tensors caching depends on whether full Q tensor sequence is processed
    #    per one task loop iteration or not. If not then K/V tiles are re-loaded.
    #  - O tensor is cached. It is used for writing only, better to use caching in this case.
    #  - WS tensor is cached, because it is small and is used for initialization only.
    kv_cache = (g_group * step1 < S1 * N)

    q = torch.randn(B, N, S1, D, dtype=torch_dtype) * 0.5
    k = torch.randn(B, N_kv, S2, D, dtype=torch_dtype)
    v = torch.randn(B, N_kv, S2, D, dtype=torch_dtype)
    o = torch.zeros(B, N, S1, D, dtype=torch_dtype)
    ws_o = torch.zeros(g_group * S1, D, dtype=torch.float32)

    with profiler.profile():
        for _ in range(runs):
            flash_attention_pyasc[block_num](q, k, v, o, ws_o, B, S1, S2, N, N_kv, D, asctile_dtype, scale_factor,
                                             step1, step2, g_group, s2_unroll, mm1_K_step, mm2_K_step, kv_cache)
    if compile_only:
        return
    o_ref = flash_attention_torch_reference(q, k, v, N, N_kv, scale_factor)
    o_ref = o_ref.to(torch_dtype)
    torch.testing.assert_close(o, o_ref, atol=1e-2, rtol=1e-2)
