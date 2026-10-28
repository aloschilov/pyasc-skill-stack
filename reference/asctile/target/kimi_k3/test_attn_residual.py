# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reference implementation: https://github.com/sgl-project/sgl-kernel-npu/blob/c28ea2a940a53c00f3a0322d9576210a7f5ae92f/python/sgl_kernel_npu/sgl_kernel_npu/kimi_k3/attn_residual.py

from asc.experimental import asctile
import pytest
import torch


@asctile.jit(reuse_alloc=2)
def attn_residual_kernel(prefix_ptr: asctile.GlobalAddress, bank_ptr: asctile.GlobalAddress,
                         combined_weight_ptr: asctile.GlobalAddress, indices_ptr: asctile.GlobalAddress,
                         output_ptr: asctile.GlobalAddress, num_tokens, num_valid_blocks,
                         hidden_size: asctile.ConstExpr, variance_epsilon: asctile.ConstExpr,
                         num_candidates_aligned: asctile.ConstExpr, unroll_factor: asctile.ConstExpr):
    """
    Ascend Kimi-K3 attention-residual score and combine pipeline.

    Keep scoring, softmax, and mixing in one persistent vector-core kernel to
    avoid materializing the score matrix and launching a second kernel.  N and
    B stay dynamic so prefill/decode shapes reuse the same compilation.
    """
    # Global memory descriptors
    prefix_gm = asctile.global_tensor(prefix_ptr, [num_tokens, hidden_size])
    bank_gm = asctile.global_tensor(bank_ptr, [num_tokens * num_valid_blocks, hidden_size])
    combined_weight_gm = asctile.global_tensor(combined_weight_ptr, [hidden_size])
    indices_gm = asctile.global_tensor(indices_ptr, [num_candidates_aligned])
    output_gm = asctile.global_tensor(output_ptr, [num_tokens, hidden_size])
    token = asctile.block_idx()
    if token < num_tokens:
        # Load combined weight once per token
        combined_weight = asctile.copy_in(combined_weight_gm, [0], [hidden_size])
        # Initialize indices and scores
        indices_2d = asctile.copy_in(indices_gm, [0], [num_candidates_aligned])
        scores_2d = asctile.full([num_candidates_aligned], -1e9, dtype=asctile.float32)
        # Score each candidate (bank rows + prefix)
        for row in asctile.range(num_valid_blocks + 1, unroll_factor=unroll_factor):
            value = asctile.full([hidden_size], 0, dtype=asctile.float32)
            if row < num_valid_blocks:
                value = asctile.copy_in(bank_gm, [token * num_valid_blocks + row, 0], [hidden_size])
            else:
                value = asctile.copy_in(prefix_gm, [token, 0], [hidden_size])
            # Compute RMSNorm score
            sum_sq = asctile.reduce_sum(value * value)
            inverse_rms = 1.0 / asctile.sqrt(sum_sq / hidden_size + variance_epsilon)
            score = asctile.reduce_sum(value * inverse_rms * combined_weight)
            mask = indices_2d == asctile.full([num_candidates_aligned], row, dtype=asctile.int32)
            scores_2d = asctile.where(mask, score, scores_2d)
        # Softmax over scores
        probabilities_2d = asctile.softmax(scores_2d)
        # Weighted sum of candidates
        output_vec = asctile.zeros([hidden_size], dtype=asctile.float32)
        for row in asctile.range(num_valid_blocks + 1, unroll_factor=unroll_factor):
            value = asctile.full([hidden_size], 0, dtype=asctile.float32)
            if row < num_valid_blocks:
                value = asctile.copy_in(bank_gm, [token * num_valid_blocks + row, 0], [hidden_size])
            else:
                value = asctile.copy_in(prefix_gm, [token, 0], [hidden_size])
            mask = indices_2d == asctile.full([num_candidates_aligned], row, dtype=asctile.int32)
            probability = asctile.reduce_sum(asctile.where(mask, probabilities_2d, 0))
            output_vec = output_vec + value * probability
        asctile.copy_out(output_vec, output_gm, [token, 0])


def reference_attn_residual(prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon):
    num_tokens, hidden_size = prefix_sum.shape
    output = torch.zeros_like(prefix_sum)
    for token in range(num_tokens):
        candidates = []
        for row in range(num_valid_blocks):
            candidates.append(bank[token, row, :])
        candidates.append(prefix_sum[token, :])
        candidates = torch.stack(candidates)
        sum_sq = (candidates * candidates).sum(dim=1)
        inverse_rms = torch.rsqrt(sum_sq / hidden_size + variance_epsilon)
        scores = (candidates * inverse_rms.unsqueeze(1) * combined_weight.unsqueeze(0)).sum(dim=1)
        probabilities = torch.softmax(scores, dim=0)
        out = (probabilities.unsqueeze(1) * candidates).sum(dim=0)
        output[token] = out
    return output


def run_attn_residual_test(profiler, runs, num_tokens, num_valid_blocks, hidden_size, unroll_factor):
    block_num = num_tokens
    variance_epsilon = 1e-5
    num_candidates = num_valid_blocks + 1
    num_candidates_aligned = max(8, 2**(num_candidates - 1).bit_length())
    prefix_sum = torch.randn(num_tokens, hidden_size, dtype=torch.float32)
    bank = torch.randn(num_tokens, num_valid_blocks, hidden_size, dtype=torch.float32)
    combined_weight = torch.randn(hidden_size, dtype=torch.float32)
    bank_2d = bank.reshape(num_tokens * num_valid_blocks, hidden_size)
    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    indices = torch.arange(num_candidates_aligned, dtype=torch.int32)
    with profiler.profile():
        for _ in range(runs):
            attn_residual_kernel[block_num](prefix_sum, bank_2d, combined_weight, indices, output, num_tokens,
                                            num_valid_blocks, hidden_size, variance_epsilon, num_candidates_aligned,
                                            unroll_factor)
    expected = reference_attn_residual(prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon)
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("num_tokens, num_valid_blocks, hidden_size, unroll_factor", [
    # fla (L, B, T, D) → (num_tokens=B*T, num_valid_blocks=L-1, hidden_size=D)
    (1000, 2, 4096, 2),  # L=3
    (15, 14, 4096, 2),  # T=15
    (1000, 6, 1000, 2),  # D=1000
    (1000, 6, 2000, 2),  # D=2000
    (5000, 28, 4096, 2),  # L=29 + B=5
    (5000, 14, 7186, 2),  # B=5 + D=7186
    (189, 28, 7186, 2),  # L=29 + D=7186 + T=63
    (16000, 9, 4096, 2),  # L=10, B=2, T=8000
    # pypto: b=2, t=4096, n=25, d=512 → num_tokens = 2*4096 = 8192
    (8192, 25, 512, 2),
    # pypto: b=1, t=1023, n=32, d=512 → num_tokens = 1*1023 = 1023
    (1023, 32, 512, 2),
])
def test_attn_residual(profiler, runs, num_tokens, num_valid_blocks, hidden_size, unroll_factor):
    run_attn_residual_test(profiler, runs, num_tokens, num_valid_blocks, hidden_size, unroll_factor)
