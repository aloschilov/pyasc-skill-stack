# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# Diagnostic extension of the target: explicit FP32 intermediate arithmetic.
import asctile


@asctile.jit(reuse_alloc=1)
def gelu(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, input_length,
         tile_length: asctile.ConstExpr, TANH_APPROX_FACTOR: asctile.ConstExpr,
         NEG_SQRT_EIGHT_OVER_PI: asctile.ConstExpr, unroll_factor: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [input_length])
    out_gm = asctile.global_tensor(output_ptr, [input_length])
    block_loop_num = asctile.ceildiv(asctile.ceildiv(input_length, asctile.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = block_length * asctile.block_idx()
    for i in asctile.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        raw = asctile.copy_in(in_gm, [current_offset], [tile_length])
        row = raw.to(asctile.float32)
        input_sq = row * row
        input_cub = input_sq * row
        input_cub = row + input_cub * TANH_APPROX_FACTOR
        input_cub = input_cub * NEG_SQRT_EIGHT_OVER_PI
        input_cub = asctile.exp(input_cub)
        input_cub = input_cub + 1
        out = row / input_cub
        asctile.copy_out(out.to(raw.dtype), out_gm, [current_offset])
