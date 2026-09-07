# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# Target-style exact GeLU diagnostic; not a qualified submission candidate.
import asctile


@asctile.jit(reuse_alloc=1)
def gelu(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
         input_length, tile_length: asctile.ConstExpr, unroll_factor: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [input_length])
    out_gm = asctile.global_tensor(output_ptr, [input_length])
    block_loop_num = asctile.ceildiv(asctile.ceildiv(input_length, asctile.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = block_length * asctile.block_idx()
    for i in asctile.range(block_loop_num, unroll_factor=unroll_factor):
        offset = block_offset + i * tile_length
        valid = max(0, min(input_length - offset, tile_length))
        raw = asctile.copy_in(in_gm, [offset], [tile_length], real_shape=[valid], pad_value=0)
        x = raw.to(asctile.float32)
        out = (x * 0.5) * (asctile.erf(x * 0.7071067811865476) + 1.0)
        asctile.copy_out(out.to(raw.dtype), out_gm, [offset], real_shape=[valid])
