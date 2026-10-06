# Exact device body extracted from submitted asctile/gelu.py.
# Specialization: {"arg_types": {"input_length": "PlainArgType:int32", "input_ptr": "PointerArgType:bfloat16", "output_ptr": "PointerArgType:bfloat16"}, "constexprs": {"approximate": "ConstExpr[bool](False)", "is_fp32": "ConstExpr[bool](False)", "tile_length": "ConstExpr[int](4608)", "unroll_factor": "ConstExpr[int](2)"}, "kernel": "gelu_reference_kernel"}
import asctile

@asctile.jit(reuse_alloc=2, vf_fusion=True)
def gelu_reference_kernel(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
                input_length, tile_length: asctile.ConstExpr,
                unroll_factor: asctile.ConstExpr, approximate: asctile.ConstExpr,
                is_fp32: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [input_length])
    out_gm = asctile.global_tensor(output_ptr, [input_length])
    block_length = asctile.ceildiv(asctile.ceildiv(input_length, 72), 512) * 512
    block_start = asctile.block_idx() * block_length
    block_end = min(block_start + block_length, input_length)
    local_tiles = asctile.ceildiv(max(block_end - block_start, 0), tile_length)
    for i in asctile.range(local_tiles, unroll_factor=unroll_factor):
        offset = block_start + i * tile_length
        valid = max(0, min(block_end - offset, tile_length))
        raw = asctile.copy_in(in_gm, [offset], [tile_length], real_shape=[valid], pad_value=0)
        x = raw.to(asctile.float32)
        if approximate:
            square = x * x
            cube = square * x
            exponent = (x + cube * 0.044715) * -1.5957691216057308
            out = x / (asctile.exp(exponent) + 1.0)
        elif is_fp32:
            erf_arg = x * -0.7071067811865476
            complement = asctile.inline_vf("""
                {
                    AscendC::Erfc<float, false>($0, $1, $1.GetSize());
                }
            """, [tile_length], asctile.float32, [erf_arg])
            out = (x * 0.5) * complement
        else:
            out = (x * 0.5) * (asctile.erf(x * 0.7071067811865476) + 1.0)
        asctile.copy_out(out.to(raw.dtype), out_gm, [offset], real_shape=[valid])
