"""AscTile transcription of ops-nn GeluV2 arch35 DAGs.

Only the FP32 Erf algorithm selection uses the authorized inline escape hatch.
Tanh expresses Axpy algebra, not a promise of fused-instruction equivalence.
"""
from asc.experimental import asctile


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def gelu_kernel(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
                size, tile: asctile.ConstExpr, tanh_mode: asctile.ConstExpr,
                fp32_exact: asctile.ConstExpr, unroll: asctile.ConstExpr):
    source = asctile.global_tensor(input_ptr, [size])
    destination = asctile.global_tensor(output_ptr, [size])
    tiles = asctile.ceildiv(size, tile)
    quotient = tiles // asctile.block_num()
    remainder = tiles % asctile.block_num()
    first = asctile.block_idx() * quotient + min(asctile.block_idx(), remainder)
    count = quotient + min(max(remainder - asctile.block_idx(), 0), 1)
    for i in asctile.range(count, unroll_factor=unroll):
        offset = (first + i) * tile
        valid = max(0, min(size - offset, tile))
        raw = asctile.copy_in(source, [offset], [tile], real_shape=[valid], pad_value=0)
        x = raw.to(asctile.float32)
        if tanh_mode:
            square = x * x
            cube = square * x
            # ops-nn float constants: 1/0.044715, -1.595769121*0.044715.
            argument = (cube + x * 22.363859176635742) * -0.07135481387376785
            result = x / (asctile.exp(argument) + 1.0)
        else:
            argument = x * 0.7071067690849304
            if fp32_exact:
                erf_value = asctile.inline_vf("""
                    {
                        static constexpr AscendC::ErfConfig config = {
                            AscendC::ErfAlgo::SUBSECTION_POLYNOMIAL_APPROXIMATION};
                        AscendC::Erf<float, false, config>($0, $1, $1.GetSize());
                    }
                """, [tile], asctile.float32, [argument])
            else:
                erf_value = asctile.erf(argument)
            result = (erf_value + 1.0) * (x * 0.5)
        asctile.copy_out(result.to(raw.dtype), destination, [offset], real_shape=[valid])
