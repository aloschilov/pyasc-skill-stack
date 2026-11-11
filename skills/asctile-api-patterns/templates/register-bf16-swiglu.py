"""Native register specialization for contiguous BF16 SwiGLU on the qualified 950PR."""
try:
    from asc.experimental import asctile
except ImportError:
    import asctile


@asctile.jit
def _swiglu_register_pairs(xptr: asctile.GlobalAddress, yptr: asctile.GlobalAddress,
                        rows: int, groups: int):
    tile = 2048
    x = asctile.global_tensor(xptr, [2 * rows])
    y = asctile.global_tensor(yptr, [rows])
    for group in asctile.range(asctile.block_idx(), groups, asctile.block_num()):
        first = group * tile
        valid = min(tile, rows - first)
        flat = asctile.copy_in(x, [2 * first], [2 * tile],
                               real_shape=[2 * valid], pad_value=0.0)
        result = asctile.inline_vf('''
            using namespace AscendC::Reg;
            auto* src = reinterpret_cast<__ubuf__ bfloat16_t*>($1.GetPhyAddr());
            auto* dst = reinterpret_cast<__ubuf__ bfloat16_t*>($0.GetPhyAddr());
            auto all = CreateMask<uint8_t>();
            static constexpr CastTrait to0 = {RegLayout::ZERO,SatMode::UNKNOWN,MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
            static constexpr CastTrait to1 = {RegLayout::ONE,SatMode::UNKNOWN,MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
            static constexpr CastTrait out0 = {RegLayout::ZERO,SatMode::NO_SAT,MaskMergeMode::ZEROING,AscendC::RoundMode::CAST_RINT};
            static constexpr CastTrait out1 = {RegLayout::ONE,SatMode::NO_SAT,MaskMergeMode::ZEROING,AscendC::RoundMode::CAST_RINT};
            RegTensor<bfloat16_t> a,b,g,v,o0,o1;
            RegTensor<float> gf,vf,tmp,one;
            Duplicate(one,1.0f,all);
            // Two 128-lane loads contain 128 adjacent gate/value pairs.
            for (uint16_t i=0; i<16; ++i) {
                LoadAlign(a,src+256*i);
                LoadAlign(b,src+256*i+128);
                DeInterleave(g,v,a,b);
                // Widen alternating lanes separately, then repack into BF16.
                Cast<float,bfloat16_t,to0>(gf,g,all);
                Cast<float,bfloat16_t,to0>(vf,v,all);
                Muls(tmp,gf,-1.0f,all);
                Exp(tmp,tmp,all);
                Adds(tmp,tmp,1.0f,all);
                Div(tmp,one,tmp,all);
                Mul(tmp,gf,tmp,all);
                Mul(tmp,tmp,vf,all);
                Cast<bfloat16_t,float,out0>(o0,tmp,all);
                Cast<float,bfloat16_t,to1>(gf,g,all);
                Cast<float,bfloat16_t,to1>(vf,v,all);
                Muls(tmp,gf,-1.0f,all);
                Exp(tmp,tmp,all);
                Adds(tmp,tmp,1.0f,all);
                Div(tmp,one,tmp,all);
                Mul(tmp,gf,tmp,all);
                Mul(tmp,tmp,vf,all);
                Cast<bfloat16_t,float,out1>(o1,tmp,all);
                Or((RegTensor<uint16_t>&)o0,(RegTensor<uint16_t>&)o0,(RegTensor<uint16_t>&)o1,all);
                StoreAlign(dst+128*i,o0,all);
            }
            ''', [2048], asctile.bfloat16, [flat])
        asctile.copy_out(result, y, [first], real_shape=[valid])


def _register_pair_tile_size(rows, ub_bytes, vector_count):
    """Use the qualified 2048-pair specialization when it fits and fills the grid."""
    if ub_bytes is None or ub_bytes < 12288 or rows <= 0:
        return None
    if (rows + 2047) // 2048 < vector_count or not hasattr(asctile, "inline_vf"):
        return None
    from asc.lib import runtime
    if getattr(runtime.get_soc_version(), "value", None) == "Ascend950PR_9599":
        return 2048
    return None
