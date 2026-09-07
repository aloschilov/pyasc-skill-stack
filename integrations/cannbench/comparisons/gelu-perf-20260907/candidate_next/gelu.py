"""UNSUBMITTED hybrid follow-up: retain faster iteration03 FP16 exact path.

The launch, tensor allocation, casts, tail copies and compilation use pyasc.
Math explicitly uses inline C++/AscendC registers, not the pure Python DSL.
No framework/baseline operator is called inside this implementation.
"""
import torch
import asc
import asctile
from ._pyasc_runtime import asctile_jit, ensure_npu_platform

_TILE = 8192
_MAX_CORES = 72

_EXACT = r'''
auto* in = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
auto* out = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
AscendC::Reg::RegTensor<float> x, a, z, u, p, tmp, e, y, one;
uint32_t count = $1.GetSize();
uint16_t loops = (count + 63) / 64;
AscendC::Reg::MaskReg all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
AscendC::Reg::Duplicate(one, 1.0f, all);
AscendC::Reg::RegTensor<float> c0,c1,c2,c3,c4,c5,c6,c7,c8;
AscendC::Reg::Duplicate(c0, -0.82215223f, all);
AscendC::Reg::Duplicate(c1, 1.48851587f, all);
AscendC::Reg::Duplicate(c2, -1.13520398f, all);
AscendC::Reg::Duplicate(c3, 0.27886807f, all);
AscendC::Reg::Duplicate(c4, -0.18628806f, all);
AscendC::Reg::Duplicate(c5, 0.09678418f, all);
AscendC::Reg::Duplicate(c6, 0.37409196f, all);
AscendC::Reg::Duplicate(c7, 1.00002368f, all);
AscendC::Reg::Duplicate(c8, -1.26551223f, all);
for (uint16_t i=0; i<loops; ++i) {
    auto mask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::DataCopy(x, in+i*64);
    AscendC::Reg::Abs(a, x, mask);
    AscendC::Reg::Muls(z, a, 0.7071067811865476f, mask);
    AscendC::Reg::Muls(u, z, 0.5f, mask);
    AscendC::Reg::Adds(u, u, 1.0f, mask);
    AscendC::Reg::Div(u, one, u, mask);
    AscendC::Reg::Duplicate(p, 0.17087277f, mask);
    AscendC::Reg::FusedMulDstAdd(p, u, c0, mask);
    AscendC::Reg::FusedMulDstAdd(p, u, c1, mask);
    AscendC::Reg::FusedMulDstAdd(p, u, c2, mask);
    AscendC::Reg::FusedMulDstAdd(p, u, c3, mask);
    AscendC::Reg::FusedMulDstAdd(p, u, c4, mask);
    AscendC::Reg::FusedMulDstAdd(p, u, c5, mask);
    AscendC::Reg::FusedMulDstAdd(p, u, c6, mask);
    AscendC::Reg::FusedMulDstAdd(p, u, c7, mask);
    AscendC::Reg::FusedMulDstAdd(p, u, c8, mask);
    AscendC::Reg::Mul(tmp, z, z, mask);
    AscendC::Reg::Sub(p, p, tmp, mask);
    AscendC::Reg::Exp(e, p, mask);
    AscendC::Reg::Mul(e, e, u, mask);
    AscendC::Reg::Muls(a, a, 0.5f, mask);
    AscendC::Reg::Mul(e, e, a, mask);
    AscendC::Reg::Muls(y, x, 0.5f, mask);
    AscendC::Reg::Add(y, y, a, mask);
    AscendC::Reg::Sub(y, y, e, mask);
    // Positive infinity has GeLU(+inf)=+inf; avoid inf*erfc(inf)=NaN.
    AscendC::Reg::MaskReg positiveInf;
    AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(
        positiveInf, (AscendC::Reg::RegTensor<uint32_t>&)x, 0x7f800000u, mask);
    AscendC::Reg::Select(y, x, y, positiveInf);
    AscendC::Reg::DataCopy(out+i*64, y, mask);
}
'''

_TANH = r'''
auto* in = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
auto* out = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
AscendC::Reg::RegTensor<float> x, s, y;
uint32_t count = $1.GetSize();
uint16_t loops = (count + 63) / 64;
for (uint16_t i=0; i<loops; ++i) {
    auto mask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::DataCopy(x, in+i*64);
    AscendC::Reg::Mul(s, x, x, mask);
    AscendC::Reg::Muls(s, s, -0.0713548163282308f, mask);
    AscendC::Reg::Adds(s, s, -1.5957691216057308f, mask);
    AscendC::Reg::Mul(s, s, x, mask);
    AscendC::Reg::Exp(s, s, mask);
    AscendC::Reg::Adds(s, s, 1.0f, mask);
    AscendC::Reg::Div(y, x, s, mask);
    AscendC::Reg::DataCopy(out+i*64, y, mask);
}
'''

def _typed_math(code, dtype):
    """Generate compile-time register casts, eliminating full-tile FP32 buffers."""
    if dtype == 'float':
        return code
    code = code.replace('__ubuf__ float*', '__ubuf__ ' + dtype + '*')
    decl = '''
static constexpr AscendC::Reg::CastTrait up = {AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait down = {AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
AscendC::Reg::RegTensor<TYPE> low;
'''.replace('TYPE', dtype)
    code = decl + code
    code = code.replace('AscendC::Reg::DataCopy(x, in+i*64);', '''
AscendC::Reg::LoadAlign<TYPE, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(low, in+i*64);
AscendC::Reg::Cast<float, TYPE, up>(x, low, mask);
'''.replace('TYPE', dtype))
    return code.replace('AscendC::Reg::DataCopy(out+i*64, y, mask);', '''
AscendC::Reg::Cast<TYPE, float, down>(low, y, mask);
AscendC::Reg::StoreAlign<TYPE, AscendC::Reg::StoreDist::DIST_PACK_B32>(out+i*64, low, mask);
'''.replace('TYPE', dtype))

_EXACT_F16 = _typed_math(_EXACT, 'half')
_EXACT_BF16 = _typed_math(_EXACT, 'bfloat16_t')
_TANH_F16 = _typed_math(_TANH, 'half')
_TANH_BF16 = _typed_math(_TANH, 'bfloat16_t')

@asctile_jit(reuse_alloc=0)
def _gelu_register(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress, size: int,
                   tile: asc.ConstExpr[int], exact: asc.ConstExpr[bool]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    for t in asctile.range(asctile.block_idx(), asctile.ceildiv(size,tile),
                           asctile.block_num(), unroll_factor=2):
        off = t*tile
        valid = tile if off+tile <= size else size-off
        x0 = asctile.copy_in(x_gm, [off], [tile], real_shape=[valid], pad_value=0)
        if exact:
            if x0.dtype == asc.float16:
                y = asctile.inline_vf(_EXACT_F16, (tile,), x0.dtype, [x0])
            elif x0.dtype == asc.bfloat16:
                y = asctile.inline_vf(_EXACT_BF16, (tile,), x0.dtype, [x0])
            else:
                y = asctile.inline_vf(_EXACT, (tile,), x0.dtype, [x0])
        else:
            if x0.dtype == asc.float16:
                y = asctile.inline_vf(_TANH_F16, (tile,), x0.dtype, [x0])
            elif x0.dtype == asc.bfloat16:
                y = asctile.inline_vf(_TANH_BF16, (tile,), x0.dtype, [x0])
            else:
                y = asctile.inline_vf(_TANH, (tile,), x0.dtype, [x0])
        asctile.copy_out(y, y_gm, [off], real_shape=[valid])

@asctile_jit(vf_fusion=True, reuse_alloc=1)
def _gelu_exact_native(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                       size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    """Iteration03's already measured FP16 exact implementation, unchanged math."""
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    for tile_id in asctile.range(asctile.block_idx(), num_tiles,
                                 asctile.block_num(), unroll_factor=2):
        offset = tile_id * tile_size
        valid = tile_size if offset + tile_size <= size else size - offset
        x = asctile.copy_in(x_gm, [offset], [tile_size], real_shape=[valid], pad_value=0)
        y = x * (asctile.erf(x * 0.7071067811865475) + 1.0) * 0.5
        asctile.copy_out(y, out_gm, [offset], real_shape=[valid])

def gelu(x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    ensure_npu_platform()
    if approximate not in ('none','tanh'): raise ValueError(approximate)
    x=x.contiguous()
    y=torch.empty_like(x)
    size=x.numel()
    if size:
        if approximate == 'none' and x.dtype == torch.float16:
            tiles=asc.ceildiv(size,13824)
            _gelu_exact_native[min(_MAX_CORES,tiles)](x,y,size,tiles,13824)
        else:
            _gelu_register[min(_MAX_CORES,asc.ceildiv(size,_TILE))](x,y,size,_TILE,approximate=='none')
    return y
