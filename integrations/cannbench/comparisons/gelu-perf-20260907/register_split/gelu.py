"""Handwritten AscendC register math embedded through pyasc v2 inline_vf.

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
for (uint16_t i=0; i<loops; ++i) {
    auto mask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::DataCopy(x, in+i*64);
    AscendC::Reg::Abs(a, x, mask);
    AscendC::Reg::Muls(z, a, 0.7071067811865476f, mask);
    AscendC::Reg::Muls(u, z, 0.5f, mask);
    AscendC::Reg::Adds(u, u, 1.0f, mask);
    AscendC::Reg::Div(u, one, u, mask);
    AscendC::Reg::Muls(p, u, 0.17087277f, mask);
    AscendC::Reg::Adds(p, p, -0.82215223f, mask);
    AscendC::Reg::Mul(p, p, u, mask);
    AscendC::Reg::Adds(p, p, 1.48851587f, mask);
    AscendC::Reg::Mul(p, p, u, mask);
    AscendC::Reg::Adds(p, p, -1.13520398f, mask);
    AscendC::Reg::Mul(p, p, u, mask);
    AscendC::Reg::Adds(p, p, 0.27886807f, mask);
    AscendC::Reg::Mul(p, p, u, mask);
    AscendC::Reg::Adds(p, p, -0.18628806f, mask);
    AscendC::Reg::Mul(p, p, u, mask);
    AscendC::Reg::Adds(p, p, 0.09678418f, mask);
    AscendC::Reg::Mul(p, p, u, mask);
    AscendC::Reg::Adds(p, p, 0.37409196f, mask);
    AscendC::Reg::Mul(p, p, u, mask);
    AscendC::Reg::Adds(p, p, 1.00002368f, mask);
    AscendC::Reg::Mul(p, p, u, mask);
    AscendC::Reg::Adds(p, p, -1.26551223f, mask);
    AscendC::Reg::Mul(tmp, z, z, mask);
    AscendC::Reg::Sub(p, p, tmp, mask);
    AscendC::Reg::Exp(e, p, mask);
    AscendC::Reg::Mul(e, e, u, mask);
    AscendC::Reg::Muls(a, a, 0.5f, mask);
    AscendC::Reg::Mul(e, e, a, mask);
    AscendC::Reg::Muls(y, x, 0.5f, mask);
    AscendC::Reg::Add(y, y, a, mask);
    AscendC::Reg::Sub(y, y, e, mask);
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

@asctile_jit(reuse_alloc=0)
def _gelu_register(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress, size: int,
                   tile: asc.ConstExpr[int], exact: asc.ConstExpr[bool]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    for t in asctile.range(asctile.block_idx(), asctile.ceildiv(size,tile),
                           asctile.block_num(), unroll_factor=1):
        off = t*tile
        valid = tile if off+tile <= size else size-off
        x0 = asctile.copy_in(x_gm, [off], [tile], real_shape=[valid], pad_value=0)
        x = x0.to(asc.float32)
        if exact:
            y = asctile.inline_vf(_EXACT, (tile,), asc.float32, [x])
        else:
            y = asctile.inline_vf(_TANH, (tile,), asc.float32, [x])
        asctile.copy_out(y.to(x0.dtype), y_gm, [off], real_shape=[valid])

def gelu(x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    ensure_npu_platform()
    if approximate not in ('none','tanh'): raise ValueError(approximate)
    x=x.contiguous()
    y=torch.empty_like(x)
    size=x.numel()
    if size:
        _gelu_register[min(_MAX_CORES,asc.ceildiv(size,_TILE))](x,y,size,_TILE,approximate=='none')
    return y
