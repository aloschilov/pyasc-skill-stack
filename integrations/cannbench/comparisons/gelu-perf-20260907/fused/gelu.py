"""Handwritten pure-pyasc GeLU: register fusion, single-unroll, exact tails.

Exact mode evaluates a cancellation-free erfc approximation (Numerical Recipes
coefficients), not tanh GeLU. All low-precision inputs accumulate in FP32.
"""
import torch
import asc
import asctile
from ._pyasc_runtime import asctile_jit, ensure_npu_platform

_TILE = 1024
_MAX_CORES = 72

@asctile_jit(vf_fusion=True, reuse_alloc=0)
def _gelu_fused(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress, size: int,
                tile: asc.ConstExpr[int], exact: asc.ConstExpr[bool]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    for t in asctile.range(asctile.block_idx(), asctile.ceildiv(size, tile),
                           asctile.block_num(), unroll_factor=1):
        off = t * tile
        valid = tile if off + tile <= size else size-off
        x0 = asctile.copy_in(x_gm, [off], [tile], real_shape=[valid], pad_value=0)
        x = x0.to(asc.float32)
        if exact:
            z = asctile.abs(x) * 0.7071067811865476
            u = 1.0 / (z * 0.5 + 1.0)
            p = u * 0.17087277 - 0.82215223
            p = p * u + 1.48851587
            p = p * u - 1.13520398
            p = p * u + 0.27886807
            p = p * u - 0.18628806
            p = p * u + 0.09678418
            p = p * u + 0.37409196
            p = p * u + 1.00002368
            p = p * u - 1.26551223
            small = x * 0.5 * u * asctile.exp(p - z*z)
            y = asctile.where(x >= 0.0, x-small, small)
        else:
            s = ((x*x)*0.0713548163282308 + 1.5957691216057308) * x
            y = x / (asctile.exp(-s) + 1.0)
        asctile.copy_out(y.to(x0.dtype), y_gm, [off], real_shape=[valid])

def gelu(x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    ensure_npu_platform()
    if approximate not in ('none', 'tanh'): raise ValueError(approximate)
    x = x.contiguous()
    y = torch.empty_like(x)
    size = x.numel()
    if size:
        _gelu_fused[min(_MAX_CORES, asc.ceildiv(size, _TILE))](x, y, size, _TILE, approximate=='none')
    return y
