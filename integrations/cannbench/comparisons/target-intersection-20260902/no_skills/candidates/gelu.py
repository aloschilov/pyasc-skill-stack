import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_EXACT_WIDE = 2048
_EXACT_NARROW = 1024
_TANH_WIDE = 1024
_TANH_NARROW = 512
_MAX_CORES = 72


@asc2.jit
def _gelu_exact_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                       size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        z = xf * 0.7071067811865476
        erf_z = asc2.erf(z)
        cdf = (erf_z + 1.0) * 0.5
        result = xf * cdf
        asc2.copy_out(result.to(x.dtype), out_gm, [off], real_shape=[n])


@asc2.jit
def _gelu_tanh_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        x_sq = xf * xf
        inner = xf * (1.0 + 0.044715 * x_sq)
        tanh_arg = 0.7978845608028654 * inner
        tanh_val = asc2.tanh(tanh_arg)
        result = xf * 0.5 * (1.0 + tanh_val)
        neg_large = asc2.full([tile_size], -3.4e38, dtype=asc.float32)
        is_neg_inf = asc2.less(xf, neg_large)
        zero_tile = asc2.full([tile_size], 0.0, dtype=asc.float32)
        result = asc2.where(is_neg_inf, zero_tile, result)
        asc2.copy_out(result.to(x.dtype), out_gm, [off], real_shape=[n])


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    if approximate == "none":
        tile = _EXACT_WIDE if size >= _MAX_CORES * _EXACT_WIDE else _EXACT_NARROW
    else:
        tile = _TANH_WIDE if size >= _MAX_CORES * _TANH_WIDE else _TANH_NARROW
    num_tiles = asc.ceildiv(size, tile)
    cores = min(_MAX_CORES, num_tiles)
    if approximate == "none":
        _gelu_exact_kernel[cores](x, out, size, num_tiles, tile)
    else:
        _gelu_tanh_kernel[cores](x, out, size, num_tiles, tile)
    return out
