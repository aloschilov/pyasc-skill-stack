"""CANN Bench Gelu interface implemented as a pyasc asctile kernel."""

import math

import torch

import asc
import asctile

from ._pyasc_runtime import ensure_npu_platform

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_TWO_SQRT_2_OVER_PI = 2.0 * math.sqrt(2.0 / math.pi)
_GELU_C = 0.044715
_GC_T = _GELU_C * _TWO_SQRT_2_OVER_PI
_MAX_CORES = 72


@asctile.jit
def _gelu_erf_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                           unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x_t = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x_t.to(asc.float32)
        z = asctile.abs(xf) * _INV_SQRT2
        den = z * 0.5 + 1.0
        one_tile = asctile.full([tile_size], 1.0, dtype=asc.float32)
        tt = one_tile / den
        p = tt * 0.17087277 - 0.82215223
        p = p * tt + 1.48851587
        p = p * tt - 1.13520398
        p = p * tt + 0.27886807
        p = p * tt - 0.18628806
        p = p * tt + 0.09678418
        p = p * tt + 0.37409196
        p = p * tt + 1.00002368
        p = p * tt - 1.26551223
        erfc_z = tt * asctile.exp(p - z * z)
        half_erfc = erfc_z * 0.5
        y_neg = xf * half_erfc
        y_pos = xf - y_neg
        y = asctile.where(xf >= 0.0, y_pos, y_neg)
        asctile.copy_out(y.to(x_t.dtype), out_gm, [off], real_shape=[n])


@asctile.jit
def _gelu_erf_kernel_u1(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                         size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                           unroll_factor=1):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x_t = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x_t.to(asc.float32)
        z = asctile.abs(xf) * _INV_SQRT2
        den = z * 0.5 + 1.0
        one_tile = asctile.full([tile_size], 1.0, dtype=asc.float32)
        tt = one_tile / den
        p = tt * 0.17087277 - 0.82215223
        p = p * tt + 1.48851587
        p = p * tt - 1.13520398
        p = p * tt + 0.27886807
        p = p * tt - 0.18628806
        p = p * tt + 0.09678418
        p = p * tt + 0.37409196
        p = p * tt + 1.00002368
        p = p * tt - 1.26551223
        erfc_z = tt * asctile.exp(p - z * z)
        half_erfc = erfc_z * 0.5
        y_neg = xf * half_erfc
        y_pos = xf - y_neg
        y = asctile.where(xf >= 0.0, y_pos, y_neg)
        asctile.copy_out(y.to(x_t.dtype), out_gm, [off], real_shape=[n])


@asctile.jit
def _gelu_tanh_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                       size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                           unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x_t = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x_t.to(asc.float32)
        x2 = xf * xf
        s = (x2 * _GC_T + _TWO_SQRT_2_OVER_PI) * xf
        min_s = asctile.minimum(s, 0.0)
        abs_s = asctile.abs(s)
        neg_abs_s = -abs_s
        exp_num = asctile.exp(min_s)
        exp_den = asctile.exp(neg_abs_s)
        den = exp_den + 1.0
        sig = exp_num / den
        y = xf * sig
        asctile.copy_out(y.to(x_t.dtype), out_gm, [off], real_shape=[n])


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out

    if approximate == "none":
        tile_sz = 1024
        num_tiles = asc.ceildiv(size, tile_sz)
        cores = min(_MAX_CORES, num_tiles)
        _gelu_erf_kernel_u1[cores](x, out, size, num_tiles, tile_sz)
    else:
        tile_sz = 1024
        num_tiles = asc.ceildiv(size, tile_sz)
        cores = min(_MAX_CORES, num_tiles)
        _gelu_tanh_kernel[cores](x, out, size, num_tiles, tile_sz)

    return out
