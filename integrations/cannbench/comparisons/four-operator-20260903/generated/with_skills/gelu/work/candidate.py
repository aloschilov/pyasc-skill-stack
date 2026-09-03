"""CANN Bench Gelu interface implemented as a pyasc asctile kernel.

Kernel design (grid-stride variant):
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with ``real_shape`` loads/stores (no host padding).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision.
  - Two separate kernels:
    * _gelu_none_kernel (TILE=512, unroll=2): exact erf-based Gelu using the
      Numerical Recipes erfc Chebyshev fit in cancellation-free form
      ``where(x>=0, x - 0.5*x*erfc(|x|/sqrt(2)), 0.5*x*erfc(|x|/sqrt(2)))``.
    * _gelu_tanh_kernel (TILE=1024, unroll=2): tanh-approximation Gelu using
      the stable-sigmoid identity ``1 + tanh(u) = 2*sigmoid(2u)`` so
      ``y = x * sigmoid(s)`` with ``s = 2*sqrt(2/pi)*(x + 0.044715*x^3)``,
      computed as ``x * exp(min(s,0)) / (1 + exp(-|s|))`` to avoid any
      positive-argument exp overflow.
  - IEEE special values (Inf/NaN) propagate naturally through both formulas,
    matching golden positions without host special-casing.
"""

import torch

import asc
import asctile

from ._pyasc_runtime import ensure_npu_platform

_INV_SQRT_2 = 0.7071067811865476

_ERFC_K1 = 0.17087277
_ERFC_K2 = -0.82215223
_ERFC_K3 = 1.48851587
_ERFC_K4 = -1.13520398
_ERFC_K5 = 0.27886807
_ERFC_K6 = -0.18628806
_ERFC_K7 = 0.09678418
_ERFC_K8 = 0.37409196
_ERFC_K9 = 1.00002368
_ERFC_K10 = -1.26551223

_CUBIC = 0.044715
_TWO_SQRT_2_PI = 1.5957691216057308

_EXACT_TILE = 512
_TANH_TILE = 1024
_MAX_CORES = 72


@asctile.jit
def _gelu_none_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      size: int, num_tiles: int,
                      tile_size: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    one = asctile.full([tile_size], 1.0, dtype=asc.float32)
    for t in asctile.range(asctile.block_idx(), num_tiles,
                           asctile.block_num(), unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        z = asctile.abs(xf) * _INV_SQRT_2
        den = z * 0.5 + 1.0
        t_recip = one / den
        p = t_recip * _ERFC_K1 + _ERFC_K2
        p = p * t_recip + _ERFC_K3
        p = p * t_recip + _ERFC_K4
        p = p * t_recip + _ERFC_K5
        p = p * t_recip + _ERFC_K6
        p = p * t_recip + _ERFC_K7
        p = p * t_recip + _ERFC_K8
        p = p * t_recip + _ERFC_K9
        p = p * t_recip + _ERFC_K10
        erfc_val = t_recip * asctile.exp(p - z * z)
        half_x_erfc = xf * 0.5 * erfc_val
        yf = asctile.where(xf >= 0.0, xf - half_x_erfc, half_x_erfc)
        asctile.copy_out(yf.to(x.dtype), out_gm, [off], real_shape=[n])


@asctile.jit
def _gelu_tanh_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      size: int, num_tiles: int,
                      tile_size: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    zero = asctile.full([tile_size], 0.0, dtype=asc.float32)
    for t in asctile.range(asctile.block_idx(), num_tiles,
                           asctile.block_num(), unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        x3 = xf * xf * xf
        s = (xf + x3 * _CUBIC) * _TWO_SQRT_2_PI
        abs_s = asctile.abs(s)
        den = asctile.exp(-abs_s) + 1.0
        min_s = asctile.minimum(s, zero)
        num = xf * asctile.exp(min_s)
        yf = num / den
        asctile.copy_out(yf.to(x.dtype), out_gm, [off], real_shape=[n])


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Element-wise GELU of an NPU tensor via a pyasc asctile kernel."""
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    if approximate == "none":
        num_tiles = asc.ceildiv(size, _EXACT_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _gelu_none_kernel[cores](x, out, size, num_tiles, _EXACT_TILE)
    elif approximate == "tanh":
        num_tiles = asc.ceildiv(size, _TANH_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _gelu_tanh_kernel[cores](x, out, size, num_tiles, _TANH_TILE)
    else:
        raise ValueError(
            f"approximate must be 'none' or 'tanh', got {approximate!r}"
        )
    return out
