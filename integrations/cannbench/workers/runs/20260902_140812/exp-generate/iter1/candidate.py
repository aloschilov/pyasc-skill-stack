"""CANN Bench Exp interface implemented as a pyasc asc2 kernel.

Kernel design (pyasc-api-patterns Pattern A, grid-stride variant):
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with ``real_shape`` loads/stores (no host padding).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision.
  - Host selects between two compiled tile sizes: wide (3072) when the
    element count fills all 72 cores, narrow (1024) for small shapes.
  - Unified formula: factor = 1.0 (base<=0) or ln(base) (base>0);
    arg = (xf * scale + shift) * factor; y = exp(arg).
  - IEEE inf/nan propagates correctly through asc2.exp; no host branches.
"""

import math

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_WIDE_TILE = 3072
_NARROW_TILE = 1024
_MAX_CORES = 72


@asc2.jit
def _exp_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                size: int, num_tiles: int,
                scale: float, shift: float, factor: float,
                tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        arg = (xf * scale + shift) * factor
        y = asc2.exp(arg)
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def exp(x: torch.Tensor, base: float = -1.0, scale: float = 1.0,
        shift: float = 0.0) -> torch.Tensor:
    """Generalized exponential: y = exp((x * scale + shift) * ln(base))."""
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    if base > 0.0:
        factor = math.log(base)
    else:
        factor = 1.0
    if size >= _MAX_CORES * _WIDE_TILE:
        num_tiles = asc.ceildiv(size, _WIDE_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _exp_kernel[cores](x, out, size, num_tiles, scale, shift, factor,
                           _WIDE_TILE)
    else:
        num_tiles = asc.ceildiv(size, _NARROW_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _exp_kernel[cores](x, out, size, num_tiles, scale, shift, factor,
                           _NARROW_TILE)
    return out
