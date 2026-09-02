"""CANN Bench Exp interface implemented as a pyasc asc2 kernel.

Spec formula: y = e^((x*scale + shift) * ln(base)) for base > 0, and the
natural base (ln(base) treated as 1) when base <= 0. Both collapse to a
single fused form y = e^(x*a + b) with host-side constants:
    base >  0: a = scale * ln(base), b = shift * ln(base)
    base <= 0: a = scale,            b = shift
(base == 1 gives a = b = 0, so y = 1 everywhere, matching the spec.)

Kernel design follows pyasc-api-patterns Pattern A (1-D flatten,
grid-stride tile loop, ``real_shape`` tails, f32 internal compute).

Performance tuning:
  - Wider tiles (3072) for f16/bf16 inputs where the UB budget allows
    (4 f32 + 2 f16 temporaries = 20 bytes/elem; at TILE=3072 the
    1.9x-calibrated UB estimate is ~233 KB, safely under the ~254 KB
    budget).  This amortizes per-tile DMA/loop setup by 1.5x versus the
    original TILE=2048.
  - f32 inputs keep TILE=2048 (6 f32 temporaries = 24 bytes/elem; at
    TILE=3072 the estimate would be ~280 KB -- overflow risk).
"""

import math

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE_WIDE = 3072
_TILE_F32 = 2048
_MAX_CORES = 72


@asc2.jit
def _exp_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                size: int, num_tiles: int, a: float, b: float,
                tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        y = asc2.exp(xf * a + b)
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def exp(x: torch.Tensor, base: float = -1.0, scale: float = 1.0,
        shift: float = 0.0) -> torch.Tensor:
    """Element-wise scaled exponential of an NPU tensor via a pyasc kernel."""
    ensure_npu_platform()
    if base > 0:
        ln_base = math.log(base)
        a = scale * ln_base
        b = shift * ln_base
    else:
        a = scale
        b = shift
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    if x.dtype == torch.float32:
        tile = _TILE_F32
    else:
        tile = _TILE_WIDE
    num_tiles = asc.ceildiv(size, tile)
    cores = min(_MAX_CORES, num_tiles)
    _exp_kernel[cores](x, out, size, num_tiles, float(a), float(b), tile)
    return out
