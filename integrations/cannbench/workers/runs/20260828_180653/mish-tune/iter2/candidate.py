"""CANN Bench Mish interface implemented as a pyasc asc2 kernel.

mish(x) = x * tanh(softplus(x)). Computing softplus then tanh naively
loses all precision for x << 0: log(1 + e^x) rounds 1 + e^x to 1.0 once
x < -16 in f32, collapsing the output to 0 while the true value is a
small nonzero number (harness cases with value ranges like [-88, 88]
fail their f32 relative-error check).

Instead use the exact algebraic identity (w = e^(-|x|), no cancellation,
every addition operates on same-sign terms):

  tanh(softplus(x)) = (1 + 2w) / (1 + 2w + 2w^2)   for x >= 0
                    = (w^2 + 2w) / (w^2 + 2w + 2)  for x <  0

derived from tanh(log(z)) = (z^2 - 1) / (z^2 + 1) with z = 1 + e^x
(negative branch) and the same identity scaled by e^(-2x) (positive
branch). e^() never sees a positive argument, so no f32 overflow.

The two branches share the sub-expression a = 2w and b = w^2. Factoring
numerator and denominator through `a` lets each be produced by a single
`asc2.where` blend that reuses the same live tiles, instead of four
distinct branch tiles held alive until a final quotient-select. This
cuts the peak simultaneous unified-buffer footprint enough to run a
TILE=2048 wide tile (vs 1024), which amortises per-tile DMA/loop setup
and keeps the AIV vector pipeline fed. The numerics are identical to
the four-branch spelling (same w, same divisions, same blend condition).

Kernel design follows pyasc-api-patterns Pattern A (1-D flatten,
grid-stride tile loop, ``real_shape`` tails, f32 internal compute).
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE = 2048
_MAX_CORES = 72


@asc2.jit
def _mish_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                 size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        w = asc2.exp(-asc2.abs(xf))
        a = w * 2.0              # 2w          (shared by both branches)
        b = w * w                # w^2         (shared by both branches)
        cond = xf >= 0.0
        one = asc2.full([tile_size], 1.0, dtype=asc.float32)
        # num = 1 + 2w (x>=0)  |  w^2 + 2w (x<0)
        num = a + asc2.where(cond, one, b)
        # den = num + 2w^2 (x>=0)  |  num + 2 (x<0)
        den = num + (asc2.where(cond, b, one) * 2.0)
        tanh_sp = num / den
        y = xf * tanh_sp
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def mish(x: torch.Tensor) -> torch.Tensor:
    """Element-wise Mish activation of an NPU tensor via a pyasc asc2 kernel."""
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    num_tiles = asc.ceildiv(size, _TILE)
    cores = min(_MAX_CORES, num_tiles)
    _mish_kernel[cores](x, out, size, num_tiles, _TILE)
    return out
