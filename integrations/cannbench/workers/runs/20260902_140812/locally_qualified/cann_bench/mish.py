"""CANN Bench Mish interface implemented as a pyasc asc2 kernel.

Kernel design (pyasc-api-patterns Pattern A, grid-stride variant):
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with ``real_shape`` loads/stores (no host padding).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision.
  - TILE=1024 fits the UB budget for the 15-value Mish chain.
  - mish(x) = x * tanh(softplus(x)) computed via cancellation-free identity:
    w = exp(-|x|), then
      x >= 0: tanh_sp = (1 + 2w) / (1 + 2w + 2w^2)
      x < 0:  tanh_sp = (w^2 + 2w) / (w^2 + 2w + 2)
    No exp() sees a positive argument; no log(1+tiny) appears.
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE_SIZE = 1024
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

        abs_xf = asc2.abs(xf)
        neg_abs = -abs_xf
        w = asc2.exp(neg_abs)
        w2 = w * w
        two_w = w + w
        two_w2 = w2 + w2

        pos_num = two_w + 1.0
        pos_den = pos_num + two_w2
        pos_th = pos_num / pos_den

        neg_num = w2 + two_w
        neg_den = neg_num + 2.0
        neg_th = neg_num / neg_den

        cond = xf >= 0.0
        th = asc2.where(cond, pos_th, neg_th)
        y = xf * th

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
    num_tiles = asc.ceildiv(size, _TILE_SIZE)
    cores = min(_MAX_CORES, num_tiles)
    _mish_kernel[cores](x, out, size, num_tiles, _TILE_SIZE)
    return out
