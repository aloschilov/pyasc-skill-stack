"""CANN Bench MaskedScale interface implemented as a pyasc asc2 kernel.

Kernel design (grid-stride tile loop):
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with real_shape loads/stores (no host padding).
  - f16/bf16 inputs promoted to f32 inside the kernel for precision.
  - uint8 mask reinterpreted as int8 on host (view, no copy) per op
    guidance; for 0/1 masks this is value-preserving.
  - y = (x * mask) * scale; IEEE mul propagates inf/nan like golden.
  - Wide tile (2048) / narrow tile (1024) host selection.
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_WIDE_TILE = 2048
_NARROW_TILE = 1024
_MAX_CORES = 72


@asc2.jit
def _masked_scale_kernel(x_ptr: asc.GlobalAddress,
                         mask_ptr: asc.GlobalAddress,
                         out_ptr: asc.GlobalAddress,
                         size: int,
                         num_tiles: int,
                         scale: float,
                         tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    m_gm = asc2.global_tensor(mask_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        m = asc2.copy_in(m_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        mf = m.to(asc.float32)
        y = (xf * mf) * scale
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def masked_scale(x: torch.Tensor, mask: torch.Tensor,
                 scale: float = 1.0) -> torch.Tensor:
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    if mask.dtype == torch.uint8:
        mask = mask.view(torch.int8)
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    if size >= _MAX_CORES * _WIDE_TILE:
        num_tiles = asc.ceildiv(size, _WIDE_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _masked_scale_kernel[cores](x, mask, out, size, num_tiles,
                                    scale, _WIDE_TILE)
    else:
        num_tiles = asc.ceildiv(size, _NARROW_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _masked_scale_kernel[cores](x, mask, out, size, num_tiles,
                                    scale, _NARROW_TILE)
    return out
