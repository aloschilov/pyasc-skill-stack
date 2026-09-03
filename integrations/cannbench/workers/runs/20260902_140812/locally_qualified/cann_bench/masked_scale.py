"""CANN Bench MaskedScale interface implemented as a pyasc asc2 kernel.

y = x * mask * scale, elementwise.  Three kernel variants handle the
independent dtype axes:
  - _masked_scale_float_kernel  : float masks (f16 / bf16 / f32)
  - _masked_scale_int8_kernel   : signed int8 masks (cast hop via f16)
  - _masked_scale_uint8_kernel  : unsigned int8 masks (host view as int8,
                                  +256 fixup in f32 via asc2.where)
All compute is f32 inside the kernel; output is cast back to x's dtype.
1-D grid-stride tiling with TILE=2048 and unroll_factor=2.
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE = 2048
_MAX_CORES = 72


@asc2.jit
def _masked_scale_float_kernel(
    x_ptr: asc.GlobalAddress, mask_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress, size: int, num_tiles: int,
    scale: float, tile_size: asc.ConstExpr[int],
):
    x_gm = asc2.global_tensor(x_ptr, [size])
    mask_gm = asc2.global_tensor(mask_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x_tile = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        m_tile = asc2.copy_in(mask_gm, [off], [tile_size], real_shape=[n])
        xf = x_tile.to(asc.float32)
        mf = m_tile.to(asc.float32)
        y = xf * mf * scale
        asc2.copy_out(y.to(x_tile.dtype), out_gm, [off], real_shape=[n])


@asc2.jit
def _masked_scale_int8_kernel(
    x_ptr: asc.GlobalAddress, mask_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress, size: int, num_tiles: int,
    scale: float, tile_size: asc.ConstExpr[int],
):
    x_gm = asc2.global_tensor(x_ptr, [size])
    mask_gm = asc2.global_tensor(mask_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x_tile = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        m_tile = asc2.copy_in(mask_gm, [off], [tile_size], real_shape=[n])
        xf = x_tile.to(asc.float32)
        mf = asc2.cast(m_tile, asc.float16).to(asc.float32)
        y = xf * mf * scale
        asc2.copy_out(y.to(x_tile.dtype), out_gm, [off], real_shape=[n])


@asc2.jit
def _masked_scale_uint8_kernel(
    x_ptr: asc.GlobalAddress, mask_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress, size: int, num_tiles: int,
    scale: float, tile_size: asc.ConstExpr[int],
):
    x_gm = asc2.global_tensor(x_ptr, [size])
    mask_gm = asc2.global_tensor(mask_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x_tile = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        m_tile = asc2.copy_in(mask_gm, [off], [tile_size], real_shape=[n])
        xf = x_tile.to(asc.float32)
        mf = asc2.cast(m_tile, asc.float16).to(asc.float32)
        mf = asc2.where(mf < 0.0, mf + 256.0, mf)
        y = xf * mf * scale
        asc2.copy_out(y.to(x_tile.dtype), out_gm, [off], real_shape=[n])


def masked_scale(
    x: torch.Tensor, mask: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """Compute y = x * mask * scale elementwise on NPU tensors."""
    ensure_npu_platform()
    x = x.contiguous()
    mask = mask.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out

    num_tiles = asc.ceildiv(size, _TILE)
    cores = min(_MAX_CORES, num_tiles)

    if mask.dtype == torch.uint8:
        mask = mask.view(torch.int8)
        _masked_scale_uint8_kernel[cores](x, mask, out, size, num_tiles,
                                          scale, _TILE)
    elif mask.dtype == torch.int8:
        _masked_scale_int8_kernel[cores](x, mask, out, size, num_tiles,
                                         scale, _TILE)
    else:
        _masked_scale_float_kernel[cores](x, mask, out, size, num_tiles,
                                          scale, _TILE)

    return out
