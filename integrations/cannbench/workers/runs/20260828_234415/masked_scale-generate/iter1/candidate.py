"""CANN Bench MaskedScale interface implemented as a pyasc asc2 kernel.

Kernel design (grid-stride, elementwise):
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with real_shape loads/stores (no host padding).
  - x and mask promoted to f32 for computation; output cast to x dtype.
  - int8/uint8 masks: int8 -> f16 hop -> f32 (int8 tiles reject direct casts);
    uint8 masks are reinterpreted as int8 on the host and corrected in-kernel
    with a +256 fixup for negative (>=128 original) bytes.
  - float16/bf16/f32 masks: direct .to(f32) cast.
  - Host selects tile size: 3072 for short chains (non-uint8), 2048 for
    uint8 (extra where temporaries keep UB under budget).
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_WIDE_TILE = 3072
_UINT8_TILE = 2048
_NARROW_TILE = 1024
_MAX_CORES = 72


@asc2.jit
def _masked_scale_kernel(
    x_ptr: asc.GlobalAddress,
    mask_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    size: int,
    num_tiles: int,
    scale: float,
    tile_size: asc.ConstExpr[int],
    mask_kind: asc.ConstExpr[int],
):
    x_gm = asc2.global_tensor(x_ptr, [size])
    mask_gm = asc2.global_tensor(mask_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        m = asc2.copy_in(mask_gm, [off], [tile_size], real_shape=[n])
        if mask_kind == 0:
            mh = asc2.cast(m, asc.float16)
            mf = mh.to(asc.float32)
        elif mask_kind == 1:
            mh = asc2.cast(m, asc.float16)
            mf = mh.to(asc.float32)
            mf = asc2.where(mf < 0.0, mf + 256.0, mf)
        else:
            mf = m.to(asc.float32)
        y = (xf * mf) * scale
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def masked_scale(
    x: torch.Tensor, mask: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """Element-wise masked scale: y = x * mask * scale, via asc2 kernel."""
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out

    md = mask.dtype
    if md == torch.uint8:
        mask = mask.view(torch.int8)
        mask_kind = 1
        wide = _UINT8_TILE
    elif md == torch.int8:
        mask_kind = 0
        wide = _WIDE_TILE
    elif md == torch.float16:
        mask_kind = 2
        wide = _WIDE_TILE
    elif md == torch.bfloat16:
        mask_kind = 3
        wide = _WIDE_TILE
    else:
        mask_kind = 4
        wide = _WIDE_TILE

    if size >= _MAX_CORES * wide:
        num_tiles = asc.ceildiv(size, wide)
        cores = min(_MAX_CORES, num_tiles)
        _masked_scale_kernel[cores](
            x, mask, out, size, num_tiles, scale, wide, mask_kind
        )
    else:
        num_tiles = asc.ceildiv(size, _NARROW_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _masked_scale_kernel[cores](
            x, mask, out, size, num_tiles, scale, _NARROW_TILE, mask_kind
        )
    return out
