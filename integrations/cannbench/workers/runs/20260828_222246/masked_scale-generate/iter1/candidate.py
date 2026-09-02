"""CANN Bench MaskedScale interface implemented as a pyasc asc2 kernel.

Kernel design:
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with ``real_shape`` loads/stores (no host padding).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision
    (the spec's precision standard expects f32 internal compute).
  - One kernel source covers all x/mask dtype combinations: the JIT
    specializes per launched-tensor dtype.
  - int8 masks: ``asc2.cast(m, asc.float16)`` then ``.to(asc.float32)``
    (direct int8->f32 fails on this hardware).
  - uint8 masks: host reinterprets bytes as int8 via ``mask.view(torch.int8)``;
    in-kernel after int8->f16->f32, negative values are fixed with
    ``mf = asc2.where(mf < 0.0, mf + 256.0, mf)``.
  - float masks (f16/bf16/f32): direct ``.to(asc.float32)``.
  - mask_kind (ConstExpr) triggers Python-level compile-time branching so
    the JIT traces only the applicable code path.
  - TILE=2048 for int8/float masks; TILE=1024 for uint8 (extra where temporaries).
  - IEEE special values (inf/nan) propagate naturally through hardware ops.
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE_INT = 2048
_TILE_FLOAT = 2048
_TILE_UINT8 = 1024
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
    """mask_kind: 0 = int8, 1 = uint8, 2 = float (f16/bf16/f32)."""
    x_gm = asc2.global_tensor(x_ptr, [size])
    mask_gm = asc2.global_tensor(mask_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(
        asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2
    ):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        m = asc2.copy_in(mask_gm, [off], [tile_size], real_shape=[n])
        if mask_kind == 2:
            mf = m.to(asc.float32)
        else:
            mh = asc2.cast(m, asc.float16)
            mf = mh.to(asc.float32)
            if mask_kind == 1:
                mf = asc2.where(mf < 0.0, mf + 256.0, mf)
        y = xf * mf * scale
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def masked_scale(
    x: torch.Tensor, mask: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """Element-wise masked scale: y = x * mask * scale, via an asc2 kernel."""
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()

    if mask.dtype == torch.int8:
        mask_kind = 0
        tile = _TILE_INT
    elif mask.dtype == torch.uint8:
        mask = mask.view(torch.int8)
        mask_kind = 1
        tile = _TILE_UINT8
    else:
        mask_kind = 2
        tile = _TILE_FLOAT

    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out

    num_tiles = asc.ceildiv(size, tile)
    cores = min(_MAX_CORES, num_tiles)
    _masked_scale_kernel[cores](
        x, mask, out, size, num_tiles, scale, tile, mask_kind
    )
    return out
