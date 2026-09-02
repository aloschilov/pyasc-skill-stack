"""CANN Bench ForeachAddcdivScalar interface implemented as a pyasc asc2 kernel.

Kernel design (pyasc-api-patterns Pattern A, grid-stride variant):
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with ``real_shape`` loads/stores (no host padding).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision
    (division is precision-sensitive; the spec's golden computes in f32).
  - Host selects between two compiled tile sizes: a wide tile (2048) when
    the element count fills all 72 cores, otherwise a narrow tile (1024) to
    maximize core utilization on small shapes.
  - y = x1 + (x2 / x3) * scalar; IEEE special values (inf/nan) propagate
    through hardware ops, matching the golden's torch semantics.
  - UB budget: 3 input loads + 3 f32 casts + div + mul + add + output cast
    ≈ 10 visible tiles; at TILE=2048 with unroll=2 and 1.6x factor
    ≈ 207 KB < 254 KB.
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_WIDE_TILE = 2048
_NARROW_TILE = 1024
_MAX_CORES = 72


@asc2.jit
def _addcdiv_scalar_kernel(
    x1_ptr: asc.GlobalAddress,
    x2_ptr: asc.GlobalAddress,
    x3_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    size: int,
    num_tiles: int,
    scalar: float,
    tile_size: asc.ConstExpr[int],
):
    x1_gm = asc2.global_tensor(x1_ptr, [size])
    x2_gm = asc2.global_tensor(x2_ptr, [size])
    x3_gm = asc2.global_tensor(x3_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x1 = asc2.copy_in(x1_gm, [off], [tile_size], real_shape=[n])
        x2 = asc2.copy_in(x2_gm, [off], [tile_size], real_shape=[n])
        x3 = asc2.copy_in(x3_gm, [off], [tile_size], real_shape=[n])
        x1f = x1.to(asc.float32)
        x2f = x2.to(asc.float32)
        x3f = x3.to(asc.float32)
        y = x1f + (x2f / x3f) * scalar
        asc2.copy_out(y.to(x1.dtype), out_gm, [off], real_shape=[n])


def foreach_addcdiv_scalar(
    x1: list, x2: list, x3: list, scalar: float
) -> list:
    """Element-wise y_i = x1_i + (x2_i / x3_i) * scalar for each tensor triple."""
    ensure_npu_platform()
    results = []
    for t1, t2, t3 in zip(x1, x2, x3):
        if not t1.is_contiguous():
            t1 = t1.contiguous()
        if not t2.is_contiguous():
            t2 = t2.contiguous()
        if not t3.is_contiguous():
            t3 = t3.contiguous()
        out = torch.empty_like(t1)
        size = t1.numel()
        if size == 0:
            results.append(out)
            continue
        if size >= _MAX_CORES * _WIDE_TILE:
            num_tiles = asc.ceildiv(size, _WIDE_TILE)
            cores = min(_MAX_CORES, num_tiles)
            _addcdiv_scalar_kernel[cores](
                t1, t2, t3, out, size, num_tiles, scalar, _WIDE_TILE)
        else:
            num_tiles = asc.ceildiv(size, _NARROW_TILE)
            cores = min(_MAX_CORES, num_tiles)
            _addcdiv_scalar_kernel[cores](
                t1, t2, t3, out, size, num_tiles, scalar, _NARROW_TILE)
        results.append(out)
    return results
