"""CANN Bench ForeachAddcdivScalar interface implemented as a pyasc asc2 kernel.

Kernel design (pyasc-api-patterns Pattern A, grid-stride variant):
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with ``real_shape`` loads/stores (no host padding).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision
    (the spec's precision standard expects f32 internal compute).
  - Host loops over list entries, one kernel launch per (x1_i, x2_i, x3_i)
    triple, allocating output per triple.
  - y_i = x1_i + (x2_i / x3_i) * scalar; IEEE propagation handles
    inf/nan scalars and inputs correctly without special-casing.
"""

import torch
from typing import List

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE_SIZE = 1024
_MAX_CORES = 72


@asc2.jit
def _addcdiv_kernel(x1_ptr: asc.GlobalAddress, x2_ptr: asc.GlobalAddress,
                    x3_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                    size: int, num_tiles: int, scalar: float,
                    tile_size: asc.ConstExpr[int]):
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
        div_tile = x2f / x3f
        scaled = div_tile * scalar
        result_f32 = x1f + scaled
        asc2.copy_out(result_f32.to(x1.dtype), out_gm, [off], real_shape=[n])


def foreach_addcdiv_scalar(
    x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor],
    scalar: float
) -> List[torch.Tensor]:
    ensure_npu_platform()
    y = []
    for i in range(len(x1)):
        a = x1[i]
        b = x2[i]
        c = x3[i]
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        if not c.is_contiguous():
            c = c.contiguous()
        out = torch.empty_like(a)
        size = a.numel()
        if size == 0:
            y.append(out)
            continue
        num_tiles = asc.ceildiv(size, _TILE_SIZE)
        cores = min(_MAX_CORES, num_tiles)
        _addcdiv_kernel[cores](a, b, c, out, size, num_tiles, scalar, _TILE_SIZE)
        y.append(out)
    return y
