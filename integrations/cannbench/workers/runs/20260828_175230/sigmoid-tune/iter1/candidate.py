"""CANN Bench Sigmoid interface implemented as a pyasc asc2 kernel.

Kernel design (pyasc-api-patterns Pattern A, grid-stride variant):
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with ``real_shape`` loads/stores (no host padding).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision
    (the spec's precision standard expects f32 internal compute).
  - sigmoid(x) = 1 / (1 + e^(-x)); IEEE saturation gives the correct
    limits at extreme inputs (e^inf -> inf -> y=0, e^-inf -> 0 -> y=1).
  - TILE=4096: sigmoid's chain is short (peak ~4 live f32 temporaries),
    so a 4096-wide tile fits the ~254KB UB budget with comfortable margin
    while doubling the DMA chunk size vs 2048 (better HBM bandwidth
    utilization on the large/bf16 cases that were bandwidth-limited).
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE = 4096
_MAX_CORES = 72


@asc2.jit
def _sigmoid_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                    size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        y = asc2.div(1.0, asc2.exp(-xf) + 1.0)
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Element-wise sigmoid of an NPU tensor via a pyasc asc2 kernel."""
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    num_tiles = asc.ceildiv(size, _TILE)
    cores = min(_MAX_CORES, num_tiles)
    _sigmoid_kernel[cores](x, out, size, num_tiles, _TILE)
    return out
