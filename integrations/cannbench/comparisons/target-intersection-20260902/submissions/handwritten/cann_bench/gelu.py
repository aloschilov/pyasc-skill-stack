"""CANNBench adapter for the pyasc v2 target-test GELU kernel.

The arithmetic is source-faithful to
``python/test/asc2/target/test_gelu.py`` at pyasc commit 4d1db41d.  The
adapter adds CANNBench tensor allocation, safe tails, and platform setup.  The
upstream target has only its sigmoid-form approximation and therefore ignores
the CANNBench ``approximate`` selector; that coverage gap is intentional in
this baseline.
"""

import math

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform


_TANH_APPROX_FACTOR = 1.0 / 0.044715
_NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715
_UB_BUDGET = 253952
_UB_RESERVE = 1024
_MAX_CORES = 72


@asc2.jit(reuse_alloc=1)
def _target_gelu(input_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress,
                 input_length: int, num_tiles: int,
                 tile_length: asc.ConstExpr[int],
                 tanh_approx_factor: asc.ConstExpr[float],
                 neg_sqrt_eight_over_pi: asc.ConstExpr[float]):
    in_gm = asc2.global_tensor(input_ptr, [input_length])
    out_gm = asc2.global_tensor(output_ptr, [input_length])
    for tile_id in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                              unroll_factor=2):
        current_offset = tile_id * tile_length
        n = (tile_length if current_offset + tile_length <= input_length
             else input_length - current_offset)
        row = asc2.copy_in(
            in_gm, [current_offset], [tile_length], real_shape=[n],
            pad_value=0.0)
        input_sq = row * row
        input_cub = input_sq * row
        input_cub = row + input_cub * tanh_approx_factor
        input_cub = input_cub * neg_sqrt_eight_over_pi
        input_cub = asc2.exp(input_cub)
        input_cub = input_cub + 1
        out = row / input_cub
        asc2.copy_out(out, out_gm, [current_offset], real_shape=[n])


def _target_tile(length: int, itemsize: int) -> int:
    align = 32 // itemsize
    per_buffer = ((_UB_BUDGET - _UB_RESERVE) // itemsize // (4 * 2))
    ub_tile = max(align, (per_buffer // align) * align)
    per_core = math.ceil(length / _MAX_CORES)
    tile = math.ceil(math.ceil(per_core / 2) / align) * align
    tile = max(128, min(tile, ub_tile, math.ceil(length / align) * align))
    return tile


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    ensure_npu_platform()
    source = x.contiguous() if not x.is_contiguous() else x
    output = torch.empty_like(source)
    length = source.numel()
    if length == 0:
        return output
    itemsize = 4 if source.dtype == torch.float32 else 2
    tile = _target_tile(length, itemsize)
    num_tiles = asc.ceildiv(length, tile)
    blocks = min(_MAX_CORES, num_tiles)
    _target_gelu[blocks](
        source, output, length, num_tiles, tile,
        _TANH_APPROX_FACTOR, _NEG_SQRT_EIGHT_OVER_PI)
    return output
