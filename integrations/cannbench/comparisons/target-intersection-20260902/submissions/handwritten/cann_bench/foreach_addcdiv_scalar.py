"""CANNBench adapter for the pyasc v2 target-test Addcdiv kernel.

The tile arithmetic and allocation policy derive from
``python/test/asc2/target/test_addcdiv.py`` and ``helpers.py`` at pyasc commit
4d1db41d.  The adapter adds TensorList iteration, a runtime scalar, safe tails,
and platform setup while retaining target-dtype arithmetic.
"""

import math
from typing import List

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform


_UB_BUDGET = 253952
_UB_RESERVE = 1024
_MAX_CORES = 72


@asc2.jit(reuse_alloc=1)
def _target_addcdiv(input_ptr: asc.GlobalAddress,
                    x1_ptr: asc.GlobalAddress,
                    x2_ptr: asc.GlobalAddress,
                    output_ptr: asc.GlobalAddress,
                    input_length: int, num_tiles: int, scalar: float,
                    tile_length: asc.ConstExpr[int]):
    input_gm = asc2.global_tensor(input_ptr, [input_length])
    x1_gm = asc2.global_tensor(x1_ptr, [input_length])
    x2_gm = asc2.global_tensor(x2_ptr, [input_length])
    output_gm = asc2.global_tensor(output_ptr, [input_length])
    for tile_id in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                              unroll_factor=2):
        current_offset = tile_id * tile_length
        n = (tile_length if current_offset + tile_length <= input_length
             else input_length - current_offset)
        input_t = asc2.copy_in(
            input_gm, [current_offset], [tile_length], real_shape=[n],
            pad_value=0.0)
        x1_t = asc2.copy_in(
            x1_gm, [current_offset], [tile_length], real_shape=[n],
            pad_value=0.0)
        x2_t = asc2.copy_in(
            x2_gm, [current_offset], [tile_length], real_shape=[n],
            pad_value=1.0)
        div_t = x1_t / x2_t
        scaled_t = div_t * scalar
        result = input_t + scaled_t
        asc2.copy_out(result, output_gm, [current_offset], real_shape=[n])


def _target_tile(length: int, itemsize: int) -> int:
    align = 32 // itemsize
    per_buffer = ((_UB_BUDGET - _UB_RESERVE) // itemsize // (4 * 2))
    ub_tile = max(align, (per_buffer // align) * align)
    per_core = math.ceil(length / _MAX_CORES)
    tile = math.ceil(math.ceil(per_core / 2) / align) * align
    tile = max(128, min(tile, ub_tile, math.ceil(length / align) * align))
    return tile


def foreach_addcdiv_scalar(
    x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor],
    scalar: float
) -> List[torch.Tensor]:
    ensure_npu_platform()
    outputs = []
    for index in range(len(x1)):
        input_t = x1[index].contiguous() if not x1[index].is_contiguous() else x1[index]
        numerator = x2[index].contiguous() if not x2[index].is_contiguous() else x2[index]
        denominator = x3[index].contiguous() if not x3[index].is_contiguous() else x3[index]
        output = torch.empty_like(input_t)
        length = input_t.numel()
        if length:
            itemsize = 4 if input_t.dtype == torch.float32 else 2
            tile = _target_tile(length, itemsize)
            num_tiles = asc.ceildiv(length, tile)
            blocks = min(_MAX_CORES, num_tiles)
            _target_addcdiv[blocks](
                input_t, numerator, denominator, output, length, num_tiles,
                scalar, tile)
        outputs.append(output)
    return outputs
