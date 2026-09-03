import torch
from typing import List

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_WIDE_TILE = 2048
_NARROW_TILE = 1024
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

        div = x2f / x3f
        mul = div * scalar
        res = x1f + mul

        asc2.copy_out(res.to(x1.dtype), out_gm, [off], real_shape=[n])


def foreach_addcdiv_scalar(
    x1: List[torch.Tensor], x2: List[torch.Tensor],
    x3: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    ensure_npu_platform()
    input_dtype = x1[0].dtype if x1 else torch.float32
    tile = _WIDE_TILE if input_dtype in (torch.float16, torch.bfloat16) else _NARROW_TILE
    results = []
    for i in range(len(x1)):
        t1 = x1[i].contiguous()
        t2 = x2[i].contiguous()
        t3 = x3[i].contiguous()
        out = torch.empty_like(t1)
        size = t1.numel()
        if size == 0:
            results.append(out)
            continue
        num_tiles = asc.ceildiv(size, tile)
        cores = min(_MAX_CORES, num_tiles)
        _addcdiv_kernel[cores](t1, t2, t3, out, size, num_tiles, float(scalar), tile)
        results.append(out)
    return results
