import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE = 1024
_MAX_CORES = 72


@asc2.jit
def _swiglu_2d(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
               outer: int, c_inner: int, half_inner: int,
               total_tiles: int, num_col_tiles: int,
               tile_cols: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [outer, c_inner])
    out_gm = asc2.global_tensor(out_ptr, [outer, half_inner])
    for t in asc2.range(asc2.block_idx(), total_tiles, asc2.block_num(),
                        unroll_factor=2):
        row = t // num_col_tiles
        ct = t - row * num_col_tiles
        col_off = ct * tile_cols
        n = tile_cols if col_off + tile_cols <= half_inner else half_inner - col_off
        x1_off = half_inner + col_off
        x0 = asc2.copy_in(x_gm, [row, col_off], [1, tile_cols], real_shape=[1, n])
        x1 = asc2.copy_in(x_gm, [row, x1_off], [1, tile_cols], real_shape=[1, n])
        x0f = x0.to(asc.float32)
        x1f = x1.to(asc.float32)
        sig = asc2.div(1.0, asc2.exp(-x0f) + 1.0)
        y = x0f * sig * x1f
        asc2.copy_out(y.to(x0.dtype), out_gm, [row, col_off], real_shape=[1, n])


@asc2.jit
def _swiglu_1d(x0_ptr: asc.GlobalAddress, x1_ptr: asc.GlobalAddress,
               out_ptr: asc.GlobalAddress, size: int, num_tiles: int,
               tile_size: asc.ConstExpr[int]):
    x0_gm = asc2.global_tensor(x0_ptr, [size])
    x1_gm = asc2.global_tensor(x1_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x0 = asc2.copy_in(x0_gm, [off], [tile_size], real_shape=[n])
        x1 = asc2.copy_in(x1_gm, [off], [tile_size], real_shape=[n])
        x0f = x0.to(asc.float32)
        x1f = x1.to(asc.float32)
        sig = asc2.div(1.0, asc2.exp(-x0f) + 1.0)
        y = x0f * sig * x1f
        asc2.copy_out(y.to(x0.dtype), out_gm, [off], real_shape=[n])


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    ensure_npu_platform()
    if not input.is_contiguous():
        input = input.contiguous()

    rank = len(input.shape)
    if dim < 0:
        dim = dim + rank

    shape = input.shape
    C = shape[dim]
    half_C = C // 2

    outer = 1
    for i in range(dim):
        outer *= shape[i]
    inner = 1
    for i in range(dim + 1, rank):
        inner *= shape[i]

    half_inner = half_C * inner
    c_inner = C * inner
    total_elements = outer * half_inner

    out_shape = list(shape)
    out_shape[dim] = half_C
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    if total_elements == 0:
        return out

    if half_inner * input.element_size() < 32:
        x0 = input.narrow(dim, 0, half_C).contiguous()
        x1 = input.narrow(dim, half_C, half_C).contiguous()
        num_tiles = asc.ceildiv(total_elements, _TILE)
        cores = min(_MAX_CORES, num_tiles)
        _swiglu_1d[cores](x0, x1, out, total_elements, num_tiles, _TILE)
    else:
        num_col_tiles = asc.ceildiv(half_inner, _TILE)
        total_tiles = outer * num_col_tiles
        cores = min(_MAX_CORES, total_tiles)
        _swiglu_2d[cores](input, out, outer, c_inner, half_inner,
                          total_tiles, num_col_tiles, _TILE)

    return out
