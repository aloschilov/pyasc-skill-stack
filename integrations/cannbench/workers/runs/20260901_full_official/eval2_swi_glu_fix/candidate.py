"""CANN Bench SwiGlu interface implemented as a pyasc asc2 kernel.

Kernel design:
  - Zero-copy 2-D path for aligned layouts: input viewed as [outer, C*inner],
    x0 in first half_cols columns, x1 in second half of each row.
    Grid-stride over (row, col_tile) with [1, TILE] tiles.
  - Padded 1-D-GM fallback for degenerate layouts
    (half_cols * elem_size < 32): one row per iteration, avoiding host-side
    narrow/contiguous and the CANN Slice dependency.
  - f16/bf16 promoted to f32 for internal compute, cast back on store.
  - Stable sigmoid: exp(min(s,0)) / (1 + exp(-|s|)); avoids exp overflow.
  - TILE=1024: the silu*glu chain (~13 visible f32 values) fits the UB
    budget with unroll_factor=2 under the measured 1.6x usage factor.
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE = 1024
_MAX_CORES = 72
_ALIGN_BYTES = 32
_SMALL_ROW_TILE = 16


@asc2.jit
def _swiglu_2d_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      outer: int, half_cols: int, total_cols: int,
                      num_col_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [outer, total_cols])
    out_gm = asc2.global_tensor(out_ptr, [outer, half_cols])
    total_tiles = outer * num_col_tiles
    for t in asc2.range(asc2.block_idx(), total_tiles, asc2.block_num(),
                        unroll_factor=2):
        row = t // num_col_tiles
        col_tile = t % num_col_tiles
        col_off = col_tile * tile_size
        x1_col_off = half_cols + col_off
        n = tile_size if col_off + tile_size <= half_cols else half_cols - col_off
        x0 = asc2.copy_in(x_gm, [row, col_off], [1, tile_size],
                         real_shape=[1, n])
        x1 = asc2.copy_in(x_gm, [row, x1_col_off], [1, tile_size],
                         real_shape=[1, n])
        x0f = x0.to(asc.float32)
        x1f = x1.to(asc.float32)
        abs_x = asc2.abs(x0f)
        neg_abs = -abs_x
        b = asc2.exp(neg_abs)
        denom = b + 1.0
        min_x = asc2.minimum(x0f, 0.0)
        a = asc2.exp(min_x)
        num = x0f * a
        silu = num / denom
        result = silu * x1f
        asc2.copy_out(result.to(x0.dtype), out_gm, [row, col_off],
                      real_shape=[1, n])


@asc2.jit
def _swiglu_small_rows_kernel(x_ptr: asc.GlobalAddress,
                              out_ptr: asc.GlobalAddress,
                              outer: int,
                              half_cols: int,
                              total_cols: int,
                              tile_size: asc.ConstExpr[int]):
    """Process sub-32-byte rows without a host Slice/contiguous fallback."""
    x_gm = asc2.global_tensor(x_ptr, [outer * total_cols])
    out_gm = asc2.global_tensor(out_ptr, [outer * half_cols])
    for row in asc2.range(asc2.block_idx(), outer, asc2.block_num(),
                          unroll_factor=2):
        x_off = row * total_cols
        out_off = row * half_cols
        x0 = asc2.copy_in(
            x_gm, [x_off], [tile_size], real_shape=[half_cols]
        )
        x1 = asc2.copy_in(
            x_gm, [x_off + half_cols], [tile_size], real_shape=[half_cols]
        )
        x0f = x0.to(asc.float32)
        x1f = x1.to(asc.float32)
        abs_x = asc2.abs(x0f)
        neg_abs = -abs_x
        b = asc2.exp(neg_abs)
        denom = b + 1.0
        min_x = asc2.minimum(x0f, 0.0)
        a = asc2.exp(min_x)
        num = x0f * a
        silu = num / denom
        result = silu * x1f
        asc2.copy_out(
            result.to(x0.dtype), out_gm, [out_off], real_shape=[half_cols]
        )


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """SwiGLU activation via a pyasc asc2 kernel.

    output = silu(x0) * x1  where  (x0, x1) = input.chunk(2, dim)
    and silu(v) = v * sigmoid(v) with the stable sigmoid form
    exp(min(v, 0)) / (1 + exp(-|v|)) to avoid exp overflow.
    """
    ensure_npu_platform()
    if not input.is_contiguous():
        input = input.contiguous()

    ndim = input.dim()
    if dim < 0:
        dim = dim + ndim

    shape = input.shape
    C = shape[dim]
    half_C = C // 2

    outer = 1
    for i in range(dim):
        outer *= shape[i]
    inner = 1
    for i in range(dim + 1, ndim):
        inner *= shape[i]

    half_cols = half_C * inner
    total_cols = C * inner

    out_shape = list(shape)
    out_shape[dim] = half_C
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    size = out.numel()
    if size == 0:
        return out

    if input.dtype == torch.float32:
        elem_size = 4
    else:
        elem_size = 2

    if half_cols * elem_size >= _ALIGN_BYTES:
        num_col_tiles = asc.ceildiv(half_cols, _TILE)
        total_tiles = outer * num_col_tiles
        cores = min(_MAX_CORES, total_tiles)
        _swiglu_2d_kernel[cores](input, out, outer, half_cols, total_cols,
                                 num_col_tiles, _TILE)
    else:
        cores = min(_MAX_CORES, outer)
        _swiglu_small_rows_kernel[cores](
            input, out, outer, half_cols, total_cols, _SMALL_ROW_TILE
        )

    return out

