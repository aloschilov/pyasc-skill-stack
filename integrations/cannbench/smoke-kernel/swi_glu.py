"""CANN Bench SwiGlu interface implemented as a pyasc asctile kernel.

Kernel design:
  - Zero-copy 2-D path for aligned layouts: input viewed as [outer, C*inner],
    x0 in first half_cols columns, x1 in second half of each row.
    Grid-stride over (row, col_tile) with [1, TILE] tiles.
  - Short rows use a padded 2-D tile and copy only valid columns directly
    from/to global memory; no host-framework slicing kernel is needed.
  - f16/bf16 promoted to f32 for internal compute, cast back on store.
  - Stable sigmoid: exp(min(s,0)) / (1 + exp(-|s|)); avoids exp overflow.
  - TILE=1024: the silu*glu chain (~13 visible f32 values) fits the UB
    budget with unroll_factor=2 under the measured 1.6x usage factor.
"""

import torch

import asc
import asctile

from ._pyasc_runtime import ensure_npu_platform

_TILE = 1024
_MAX_CORES = 72
_ALIGN_BYTES = 32


@asctile.jit
def _swiglu_2d_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      outer: int, half_cols: int, total_cols: int,
                      num_col_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [outer, total_cols])
    out_gm = asctile.global_tensor(out_ptr, [outer, half_cols])
    total_tiles = outer * num_col_tiles
    for t in asctile.range(asctile.block_idx(), total_tiles, asctile.block_num(),
                        unroll_factor=2):
        row = t // num_col_tiles
        col_tile = t % num_col_tiles
        col_off = col_tile * tile_size
        x1_col_off = half_cols + col_off
        n = tile_size if col_off + tile_size <= half_cols else half_cols - col_off
        x0 = asctile.copy_in(x_gm, [row, col_off], [1, tile_size],
                         real_shape=[1, n])
        x1 = asctile.copy_in(x_gm, [row, x1_col_off], [1, tile_size],
                         real_shape=[1, n])
        x0f = x0.to(asc.float32)
        x1f = x1.to(asc.float32)
        abs_x = asctile.abs(x0f)
        neg_abs = -abs_x
        b = asctile.exp(neg_abs)
        denom = b + 1.0
        min_x = asctile.minimum(x0f, 0.0)
        a = asctile.exp(min_x)
        num = x0f * a
        silu = num / denom
        result = silu * x1f
        asctile.copy_out(result.to(x0.dtype), out_gm, [row, col_off],
                      real_shape=[1, n])


@asctile.jit
def _swiglu_short_rows(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                        outer: int, half_cols: int, total_cols: int,
                        num_tiles: int, rows: asc.ConstExpr[int],
                        cols: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [outer, total_cols])
    out_gm = asctile.global_tensor(out_ptr, [outer, half_cols])
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                          unroll_factor=2):
        row = t * rows
        n = rows if row + rows <= outer else outer - row
        x0 = asctile.copy_in(x_gm, [row, 0], [rows, cols],
                            real_shape=[n, half_cols])
        x1 = asctile.copy_in(x_gm, [row, half_cols], [rows, cols],
                            real_shape=[n, half_cols])
        x0f = x0.to(asc.float32)
        x1f = x1.to(asc.float32)
        b = asctile.exp(-asctile.abs(x0f))
        a = asctile.exp(asctile.minimum(x0f, 0.0))
        silu = (x0f * a) / (b + 1.0)
        result = silu * x1f
        asctile.copy_out(result.to(x0.dtype), out_gm, [row, 0],
                        real_shape=[n, half_cols])


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """SwiGLU activation via a pyasc asctile kernel.

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
        rows = 64
        cols = _ALIGN_BYTES // elem_size
        num_tiles = asc.ceildiv(outer, rows)
        cores = min(_MAX_CORES, num_tiles)
        _swiglu_short_rows[cores](input, out, outer, half_cols, total_cols,
                                  num_tiles, rows, cols)

    return out

