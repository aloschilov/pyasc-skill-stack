"""CANN Bench SwiGlu interface implemented as a pyasc asc2 kernel.

Kernel design (2-D row+column tiled, grid-stride):
  - The contiguous input is viewed as [outer, C*inner] where
    outer = prod(shape[:dim]), C = shape[dim], inner = prod(shape[dim+1:]).
    x0 occupies columns [0, half_cols) and x1 columns [half_cols, 2*half_cols)
    of every row, half_cols = (C//2)*inner. Output is [outer, half_cols],
    whose contiguous layout exactly matches the golden's output, so no
    on-device copies are needed to materialize x0 / x1.
  - Work is tiled in 2-D blocks [rows_per_tile, cols_per_tile] and iterated
    grid-stride over (row_block, col_block) so small-outer shapes still
    saturate cores. Three compile-time tile shapes are selected on the host:
    (1, 1024)  for half_cols >= 1024  (column tiling, long contiguous runs),
    (64, 16)   for 2 <= half_cols < 1024 (row packing, full 1024-elem tiles),
    (1024, 1)  for half_cols == 1 (row packing the strided single-column case).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision.
  - silu(v) = v * sigmoid(v). sigmoid uses the cancellation-free stable form
    sig = exp(min(s,0)) / (1 + exp(-|s|)); both exp arguments are non-positive
    so neither can overflow. min(s,0) uses asc2.minimum (NOT the arithmetic
    (s-|s|)*0.5 form, which yields inf-inf=NaN for s=+inf and would plant NaN
    at every +inf input where the golden produces a finite +/-inf). To match
    the golden's 1/(1+exp(-s)) bit-for-bit where exp(-s) overflows f32
    (s < -88.7228), sigmoid is clamped to 0 there via asc2.where.
  - UB budget: ~12 visible f32 temporaries * 4 * 1024 * 2(unroll) * 1.6
    ~ 188 KB, under the ~253952 byte static limit.
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_MAX_CORES = 72


@asc2.jit
def _swiglu_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                   outer: int, half_cols: int, total_cols: int,
                   num_row_tiles: int, num_col_tiles: int,
                   rows_per_tile: asc.ConstExpr[int],
                   cols_per_tile: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [outer, total_cols])
    out_gm = asc2.global_tensor(out_ptr, [outer, half_cols])
    total_tiles = num_row_tiles * num_col_tiles
    for t in asc2.range(asc2.block_idx(), total_tiles, asc2.block_num(),
                        unroll_factor=2):
        rb = t // num_col_tiles
        ct = t % num_col_tiles
        r0 = rb * rows_per_tile
        c0 = ct * cols_per_tile
        rows = rows_per_tile if r0 + rows_per_tile <= outer else outer - r0
        cols = (cols_per_tile if c0 + cols_per_tile <= half_cols
                else half_cols - c0)
        x0 = asc2.copy_in(x_gm, [r0, c0], [rows_per_tile, cols_per_tile],
                          real_shape=[rows, cols])
        x1 = asc2.copy_in(x_gm, [r0, c0 + half_cols],
                          [rows_per_tile, cols_per_tile],
                          real_shape=[rows, cols])
        s = x0.to(asc.float32)
        x1f = x1.to(asc.float32)
        zero_tile = asc2.full([rows_per_tile, cols_per_tile], 0.0,
                              dtype=asc.float32)
        abs_s = asc2.abs(s)
        exp_neg_abs = asc2.exp(-abs_s)
        min_s = asc2.minimum(s, zero_tile)
        sig = asc2.exp(min_s) / (exp_neg_abs + 1.0)
        sig_clamped = asc2.where(s >= -88.7228, sig, zero_tile)
        out = (s * sig_clamped) * x1f
        asc2.copy_out(out.to(x0.dtype), out_gm, [r0, c0],
                      real_shape=[rows, cols])


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """SwiGLU activation of an NPU tensor via a pyasc asc2 kernel.

    output = silu(x0) * x1 where (x0, x1) = chunk(input, 2, dim) and
    silu(v) = v * sigmoid(v).
    """
    ensure_npu_platform()
    if not input.is_contiguous():
        input = input.contiguous()
    shape = tuple(input.shape)
    rank = len(shape)
    total = input.numel()
    if rank == 0 or total == 0:
        return torch.empty(shape, dtype=input.dtype, device=input.device)
    d = dim if dim >= 0 else dim + rank
    if d < 0 or d >= rank:
        raise ValueError(
            "SwiGlu: dim {} out of range for rank {}".format(dim, rank))
    c_dim = shape[d]
    if c_dim % 2 != 0:
        raise ValueError(
            "SwiGlu: size along dim {} must be even, got {}".format(dim, c_dim))
    outer = 1
    for i in range(d):
        outer *= shape[i]
    inner = 1
    for i in range(d + 1, rank):
        inner *= shape[i]
    half_cols = (c_dim // 2) * inner
    total_cols = c_dim * inner
    out_shape = list(shape)
    out_shape[d] = c_dim // 2
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    if outer * half_cols == 0:
        return out
    if half_cols >= 1024:
        rows_per_tile, cols_per_tile = 1, 1024
    elif half_cols >= 2:
        rows_per_tile, cols_per_tile = 64, 16
    else:
        rows_per_tile, cols_per_tile = 1024, 1
    num_row_tiles = asc.ceildiv(outer, rows_per_tile)
    num_col_tiles = asc.ceildiv(half_cols, cols_per_tile)
    total_tiles = num_row_tiles * num_col_tiles
    cores = min(_MAX_CORES, total_tiles)
    _swiglu_kernel[cores](input, out, outer, half_cols, total_cols,
                          num_row_tiles, num_col_tiles,
                          rows_per_tile, cols_per_tile)
    return out
