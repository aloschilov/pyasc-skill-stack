"""CANN Bench RmsNorm interface implemented as a pyasc asc2 kernel.

Kernel design (proven full-row / split-D two-kernel pattern):
  - Flatten x to (rows, cols) where cols = last dim D. Each row is
    normalized independently along the last dimension.
  - Full-row kernel: when D <= 2048 and D is 32-byte aligned, the whole
    row fits in UB -- single-pass load-reduce-scale-store per row.
  - Split-D kernel: otherwise, stream D in aligned tile chunks with a
    two-pass reduce-then-write per row (loop-carried f32 accumulator).
    Tile width is dtype-aware: 512 for f16/bf16, 1024 for f32 (both
    measured UB-safe on Ascend950PR_9599; TILE=2048 overflows).
  - f16/bf16 inputs are promoted to f32 inside the kernel before
    squaring (prevents f16 x^2 overflow on boundary values like +-65504).
  - Rows are distributed grid-stride across up to 72 AIV cores.
  - Tail tiles handled with real_shape (no host zero-padding); zero
    padding does not bias the reduction since the mean divides by real D.
  - epsilon is a runtime float argument (no per-epsilon recompilation).
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_MAX_CORES = 72
_FULL_ROW_LIMIT = 2048
_SMALL_D_LIMIT = 256
_TILE_SMALL = 64
_TILE_LARGE_F16 = 512
_TILE_LARGE_F32 = 1024


@asc2.jit
def _rms_norm_full_row_kernel(x_ptr: asc.GlobalAddress,
                             gamma_ptr: asc.GlobalAddress,
                             out_ptr: asc.GlobalAddress,
                             rows: int,
                             cols: asc.ConstExpr[int],
                             epsilon: float):
    x_gm = asc2.global_tensor(x_ptr, [rows, cols])
    gamma_gm = asc2.global_tensor(gamma_ptr, [1, cols])
    out_gm = asc2.global_tensor(out_ptr, [rows, cols])
    for row in asc2.range(asc2.block_idx(), rows, asc2.block_num(),
                          unroll_factor=2):
        x_row = asc2.copy_in(x_gm, [row, 0], [1, cols])
        xf = x_row.to(asc.float32)
        sum_sq = asc2.reduce_sum(xf * xf)
        rms = asc2.sqrt(sum_sq / cols + epsilon)
        gamma_row = asc2.copy_in(gamma_gm, [0, 0], [1, cols])
        gf = gamma_row.to(asc.float32)
        y = xf * gf / rms
        asc2.copy_out(y.to(x_row.dtype), out_gm, [row, 0])


@asc2.jit
def _rms_norm_split_d_kernel(x_ptr: asc.GlobalAddress,
                             gamma_ptr: asc.GlobalAddress,
                             out_ptr: asc.GlobalAddress,
                             rows: int,
                             cols: int,
                             num_tiles: int,
                             tile_size: asc.ConstExpr[int],
                             epsilon: float):
    x_gm = asc2.global_tensor(x_ptr, [rows, cols])
    gamma_gm = asc2.global_tensor(gamma_ptr, [1, cols])
    out_gm = asc2.global_tensor(out_ptr, [rows, cols])
    for row in asc2.range(asc2.block_idx(), rows, asc2.block_num(),
                          unroll_factor=2):
        zero_seed = asc2.full([1, tile_size], 0.0, dtype=asc.float32)
        acc = asc2.reduce_sum(zero_seed)
        for tile_id in asc2.range(num_tiles, unroll_factor=2):
            col = tile_id * tile_size
            n = tile_size if col + tile_size <= cols else cols - col
            x = asc2.copy_in(x_gm, [row, col], [1, tile_size],
                             real_shape=[1, n])
            xf = x.to(asc.float32)
            acc = acc + asc2.reduce_sum(xf * xf)
        rms = asc2.sqrt(acc / cols + epsilon)
        for tile_id in asc2.range(num_tiles, unroll_factor=2):
            col = tile_id * tile_size
            n = tile_size if col + tile_size <= cols else cols - col
            x = asc2.copy_in(x_gm, [row, col], [1, tile_size],
                             real_shape=[1, n])
            gamma = asc2.copy_in(gamma_gm, [0, col], [1, tile_size],
                                 real_shape=[1, n])
            y = x.to(asc.float32) * gamma.to(asc.float32) / rms
            asc2.copy_out(y.to(x.dtype), out_gm, [row, col],
                          real_shape=[1, n])


def rms_norm(x: torch.Tensor, gamma: torch.Tensor,
             epsilon: float = 1e-6) -> torch.Tensor:
    """RMS normalization along the last dimension via pyasc asc2 kernels.

    Args:
        x: input tensor of shape (..., D); D is the normalization dim.
        gamma: scaling parameter of shape (D,), same dtype as x.
        epsilon: numerical stability constant (default 1e-6).

    Returns:
        RMS-normalized tensor with the same shape and dtype as x.
    """
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()
    out = torch.empty_like(x)
    total = x.numel()
    if total == 0:
        return out
    cols = x.shape[-1]
    rows = total // cols
    cores = min(_MAX_CORES, rows)
    if x.dtype == torch.float32:
        align = 8
    else:
        align = 16
    eps = float(epsilon)
    if cols <= _FULL_ROW_LIMIT and cols % align == 0:
        _rms_norm_full_row_kernel[cores](x, gamma, out, rows, cols, eps)
    elif cols <= _SMALL_D_LIMIT:
        num_tiles = asc.ceildiv(cols, _TILE_SMALL)
        _rms_norm_split_d_kernel[cores](x, gamma, out, rows, cols,
                                        num_tiles, _TILE_SMALL, eps)
    else:
        if x.dtype == torch.float32:
            tile = _TILE_LARGE_F32
        else:
            tile = _TILE_LARGE_F16
        num_tiles = asc.ceildiv(cols, tile)
        _rms_norm_split_d_kernel[cores](x, gamma, out, rows, cols,
                                        num_tiles, tile, eps)
    return out
