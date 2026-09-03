"""CANN Bench RmsNorm interface implemented as a pyasc asctile kernel.

Two-path dispatch:
  - builtin path: asctile.rms_norm on 2-D tiles for aligned D <= 4096;
    bf16 handled via f16 hop (bf16 -> f16 -> rms_norm -> f16 -> bf16).
  - manual path: two-pass streaming for unaligned D or D > 4096.
    Pass 1 accumulates sum(x^2) in f32; pass 2 normalizes y = x * rsqrt * gamma.

All numerical work promotes to f32 internally for precision.
"""

import torch

import asc
import asctile

from ._pyasc_runtime import ensure_npu_platform

_MAX_CORES = 72
_D_BUILTIN_MAX = 4096
_MANUAL_TILE_D = 1024


@asctile.jit
def _rms_norm_builtin_kernel(x_ptr: asc.GlobalAddress,
                              gamma_ptr: asc.GlobalAddress,
                              out_ptr: asc.GlobalAddress,
                              rows: int,
                              D: asc.ConstExpr[int],
                              total_tiles: int,
                              rows_per_tile: asc.ConstExpr[int],
                              epsilon: float):
    x_gm = asctile.global_tensor(x_ptr, [rows, D])
    g_gm = asctile.global_tensor(gamma_ptr, [D])
    o_gm = asctile.global_tensor(out_ptr, [rows, D])
    gamma = asctile.copy_in(g_gm, [0], [D])
    for tile_id in asctile.range(asctile.block_idx(), total_tiles,
                                  asctile.block_num(), unroll_factor=2):
        row = tile_id * rows_per_tile
        active_rows = (rows_per_tile if row + rows_per_tile <= rows
                       else rows - row)
        x = asctile.copy_in(x_gm, [row, 0], [rows_per_tile, D],
                            real_shape=[active_rows, D])
        if x.dtype == asc.bfloat16:
            y = asctile.rms_norm(x.to(asc.float16),
                                 gamma.to(asc.float16),
                                 epsilon).to(asc.bfloat16)
        else:
            y = asctile.rms_norm(x, gamma, epsilon)
        asctile.copy_out(y, o_gm, [row, 0],
                          real_shape=[active_rows, D])


@asctile.jit
def _rms_norm_manual_kernel(x_ptr: asc.GlobalAddress,
                             gamma_ptr: asc.GlobalAddress,
                             out_ptr: asc.GlobalAddress,
                             S: int, D: int, num_d_tiles: int,
                             epsilon: float, inv_D: float,
                             tile_d: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [S * D])
    g_gm = asctile.global_tensor(gamma_ptr, [D])
    o_gm = asctile.global_tensor(out_ptr, [S * D])
    for r in asctile.range(asctile.block_idx(), S, asctile.block_num()):
        row_off = r * D
        acc = asctile.reduce_sum(
            asctile.full([1, 64], 0.0, dtype=asc.float32))
        for dt in asctile.range(num_d_tiles):
            od = dt * tile_d
            n = tile_d if od + tile_d <= D else D - od
            x = asctile.copy_in(x_gm, [row_off + od], [tile_d],
                                real_shape=[n])
            xf = x.to(asc.float32)
            acc = acc + asctile.reduce_sum(xf * xf)
        inv_rms_tile = asctile.rsqrt(
            asctile.full([tile_d], acc * inv_D + epsilon,
                         dtype=asc.float32))
        for dt in asctile.range(num_d_tiles):
            od = dt * tile_d
            n = tile_d if od + tile_d <= D else D - od
            x = asctile.copy_in(x_gm, [row_off + od], [tile_d],
                                real_shape=[n])
            g = asctile.copy_in(g_gm, [od], [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            gf = g.to(asc.float32)
            y = xf * inv_rms_tile * gf
            asctile.copy_out(y.to(x.dtype), o_gm, [row_off + od],
                              real_shape=[n])


def rms_norm(x: torch.Tensor, gamma: torch.Tensor,
             epsilon: float = 1e-6) -> torch.Tensor:
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()
    D = x.shape[-1]
    S = x.numel() // D
    out = torch.empty_like(x)
    if S == 0 or D == 0:
        return out
    esize = x.element_size()
    copy_aligned = (D * esize) % 32 == 0
    use_builtin = copy_aligned and D <= _D_BUILTIN_MAX
    if use_builtin:
        rows_per_tile = max(1, _D_BUILTIN_MAX // D)
        total_tiles = (S + rows_per_tile - 1) // rows_per_tile
        cores = min(_MAX_CORES, total_tiles)
        _rms_norm_builtin_kernel[cores](x, gamma, out, S, D,
                                         total_tiles, rows_per_tile,
                                         epsilon)
    else:
        tile_d = _MANUAL_TILE_D
        num_d_tiles = (D + tile_d - 1) // tile_d
        inv_D = 1.0 / D
        cores = min(_MAX_CORES, S)
        _rms_norm_manual_kernel[cores](x, gamma, out, S, D,
                                        num_d_tiles, epsilon, inv_D,
                                        tile_d)
    return out
