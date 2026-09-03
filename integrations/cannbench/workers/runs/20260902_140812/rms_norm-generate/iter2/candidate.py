"""CANN Bench RmsNorm interface implemented as a pyasc asc2 kernel.

Kernel design (two-pass streaming per row):
  - Pass 1: stream D in TILE_D chunks, accumulate sum-of-squares in f32 scalar.
  - Scalar compute: inv_rms = rsqrt(acc * inv_D + epsilon).
  - Pass 2: re-stream D chunks, reload x and gamma, emit y = x * inv_rms * gamma.
  - f16/bf16 inputs promoted to f32 inside the kernel for precision.
  - Grid-stride outer loop over rows across 72 AIV cores.
  - TILE_D = 1024 keeps UB well under 253 952 bytes (single-buffered, no unroll).
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE_D = 1024
_MAX_CORES = 72


@asc2.jit
def _rms_norm_kernel(x_ptr: asc.GlobalAddress, gamma_ptr: asc.GlobalAddress,
                     out_ptr: asc.GlobalAddress,
                     S: int, D: int, num_d_tiles: int,
                     epsilon: float, inv_D: float,
                     tile_d: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [S * D])
    g_gm = asc2.global_tensor(gamma_ptr, [D])
    o_gm = asc2.global_tensor(out_ptr, [S * D])

    for r in asc2.range(asc2.block_idx(), S, asc2.block_num()):
        row = r * D
        acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
        for dt in asc2.range(num_d_tiles):
            od = dt * tile_d
            n = tile_d if od + tile_d <= D else D - od
            x = asc2.copy_in(x_gm, [row + od], [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            acc = acc + asc2.reduce_sum(xf * xf)
        inv_rms = asc2.rsqrt(acc * inv_D + epsilon)

        for dt in asc2.range(num_d_tiles):
            od = dt * tile_d
            n = tile_d if od + tile_d <= D else D - od
            x = asc2.copy_in(x_gm, [row + od], [tile_d], real_shape=[n])
            g = asc2.copy_in(g_gm, [od], [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            gf = g.to(asc.float32)
            y = xf * inv_rms * gf
            asc2.copy_out(y.to(x.dtype), o_gm, [row + od], real_shape=[n])


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
    tile_d = _TILE_D
    num_d_tiles = (D + tile_d - 1) // tile_d
    inv_D = 1.0 / float(D)
    cores = min(_MAX_CORES, S)
    _rms_norm_kernel[cores](x, gamma, out, S, D, num_d_tiles,
                            epsilon, inv_D, tile_d)
    return out
