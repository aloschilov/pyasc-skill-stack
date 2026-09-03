"""CANN Bench RmsNorm interface implemented as a pyasc asctile kernel.

Kernel design (two-pass streaming per row):
  - Outer grid-stride row loop; each core strides over rows.
  - Pass 1: stream D in TILE_D chunks, accumulate sum(x^2) in an f32
    loop-carried scalar.
  - Per-row scalar compute: inv_rms = rsqrt(mean_sq + epsilon), broadcast
    to a full tile so the apply step is pure tile-tile arithmetic.
  - Pass 2: re-stream D chunks, reload x and gamma from GM, emit
    y = x * inv_rms * gamma in f32, cast back to input dtype, store.
  - f16/bf16 inputs are promoted to f32 inside the kernel (the precision
    standard expects f32 internal compute; squares of |x| up to 65504
    overflow f16 otherwise).
  - TILE_D = 1024 with no unroll keeps UB single-buffered (~59-79 KB peak,
    well under the 253952 B budget).
"""

import torch

import asc
import asctile

from ._pyasc_runtime import ensure_npu_platform

_TILE_D = 1024
_MAX_CORES = 72


@asctile.jit
def _rms_norm_kernel(x_ptr: asc.GlobalAddress, gamma_ptr: asc.GlobalAddress,
                     out_ptr: asc.GlobalAddress,
                     S: int, D: int, num_d_tiles: int,
                     epsilon: float, inv_D: float,
                     tile_d: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [S * D])
    g_gm = asctile.global_tensor(gamma_ptr, [D])
    o_gm = asctile.global_tensor(out_ptr, [S * D])

    for r in asctile.range(asctile.block_idx(), S, asctile.block_num()):
        row = r * D
        acc = asctile.reduce_sum(asctile.full([1, 64], 0.0, dtype=asc.float32))
        for dt in asctile.range(num_d_tiles):
            od = dt * tile_d
            n = tile_d if od + tile_d <= D else D - od
            x = asctile.copy_in(x_gm, [row + od], [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            acc = acc + asctile.reduce_sum(xf * xf)

        inv_rms_tile = asctile.rsqrt(
            asctile.full([tile_d], acc * inv_D + epsilon, dtype=asc.float32))

        for dt in asctile.range(num_d_tiles):
            od = dt * tile_d
            n = tile_d if od + tile_d <= D else D - od
            x = asctile.copy_in(x_gm, [row + od], [tile_d], real_shape=[n])
            g = asctile.copy_in(g_gm, [od], [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            gf = g.to(asc.float32)
            y = xf * inv_rms_tile * gf
            asctile.copy_out(y.to(x.dtype), o_gm, [row + od], real_shape=[n])


def rms_norm(x: torch.Tensor, gamma: torch.Tensor,
             epsilon: float = 1e-6) -> torch.Tensor:
    """RMS (root-mean-square) normalization over the last dimension via a pyasc asctile kernel."""
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
    num_d_tiles = asc.ceildiv(D, tile_d)
    inv_D = 1.0 / float(D)
    cores = min(_MAX_CORES, S)
    _rms_norm_kernel[cores](x, gamma, out, S, D, num_d_tiles,
                            epsilon, inv_D, tile_d)
    return out
