import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE_D = 2048
_MAX_CORES = 72


@asc2.jit
def _rms_norm_kernel(
    x_ptr: asc.GlobalAddress,
    gamma_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    S: int,
    D: int,
    num_d_tiles: int,
    epsilon: float,
    tile_d: asc.ConstExpr[int],
):
    x_gm = asc2.global_tensor(x_ptr, [S * D])
    gamma_gm = asc2.global_tensor(gamma_ptr, [D])
    out_gm = asc2.global_tensor(out_ptr, [S * D])

    for r in asc2.range(asc2.block_idx(), S, asc2.block_num(), unroll_factor=2):
        row_off = r * D

        ss = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
        for dt in asc2.range(num_d_tiles):
            off_d = dt * tile_d
            n_d = tile_d if off_d + tile_d <= D else D - off_d
            x_tile = asc2.copy_in(x_gm, [row_off + off_d], [tile_d], real_shape=[n_d])
            xf = x_tile.to(asc.float32)
            sq = xf * xf
            ss = ss + asc2.reduce_sum(sq)

        ss_tile = asc2.full([tile_d], ss, dtype=asc.float32)
        inv_rms_tile = asc2.rsqrt(ss_tile / D + epsilon)

        for dt in asc2.range(num_d_tiles):
            off_d = dt * tile_d
            n_d = tile_d if off_d + tile_d <= D else D - off_d
            x_tile = asc2.copy_in(x_gm, [row_off + off_d], [tile_d], real_shape=[n_d])
            g_tile = asc2.copy_in(gamma_gm, [off_d], [tile_d], real_shape=[n_d])
            xf = x_tile.to(asc.float32)
            gf = g_tile.to(asc.float32)
            yf = xf * gf * inv_rms_tile
            asc2.copy_out(yf.to(x_tile.dtype), out_gm, [row_off + off_d], real_shape=[n_d])


def rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()
    out = torch.empty_like(x)
    D = x.shape[-1]
    S = x.numel() // D
    if S == 0 or D == 0:
        return out
    num_d_tiles = (D + _TILE_D - 1) // _TILE_D
    cores = min(_MAX_CORES, S)
    _rms_norm_kernel[cores](x, gamma, out, S, D, num_d_tiles, float(epsilon), _TILE_D)
    return out
