"""CANN Bench RMSNorm implemented with a row-wise pyasc asc2 kernel."""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_MAX_CORES = 72


@asc2.jit(reuse_alloc=1)
def _rms_norm_kernel(x_ptr: asc.GlobalAddress,
                     gamma_ptr: asc.GlobalAddress,
                     out_ptr: asc.GlobalAddress,
                     rows: int,
                     cols: int,
                     num_tiles: int,
                     tile_size: asc.ConstExpr[int],
                     epsilon: float):
    x_gm = asc2.global_tensor(x_ptr, [rows, cols])
    gamma_gm = asc2.global_tensor(gamma_ptr, [cols])
    out_gm = asc2.global_tensor(out_ptr, [rows, cols])

    for row in asc2.range(asc2.block_idx(), rows, asc2.block_num()):
        acc = asc2.reduce_sum(
            asc2.full([1, 64], 0.0, dtype=asc.float32)
        )
        for tile_id in asc2.range(num_tiles):
            col = tile_id * tile_size
            n = tile_size if col + tile_size <= cols else cols - col
            x = asc2.copy_in(
                x_gm, [row, col], [1, tile_size], real_shape=[1, n]
            )
            xf = x.to(asc.float32)
            acc = acc + asc2.reduce_sum(xf * xf)

        inv_rms = 1.0 / asc2.sqrt(acc / cols + epsilon)

        for tile_id in asc2.range(num_tiles):
            col = tile_id * tile_size
            n = tile_size if col + tile_size <= cols else cols - col
            x = asc2.copy_in(
                x_gm, [row, col], [1, tile_size], real_shape=[1, n]
            )
            gamma = asc2.copy_in(
                gamma_gm, [col], [tile_size], real_shape=[n]
            )
            y = x.to(asc.float32) * gamma.to(asc.float32) * inv_rms
            asc2.copy_out(
                y.to(x.dtype), out_gm, [row, col], real_shape=[1, n]
            )


def _tile_size(cols: int) -> int:
    if cols <= 16:
        return 16
    if cols <= 128:
        return 128
    if cols <= 512:
        return 512
    if cols <= 1024:
        return 1024
    if cols <= 2048:
        return 2048
    if cols <= 4096:
        return 4096
    return 8192


def rms_norm(x: torch.Tensor, gamma: torch.Tensor,
             epsilon: float = 1e-6) -> torch.Tensor:
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()

    out = torch.empty_like(x)
    if x.numel() == 0:
        return out

    cols = x.shape[-1]
    rows = x.numel() // cols
    tile = _tile_size(cols)
    num_tiles = asc.ceildiv(cols, tile)
    cores = min(_MAX_CORES, rows)
    _rms_norm_kernel[cores](
        x, gamma, out, rows, cols, num_tiles, tile, float(epsilon)
    )
    return out
