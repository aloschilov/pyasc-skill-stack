"""CANN Bench Softmax implemented as pyasc asctile kernels."""

import torch
import asc
import asctile

from ._pyasc_runtime import ensure_npu_platform

_NEG_INF = float('-inf')
_MAX_CORES = 72
_TA = 128
_TB = 128


def _select_tile(axis):
    if axis <= 2048:
        return 2048, 2
    elif axis <= 4096:
        return 4096, 1
    else:
        return 8448, 1


@asctile.jit
def _softmax_row_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                        outer: int, axis: int, num_tiles: int,
                        tile_size: asc.ConstExpr[int],
                        unroll: asc.ConstExpr[int]):
    total = outer * axis
    x_gm = asctile.global_tensor(x_ptr, [total])
    out_gm = asctile.global_tensor(out_ptr, [total])
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                           unroll_factor=unroll):
        off = t * axis
        x_tile = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[axis],
                                 pad_value=_NEG_INF)
        xf = x_tile.to(asc.float32)
        m = asctile.reduce_max(xf)
        shifted = xf - m
        e = asctile.exp(shifted)
        s = asctile.reduce_sum(e)
        y = e / s
        asctile.copy_out(y.to(x_tile.dtype), out_gm, [off], real_shape=[axis])


@asctile.jit
def _transpose_kernel(src_ptr: asc.GlobalAddress, dst_ptr: asc.GlobalAddress,
                      outer_dim: int, dim_a: int, dim_b: int,
                      num_a_blocks: int, num_b_blocks: int, num_blocks: int,
                      ta: asc.ConstExpr[int], tb: asc.ConstExpr[int]):
    src_rows = outer_dim * dim_a
    src_cols = dim_b
    dst_rows = outer_dim * dim_b
    dst_cols = dim_a
    src_gm = asctile.global_tensor(src_ptr, [src_rows, src_cols])
    dst_gm = asctile.global_tensor(dst_ptr, [dst_rows, dst_cols])
    bps = num_a_blocks * num_b_blocks
    for t in asctile.range(asctile.block_idx(), num_blocks, asctile.block_num(),
                           unroll_factor=1):
        oi = t // bps
        rem = t - oi * bps
        ab = rem // num_b_blocks
        bb = rem - ab * num_b_blocks
        r_off = oi * dim_a + ab * ta
        c_off = bb * tb
        ra = ta if ab * ta + ta <= dim_a else dim_a - ab * ta
        rb = tb if bb * tb + tb <= dim_b else dim_b - bb * tb
        tile = asctile.copy_in(src_gm, [r_off, c_off], [ta, tb],
                               real_shape=[ra, rb])
        tile_t = asctile.transpose(tile)
        doff_r = oi * dim_b + bb * tb
        doff_c = ab * ta
        asctile.copy_out(tile_t, dst_gm, [doff_r, doff_c],
                         real_shape=[rb, ra])


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    if x.numel() == 0:
        return torch.empty_like(x)
    rank = len(x.shape)
    if dim < 0:
        dim += rank
    outer = 1
    for i in range(dim):
        outer *= x.shape[i]
    axis = x.shape[dim]
    inner = 1
    for i in range(dim + 1, rank):
        inner *= x.shape[i]

    if inner == 1:
        out = torch.empty_like(x)
        tile_size, unroll = _select_tile(axis)
        cores = min(_MAX_CORES, outer)
        _softmax_row_kernel[cores](x, out, outer, axis, outer, tile_size, unroll)
        return out

    n = outer * axis * inner
    tmp1 = torch.empty(n, dtype=x.dtype, device=x.device)
    num_a_fwd = (axis + _TA - 1) // _TA
    num_b_fwd = (inner + _TB - 1) // _TB
    num_blocks_fwd = outer * num_a_fwd * num_b_fwd
    cores_t = min(_MAX_CORES, num_blocks_fwd)
    _transpose_kernel[cores_t](x, tmp1, outer, axis, inner,
                               num_a_fwd, num_b_fwd, num_blocks_fwd,
                               _TA, _TB)

    soft_out = torch.empty(n, dtype=x.dtype, device=x.device)
    outer_soft = outer * inner
    tile_size, unroll = _select_tile(axis)
    cores_s = min(_MAX_CORES, outer_soft)
    _softmax_row_kernel[cores_s](tmp1, soft_out, outer_soft, axis,
                                 outer_soft, tile_size, unroll)

    out = torch.empty_like(x)
    num_a_bwd = (inner + _TA - 1) // _TA
    num_b_bwd = (axis + _TB - 1) // _TB
    num_blocks_bwd = outer * num_a_bwd * num_b_bwd
    cores_b = min(_MAX_CORES, num_blocks_bwd)
    _transpose_kernel[cores_b](soft_out, out, outer, inner, axis,
                               num_a_bwd, num_b_bwd, num_blocks_bwd,
                               _TA, _TB)
    return out
