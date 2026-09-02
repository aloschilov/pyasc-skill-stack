"""CANN Bench ForeachNorm interface implemented as a pyasc asc2 kernel.

Two-stage parallel reduction per tensor:
  Stage 1: grid-stride tile loop; each core accumulates a partial (sum or
    max) and writes it to a per-core slot in a [72*8] f32 scratch buffer.
  Stage 2: single core reduces the partials and applies the final power,
    writing an [8]-padded scalar result.
  L1 (p=1) uses direct sum(abs(x)); L2 (p=2) uses direct sum(x*x);
  Linf (p=inf) uses max(abs(x)); other p uses sum(exp(log(abs(x))*p))
  with result = exp(log(S)/p).  f16/bf16 inputs are promoted to f32 for
  the internal compute and cast back on output.
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE = 4096
_POW_TILE = 2048
_MAX_CORES = 72
_SCRATCH = 576


@asc2.jit
def _partial_sum_abs(
    x_ptr: asc.GlobalAddress,
    sc_ptr: asc.GlobalAddress,
    size: int,
    num_tiles: int,
    tile_size: asc.ConstExpr[int],
):
    x_gm = asc2.global_tensor(x_ptr, [size])
    sc_gm = asc2.global_tensor(sc_ptr, [576])
    acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
    for t in asc2.range(
        asc2.block_idx(),
        num_tiles,
        asc2.block_num(),
        unroll_factor=2,
        gm_barrier=True,
    ):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        abs_x = asc2.abs(xf)
        s = asc2.reduce_sum(abs_x)
        acc = acc + s
    sc_off = asc2.block_idx() * 8
    asc2.copy_out(asc2.full([8], acc, dtype=asc.float32), sc_gm, [sc_off])


@asc2.jit
def _partial_sum_sq(
    x_ptr: asc.GlobalAddress,
    sc_ptr: asc.GlobalAddress,
    size: int,
    num_tiles: int,
    tile_size: asc.ConstExpr[int],
):
    x_gm = asc2.global_tensor(x_ptr, [size])
    sc_gm = asc2.global_tensor(sc_ptr, [576])
    acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
    for t in asc2.range(
        asc2.block_idx(),
        num_tiles,
        asc2.block_num(),
        unroll_factor=2,
        gm_barrier=True,
    ):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        sq = xf * xf
        s = asc2.reduce_sum(sq)
        acc = acc + s
    sc_off = asc2.block_idx() * 8
    asc2.copy_out(asc2.full([8], acc, dtype=asc.float32), sc_gm, [sc_off])


@asc2.jit
def _partial_max_abs(
    x_ptr: asc.GlobalAddress,
    sc_ptr: asc.GlobalAddress,
    size: int,
    num_tiles: int,
    tile_size: asc.ConstExpr[int],
):
    x_gm = asc2.global_tensor(x_ptr, [size])
    sc_gm = asc2.global_tensor(sc_ptr, [576])
    acc_tile = asc2.full([1, 64], 0.0, dtype=asc.float32)
    for t in asc2.range(
        asc2.block_idx(),
        num_tiles,
        asc2.block_num(),
        unroll_factor=2,
        gm_barrier=True,
    ):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        abs_x = asc2.abs(xf)
        m = asc2.reduce_max(abs_x)
        m_tile = asc2.full([1, 64], m, dtype=asc.float32)
        acc_tile = asc2.maximum(acc_tile, m_tile)
    acc = asc2.reduce_max(acc_tile)
    sc_off = asc2.block_idx() * 8
    asc2.copy_out(asc2.full([8], acc, dtype=asc.float32), sc_gm, [sc_off])


@asc2.jit
def _partial_sum_pow(
    x_ptr: asc.GlobalAddress,
    sc_ptr: asc.GlobalAddress,
    size: int,
    num_tiles: int,
    p: float,
    tile_size: asc.ConstExpr[int],
):
    x_gm = asc2.global_tensor(x_ptr, [size])
    sc_gm = asc2.global_tensor(sc_ptr, [576])
    acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
    for t in asc2.range(
        asc2.block_idx(),
        num_tiles,
        asc2.block_num(),
        unroll_factor=2,
        gm_barrier=True,
    ):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        abs_x = asc2.abs(xf)
        log_abs = asc2.log(abs_x)
        scaled = log_abs * p
        pw = asc2.exp(scaled)
        s = asc2.reduce_sum(pw)
        acc = acc + s
    sc_off = asc2.block_idx() * 8
    asc2.copy_out(asc2.full([8], acc, dtype=asc.float32), sc_gm, [sc_off])


@asc2.jit
def _final_sum(
    x_ptr: asc.GlobalAddress,
    sc_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    x_size: int,
    num_slots: int,
    p: float,
    sc_size: asc.ConstExpr[int],
):
    sc_gm = asc2.global_tensor(sc_ptr, [sc_size])
    x_gm = asc2.global_tensor(x_ptr, [x_size])
    out_gm = asc2.global_tensor(out_ptr, [8])
    x_ref = asc2.copy_in(x_gm, [0], [8], real_shape=[1])
    partials = asc2.copy_in(sc_gm, [0], [sc_size], real_shape=[num_slots])
    total = asc2.reduce_sum(partials)
    total = total / 8.0
    total_tile = asc2.full([8], total, dtype=asc.float32)
    log_tile = asc2.log(total_tile)
    scaled_tile = log_tile / p
    result_tile = asc2.exp(scaled_tile)
    asc2.copy_out(result_tile.to(x_ref.dtype), out_gm, [0])


@asc2.jit
def _final_max(
    x_ptr: asc.GlobalAddress,
    sc_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    x_size: int,
    num_slots: int,
    sc_size: asc.ConstExpr[int],
):
    sc_gm = asc2.global_tensor(sc_ptr, [sc_size])
    x_gm = asc2.global_tensor(x_ptr, [x_size])
    out_gm = asc2.global_tensor(out_ptr, [8])
    x_ref = asc2.copy_in(x_gm, [0], [8], real_shape=[1])
    partials = asc2.copy_in(sc_gm, [0], [sc_size], real_shape=[num_slots])
    result = asc2.reduce_max(partials)
    asc2.copy_out(
        asc2.full([8], result, dtype=asc.float32).to(x_ref.dtype),
        out_gm,
        [0],
    )


def foreach_norm(x, scalar):
    """Compute p-norm of each tensor in the list via asc2 kernels."""
    ensure_npu_platform()
    if not x:
        return []
    input_dtype = x[0].dtype
    results = []
    is_inf = scalar == float("inf")
    for t in x:
        if not t.is_contiguous():
            t = t.contiguous()
        size = t.numel()
        scratch = torch.empty(576, dtype=torch.float32, device=t.device)
        out8 = torch.empty(8, dtype=input_dtype, device=t.device)
        if is_inf:
            num_tiles = asc.ceildiv(size, _TILE)
            cores = min(_MAX_CORES, num_tiles)
            _partial_max_abs[cores](t, scratch, size, num_tiles, _TILE)
            _final_max[1](t, scratch, out8, size, cores * 8, _SCRATCH)
        elif scalar == 1.0:
            num_tiles = asc.ceildiv(size, _TILE)
            cores = min(_MAX_CORES, num_tiles)
            _partial_sum_abs[cores](t, scratch, size, num_tiles, _TILE)
            _final_sum[1](
                t, scratch, out8, size, cores * 8, scalar, _SCRATCH
            )
        elif scalar == 2.0:
            num_tiles = asc.ceildiv(size, _TILE)
            cores = min(_MAX_CORES, num_tiles)
            _partial_sum_sq[cores](t, scratch, size, num_tiles, _TILE)
            _final_sum[1](
                t, scratch, out8, size, cores * 8, scalar, _SCRATCH
            )
        else:
            num_tiles = asc.ceildiv(size, _POW_TILE)
            cores = min(_MAX_CORES, num_tiles)
            _partial_sum_pow[cores](
                t, scratch, size, num_tiles, scalar, _POW_TILE
            )
            _final_sum[1](
                t, scratch, out8, size, cores * 8, scalar, _SCRATCH
            )
        results.append(out8[0])
    return results
