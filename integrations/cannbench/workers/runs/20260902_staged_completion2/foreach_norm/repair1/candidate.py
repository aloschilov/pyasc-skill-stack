"""CANN Bench ForeachNorm: per-tensor p-norm via pyasc asc2 kernels.

Kernels:
  _k_sum_abs  : p=1, TILE=2048, reduce_sum(abs(xf))
  _k_sum_sq   : p=2, TILE=2048, reduce_sum(xf*xf), final=sqrt
  _k_max_abs  : p=inf, TILE=2048, reduce_max(abs(xf))
  _k_sum_gen  : general p, TILE=1024, reduce_sum(exp(log(abs(xf))*p)), final=exp(log(S)/p)

All accumulation in f32; cross-core via atomic_add / atomic_max into a
zero-seeded global buffer; second single-core pass applies final transform,
casts to input dtype, and writes to an 8-element target-dtype buffer.
"""

import math
import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform


_WIDE_TILE = 2048
_NARROW_TILE = 1024
_MAX_CORES = 72
_INF = math.inf


@asc2.jit
def _k_sum_abs(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
               size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [8])
    acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        a = asc2.abs(xf)
        acc = acc + asc2.reduce_sum(a)
    s = asc2.full([8], acc, dtype=asc.float32)
    asc2.atomic_add(s, out_gm, [0])


@asc2.jit
def _k_sum_sq(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
              size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [8])
    acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        sq = xf * xf
        acc = acc + asc2.reduce_sum(sq)
    s = asc2.full([8], acc, dtype=asc.float32)
    asc2.atomic_add(s, out_gm, [0])


@asc2.jit
def _k_max_abs(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
               size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [8])
    acc = asc2.full([8], 0.0, dtype=asc.float32)
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        a = asc2.abs(xf)
        m = asc2.reduce_max(a)
        m_tile = asc2.full([8], m, dtype=asc.float32)
        acc = asc2.maximum(acc, m_tile)
    asc2.atomic_max(acc, out_gm, [0])


@asc2.jit
def _k_sum_gen(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
               size: int, num_tiles: int, p: float,
               tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [8])
    acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        a = asc2.abs(xf)
        log_a = asc2.log(a)
        log_a_p = log_a * p
        e = asc2.exp(log_a_p)
        acc = acc + asc2.reduce_sum(e)
    s = asc2.full([8], acc, dtype=asc.float32)
    asc2.atomic_add(s, out_gm, [0])


@asc2.jit
def _k_final_identity(buf_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      x_ptr: asc.GlobalAddress):
    buf_gm = asc2.global_tensor(buf_ptr, [8])
    out_gm = asc2.global_tensor(out_ptr, [8])
    x_gm = asc2.global_tensor(x_ptr, [8])
    b = asc2.copy_in(buf_gm, [0], [8], real_shape=[8])
    x_ref = asc2.copy_in(x_gm, [0], [8], real_shape=[8])
    asc2.copy_out(b.to(x_ref.dtype), out_gm, [0], real_shape=[8])


@asc2.jit
def _k_final_sqrt(buf_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                  x_ptr: asc.GlobalAddress):
    buf_gm = asc2.global_tensor(buf_ptr, [8])
    out_gm = asc2.global_tensor(out_ptr, [8])
    x_gm = asc2.global_tensor(x_ptr, [8])
    b = asc2.copy_in(buf_gm, [0], [8], real_shape=[8])
    x_ref = asc2.copy_in(x_gm, [0], [8], real_shape=[8])
    r = asc2.sqrt(b)
    asc2.copy_out(r.to(x_ref.dtype), out_gm, [0], real_shape=[8])


@asc2.jit
def _k_final_power(buf_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                   x_ptr: asc.GlobalAddress, p: float):
    buf_gm = asc2.global_tensor(buf_ptr, [8])
    out_gm = asc2.global_tensor(out_ptr, [8])
    x_gm = asc2.global_tensor(x_ptr, [8])
    b = asc2.copy_in(buf_gm, [0], [8], real_shape=[8])
    x_ref = asc2.copy_in(x_gm, [0], [8], real_shape=[8])
    log_s = asc2.log(b)
    log_s_div_p = log_s / p
    r = asc2.exp(log_s_div_p)
    asc2.copy_out(r.to(x_ref.dtype), out_gm, [0], real_shape=[8])


def _launch_norm(x: torch.Tensor, scalar: float):
    size = x.numel()
    buf = torch.zeros([8], dtype=torch.float32, device=x.device)
    out8 = torch.empty(8, dtype=x.dtype, device=x.device)

    if scalar == _INF:
        tile = _WIDE_TILE
        num_tiles = asc.ceildiv(size, tile)
        cores = min(_MAX_CORES, num_tiles) if num_tiles > 0 else 1
        _k_max_abs[cores](x, buf, size, num_tiles, tile)
        _k_final_identity[1](buf, out8, x)
    elif scalar == 1.0:
        tile = _WIDE_TILE
        num_tiles = asc.ceildiv(size, tile)
        cores = min(_MAX_CORES, num_tiles) if num_tiles > 0 else 1
        _k_sum_abs[cores](x, buf, size, num_tiles, tile)
        _k_final_identity[1](buf, out8, x)
    elif scalar == 2.0:
        tile = _WIDE_TILE
        num_tiles = asc.ceildiv(size, tile)
        cores = min(_MAX_CORES, num_tiles) if num_tiles > 0 else 1
        _k_sum_sq[cores](x, buf, size, num_tiles, tile)
        _k_final_sqrt[1](buf, out8, x)
    else:
        tile = _NARROW_TILE
        num_tiles = asc.ceildiv(size, tile)
        cores = min(_MAX_CORES, num_tiles) if num_tiles > 0 else 1
        _k_sum_gen[cores](x, buf, size, num_tiles, scalar, tile)
        _k_final_power[1](buf, out8, x, scalar)

    return out8.narrow(0, 0, 1).reshape(())


def foreach_norm(x: list, scalar: float) -> list:
    ensure_npu_platform()
    results = []
    for t in x:
        tc = t.contiguous() if not t.is_contiguous() else t
        result = _launch_norm(tc, scalar)
        results.append(result)
    return results
