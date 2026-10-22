"""CANN Bench ForeachNorm interface implemented as a pyasc asctile kernel.

Kernel design (reduction variant):
  - Per-tensor two-pass reduction: multi-core partial reduce with atomic
    combine, then a single-core finalize that applies the norm's outer
    power and casts to the input dtype.
  - p==1  : sum(abs(x)), no final transform.
  - p==2  : sum(x*x), final sqrt.
  - p==inf: max(abs(x)) via per-tile atomic_max, no final transform.
  - general p: sum(exp(log(abs(x))*p)), final exp(log(S)/p).
  - f16/bf16 inputs promoted to f32 inside the kernel for precision.
  - Loop-carried accumulators seeded with the verified
    asctile.reduce_sum(asctile.full([1,64],0.0,...)) pattern.
  - Final scalar widened to [8] (32-byte min store) and returned as
    out8[0] (0-dim view, matching torch.norm output shape).
  - Accumulator buffer zeroed by a tiny asctile kernel (no torch.zeros).
"""

import math

import torch

import asc
import asctile

from ._pyasc_runtime import ensure_npu_platform

_TILE = 2048
_MAX_CORES = 72


# ── Utility kernel ──────────────────────────────────────────────────

@asctile.jit
def _zero_f32_kernel(acc_ptr: asc.GlobalAddress):
    """Zero an 8-element f32 buffer (32-byte min store)."""
    acc_gm = asctile.global_tensor(acc_ptr, [8])
    z = asctile.full([8], 0.0, dtype=asc.float32)
    asctile.copy_out(z, acc_gm, [0], real_shape=[8])


# ── Reduction kernels (multi-core, grid-stride) ────────────────────

@asctile.jit
def _reduce_abs_sum_kernel(x_ptr: asc.GlobalAddress, acc_ptr: asc.GlobalAddress,
                           size: int, num_tiles: int,
                           tile_size: asc.ConstExpr[int]):
    """Partial sum of abs(x); atomic_add combines across cores."""
    x_gm = asctile.global_tensor(x_ptr, [size])
    acc_gm = asctile.global_tensor(acc_ptr, [8])
    acc = asctile.reduce_sum(asctile.full([1, 64], 0.0, dtype=asc.float32))
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        acc = acc + asctile.reduce_sum(asctile.abs(xf))
    asctile.atomic_add(asctile.full([8], acc, dtype=asc.float32), acc_gm, [0])


@asctile.jit
def _reduce_sq_sum_kernel(x_ptr: asc.GlobalAddress, acc_ptr: asc.GlobalAddress,
                          size: int, num_tiles: int,
                          tile_size: asc.ConstExpr[int]):
    """Partial sum of x*x; atomic_add combines across cores."""
    x_gm = asctile.global_tensor(x_ptr, [size])
    acc_gm = asctile.global_tensor(acc_ptr, [8])
    acc = asctile.reduce_sum(asctile.full([1, 64], 0.0, dtype=asc.float32))
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        acc = acc + asctile.reduce_sum(xf * xf)
    asctile.atomic_add(asctile.full([8], acc, dtype=asc.float32), acc_gm, [0])


@asctile.jit
def _reduce_max_abs_kernel(x_ptr: asc.GlobalAddress, acc_ptr: asc.GlobalAddress,
                           size: int, num_tiles: int,
                           tile_size: asc.ConstExpr[int]):
    """Per-tile atomic_max of abs(x) into the accumulator."""
    x_gm = asctile.global_tensor(x_ptr, [size])
    acc_gm = asctile.global_tensor(acc_ptr, [8])
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        ax = asctile.abs(xf)
        tile_max = asctile.reduce_max(ax)
        asctile.atomic_max(asctile.full([8], tile_max, dtype=asc.float32),
                        acc_gm, [0])


@asctile.jit
def _reduce_pow_sum_kernel(x_ptr: asc.GlobalAddress, acc_ptr: asc.GlobalAddress,
                           size: int, num_tiles: int,
                           tile_size: asc.ConstExpr[int], p: float):
    """Partial sum of |x|^p = exp(log(abs(x))*p); atomic_add combines."""
    x_gm = asctile.global_tensor(x_ptr, [size])
    acc_gm = asctile.global_tensor(acc_ptr, [8])
    acc = asctile.reduce_sum(asctile.full([1, 64], 0.0, dtype=asc.float32))
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        ax = asctile.abs(xf)
        lx = asctile.log(ax)
        xp = asctile.exp(lx * p)
        acc = acc + asctile.reduce_sum(xp)
    asctile.atomic_add(asctile.full([8], acc, dtype=asc.float32), acc_gm, [0])


# ── Finalize kernels (single-core) ──────────────────────────────────

@asctile.jit
def _finalize_copy_kernel(acc_ptr: asc.GlobalAddress, x_ptr: asc.GlobalAddress,
                          out_ptr: asc.GlobalAddress):
    """No transform -- read S, cast to input dtype, write."""
    acc_gm = asctile.global_tensor(acc_ptr, [8])
    x_gm = asctile.global_tensor(x_ptr, [8])
    out_gm = asctile.global_tensor(out_ptr, [8])
    x_tmpl = asctile.copy_in(x_gm, [0], [8], real_shape=[1])
    s_tile = asctile.copy_in(acc_gm, [0], [8])
    s = asctile.reduce_max(s_tile)
    y = asctile.full([8], s, dtype=asc.float32)
    asctile.copy_out(y.to(x_tmpl.dtype), out_gm, [0], real_shape=[8])


@asctile.jit
def _finalize_sqrt_kernel(acc_ptr: asc.GlobalAddress, x_ptr: asc.GlobalAddress,
                          out_ptr: asc.GlobalAddress):
    """sqrt(S) -- for p==2."""
    acc_gm = asctile.global_tensor(acc_ptr, [8])
    x_gm = asctile.global_tensor(x_ptr, [8])
    out_gm = asctile.global_tensor(out_ptr, [8])
    x_tmpl = asctile.copy_in(x_gm, [0], [8], real_shape=[1])
    s_tile = asctile.copy_in(acc_gm, [0], [8])
    s = asctile.reduce_max(s_tile)
    y = asctile.sqrt(asctile.full([8], s, dtype=asc.float32))
    asctile.copy_out(y.to(x_tmpl.dtype), out_gm, [0], real_shape=[8])


@asctile.jit
def _finalize_pow_kernel(acc_ptr: asc.GlobalAddress, x_ptr: asc.GlobalAddress,
                         out_ptr: asc.GlobalAddress, p: float):
    """exp(log(S)/p) -- for general p."""
    acc_gm = asctile.global_tensor(acc_ptr, [8])
    x_gm = asctile.global_tensor(x_ptr, [8])
    out_gm = asctile.global_tensor(out_ptr, [8])
    x_tmpl = asctile.copy_in(x_gm, [0], [8], real_shape=[1])
    s_tile = asctile.copy_in(acc_gm, [0], [8])
    s = asctile.reduce_max(s_tile)
    s_wide = asctile.full([8], s, dtype=asc.float32)
    y = asctile.exp(asctile.log(s_wide) / p)
    asctile.copy_out(y.to(x_tmpl.dtype), out_gm, [0], real_shape=[8])


# ── Public callable ─────────────────────────────────────────────────

def foreach_norm(x, scalar):
    """Per-tensor p-norm of each tensor in the list via asctile kernels."""
    ensure_npu_platform()
    if not x:
        return []

    input_dtype = x[0].dtype
    p = float(scalar)
    results = []

    for tensor in x:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        size = tensor.numel()

        acc8 = torch.empty(8, dtype=torch.float32, device=tensor.device)
        _zero_f32_kernel[1](acc8)
        out8 = torch.empty(8, dtype=input_dtype, device=tensor.device)

        if size == 0:
            tmpl = torch.empty(1, dtype=input_dtype, device=tensor.device)
            _finalize_copy_kernel[1](acc8, tmpl, out8)
            results.append(out8[0])
            continue

        num_tiles = asc.ceildiv(size, _TILE)
        cores = min(_MAX_CORES, num_tiles)

        if p == 1.0:
            _reduce_abs_sum_kernel[cores](tensor, acc8, size, num_tiles, _TILE)
            _finalize_copy_kernel[1](acc8, tensor, out8)
        elif p == 2.0:
            _reduce_sq_sum_kernel[cores](tensor, acc8, size, num_tiles, _TILE)
            _finalize_sqrt_kernel[1](acc8, tensor, out8)
        elif math.isinf(p) and p > 0:
            _reduce_max_abs_kernel[cores](tensor, acc8, size, num_tiles, _TILE)
            _finalize_copy_kernel[1](acc8, tensor, out8)
        else:
            _reduce_pow_sum_kernel[cores](tensor, acc8, size, num_tiles,
                                          _TILE, p)
            _finalize_pow_kernel[1](acc8, tensor, out8, p)

        results.append(out8[0])

    return results
