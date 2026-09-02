"""CANN Bench ForeachNorm interface implemented as a pyasc asc2 kernel.

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
    asc2.reduce_sum(asc2.full([1,64],0.0,...)) pattern.
  - Final scalar widened to [8] (32-byte min store) and returned as
    out8[0] (0-dim view, matching torch.norm output shape).
  - Accumulator buffer zeroed by a tiny asc2 kernel (no torch.zeros).
"""

import math

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE = 2048
_MAX_CORES = 72


# ── Utility kernel ──────────────────────────────────────────────────

@asc2.jit
def _zero_f32_kernel(acc_ptr: asc.GlobalAddress):
    """Zero an 8-element f32 buffer (32-byte min store)."""
    acc_gm = asc2.global_tensor(acc_ptr, [8])
    z = asc2.full([8], 0.0, dtype=asc.float32)
    asc2.copy_out(z, acc_gm, [0], real_shape=[8])


# ── Reduction kernels (multi-core, grid-stride) ────────────────────

@asc2.jit
def _reduce_abs_sum_kernel(x_ptr: asc.GlobalAddress, acc_ptr: asc.GlobalAddress,
                           size: int, num_tiles: int,
                           tile_size: asc.ConstExpr[int]):
    """Partial sum of abs(x); atomic_add combines across cores."""
    x_gm = asc2.global_tensor(x_ptr, [size])
    acc_gm = asc2.global_tensor(acc_ptr, [8])
    acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        acc = acc + asc2.reduce_sum(asc2.abs(xf))
    asc2.atomic_add(asc2.full([8], acc, dtype=asc.float32), acc_gm, [0])


@asc2.jit
def _reduce_sq_sum_kernel(x_ptr: asc.GlobalAddress, acc_ptr: asc.GlobalAddress,
                          size: int, num_tiles: int,
                          tile_size: asc.ConstExpr[int]):
    """Partial sum of x*x; atomic_add combines across cores."""
    x_gm = asc2.global_tensor(x_ptr, [size])
    acc_gm = asc2.global_tensor(acc_ptr, [8])
    acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        acc = acc + asc2.reduce_sum(xf * xf)
    asc2.atomic_add(asc2.full([8], acc, dtype=asc.float32), acc_gm, [0])


@asc2.jit
def _reduce_max_abs_kernel(x_ptr: asc.GlobalAddress, acc_ptr: asc.GlobalAddress,
                           size: int, num_tiles: int,
                           tile_size: asc.ConstExpr[int]):
    """Per-tile atomic_max of abs(x) into the accumulator."""
    x_gm = asc2.global_tensor(x_ptr, [size])
    acc_gm = asc2.global_tensor(acc_ptr, [8])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        ax = asc2.abs(xf)
        tile_max = asc2.reduce_max(ax)
        asc2.atomic_max(asc2.full([8], tile_max, dtype=asc.float32),
                        acc_gm, [0])


@asc2.jit
def _reduce_pow_sum_kernel(x_ptr: asc.GlobalAddress, acc_ptr: asc.GlobalAddress,
                           size: int, num_tiles: int,
                           tile_size: asc.ConstExpr[int], p: float):
    """Partial sum of |x|^p = exp(log(abs(x))*p); atomic_add combines."""
    x_gm = asc2.global_tensor(x_ptr, [size])
    acc_gm = asc2.global_tensor(acc_ptr, [8])
    acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        ax = asc2.abs(xf)
        lx = asc2.log(ax)
        xp = asc2.exp(lx * p)
        acc = acc + asc2.reduce_sum(xp)
    asc2.atomic_add(asc2.full([8], acc, dtype=asc.float32), acc_gm, [0])


# ── Finalize kernels (single-core) ──────────────────────────────────

@asc2.jit
def _finalize_copy_kernel(acc_ptr: asc.GlobalAddress, x_ptr: asc.GlobalAddress,
                          out_ptr: asc.GlobalAddress):
    """No transform -- read S, cast to input dtype, write."""
    acc_gm = asc2.global_tensor(acc_ptr, [8])
    x_gm = asc2.global_tensor(x_ptr, [8])
    out_gm = asc2.global_tensor(out_ptr, [8])
    x_tmpl = asc2.copy_in(x_gm, [0], [8], real_shape=[1])
    s_tile = asc2.copy_in(acc_gm, [0], [8])
    s = asc2.reduce_max(s_tile)
    y = asc2.full([8], s, dtype=asc.float32)
    asc2.copy_out(y.to(x_tmpl.dtype), out_gm, [0], real_shape=[8])


@asc2.jit
def _finalize_sqrt_kernel(acc_ptr: asc.GlobalAddress, x_ptr: asc.GlobalAddress,
                          out_ptr: asc.GlobalAddress):
    """sqrt(S) -- for p==2."""
    acc_gm = asc2.global_tensor(acc_ptr, [8])
    x_gm = asc2.global_tensor(x_ptr, [8])
    out_gm = asc2.global_tensor(out_ptr, [8])
    x_tmpl = asc2.copy_in(x_gm, [0], [8], real_shape=[1])
    s_tile = asc2.copy_in(acc_gm, [0], [8])
    s = asc2.reduce_max(s_tile)
    y = asc2.sqrt(asc2.full([8], s, dtype=asc.float32))
    asc2.copy_out(y.to(x_tmpl.dtype), out_gm, [0], real_shape=[8])


@asc2.jit
def _finalize_pow_kernel(acc_ptr: asc.GlobalAddress, x_ptr: asc.GlobalAddress,
                         out_ptr: asc.GlobalAddress, p: float):
    """exp(log(S)/p) -- for general p."""
    acc_gm = asc2.global_tensor(acc_ptr, [8])
    x_gm = asc2.global_tensor(x_ptr, [8])
    out_gm = asc2.global_tensor(out_ptr, [8])
    x_tmpl = asc2.copy_in(x_gm, [0], [8], real_shape=[1])
    s_tile = asc2.copy_in(acc_gm, [0], [8])
    s = asc2.reduce_max(s_tile)
    s_wide = asc2.full([8], s, dtype=asc.float32)
    y = asc2.exp(asc2.log(s_wide) / p)
    asc2.copy_out(y.to(x_tmpl.dtype), out_gm, [0], real_shape=[8])


# ── Public callable ─────────────────────────────────────────────────

def foreach_norm(x, scalar):
    """Per-tensor p-norm of each tensor in the list via asc2 kernels."""
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
