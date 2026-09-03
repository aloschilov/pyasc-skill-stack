"""Compile-only probe for native-dtype low-level GeLU primitives.

This module is deliberately not the queued CANNBench candidate.  It answers a
smaller pyasc-v2 question before another submission credit is spent: can the
C310 low-level pipeline lower ``adv.erfc`` and ``adv.tanh`` directly for each
of FP16, BF16, and FP32 while retaining the repaired launch ABI?

Numerical correctness and performance are intentionally not claimed by this
probe.  The public wrapper exists only so the shared 20-case compile gate can
exercise all six dtype/mode specializations.
"""

import math

import torch

import asc

from ._pyasc_runtime import c310_asc_jit, ensure_npu_platform


_MAX_CORES = 72
_TILE = 13824
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_C = 0.044715
_CLAMP = 10.0


@asc.jit
def _exact_tile(x_gm: asc.GlobalTensor, out_gm: asc.GlobalTensor,
                offset: int, tile_size: asc.ConstExpr[int]):
    x = asc.LocalTensorAuto(x_gm.dtype, tile_size)
    y = asc.LocalTensorAuto(out_gm.dtype, tile_size)
    asc.data_copy(x, x_gm[offset:], count=tile_size)
    asc.muls(y, x, -_INV_SQRT2, count=tile_size)
    asc.adv.erfc(y, y, count=tile_size, is_reuse_source=True)
    asc.muls(y, y, 0.5, count=tile_size)
    asc.mul(y, y, x, count=tile_size)
    asc.data_copy(out_gm[offset:], y, count=tile_size)


@c310_asc_jit
def _exact_native(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                  size: int, num_tiles: int,
                  tile_size: asc.ConstExpr[int]):
    x_gm = asc.GlobalTensor()
    out_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x_ptr)
    out_gm.set_global_buffer(out_ptr)
    for tile_id in range(asc.get_block_idx(), num_tiles,
                         asc.get_block_num()):
        offset = tile_id * tile_size
        if offset + tile_size > size:
            offset = size - tile_size
        _exact_tile(x_gm, out_gm, offset, tile_size)


@asc.jit
def _tanh_tile(x_gm: asc.GlobalTensor, out_gm: asc.GlobalTensor,
               offset: int, tile_size: asc.ConstExpr[int]):
    x = asc.LocalTensorAuto(x_gm.dtype, tile_size)
    z = asc.LocalTensorAuto(x_gm.dtype, tile_size)
    u = asc.LocalTensorAuto(x_gm.dtype, tile_size)
    y = asc.LocalTensorAuto(out_gm.dtype, tile_size)
    asc.data_copy(x, x_gm[offset:], count=tile_size)

    # Clamping only the tanh argument input avoids FP16 x**2/x**3 overflow.
    # The final multiply still uses the original x, preserving the saturated
    # limits GeLU(x)->x for positive x and GeLU(x)->0 for negative x.
    asc.mins(z, x, _CLAMP, count=tile_size)
    asc.maxs(z, z, -_CLAMP, count=tile_size)
    asc.mul(u, z, z, count=tile_size)
    asc.muls(u, u, _GELU_C, count=tile_size)
    asc.adds(u, u, 1.0, count=tile_size)
    asc.mul(u, u, z, count=tile_size)
    asc.muls(u, u, _SQRT_2_OVER_PI, count=tile_size)
    asc.adv.tanh(u, u, count=tile_size, is_reuse_source=True)
    asc.adds(u, u, 1.0, count=tile_size)
    asc.muls(u, u, 0.5, count=tile_size)
    asc.mul(y, x, u, count=tile_size)
    asc.data_copy(out_gm[offset:], y, count=tile_size)


@c310_asc_jit
def _tanh_native(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                 size: int, num_tiles: int,
                 tile_size: asc.ConstExpr[int]):
    x_gm = asc.GlobalTensor()
    out_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x_ptr)
    out_gm.set_global_buffer(out_ptr)
    for tile_id in range(asc.get_block_idx(), num_tiles,
                         asc.get_block_num()):
        offset = tile_id * tile_size
        if offset + tile_size > size:
            offset = size - tile_size
        _tanh_tile(x_gm, out_gm, offset, tile_size)


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    ensure_npu_platform()
    if approximate not in ("none", "tanh"):
        raise ValueError("approximate must be 'none' or 'tanh'")
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    num_tiles = asc.ceildiv(size, _TILE)
    cores = min(_MAX_CORES, num_tiles)
    kernel = _exact_native if approximate == "none" else _tanh_native
    kernel[cores](x, out, size, num_tiles, _TILE)
    return out
