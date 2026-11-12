# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""SwiGLU activation for Ascend950PR using the public AscTile JIT.

output = silu(x0) * x1 where the input is split in half along ``dim``.

The implementation dispatches between three physical copy layouts based on the
logical layout ``[R, 2*H]`` of the contiguous input:

* BF16 short halves (``H * itemsize <= 32`` bytes) use a rank-one slab copy of
  complete rows followed by two cached ``asctile.reindex`` gathers.
* Moderate halves use strided rank-two row tiles sized from the DMA / UB budget.
* Wide halves / long rows / few rows use flat rank-one chunk streams.
"""

import math

import torch

from asc.experimental import asctile

from ._pyasc_runtime import (
    ensure_npu_platform,
    query_ub_bytes,
    query_aivs,
    vector_blocks,
)


# ---------------------------------------------------------------------------
# Kernel 1: short-half gather slab (rank-one copy + two reindex gathers)
# ---------------------------------------------------------------------------
@asctile.jit(reuse_alloc=2)
def _swiglu_slab(
    x: asctile.GlobalAddress,
    y: asctile.GlobalAddress,
    gate_idx: asctile.GlobalAddress,
    value_idx: asctile.GlobalAddress,
    R: asctile.ConstExpr[int],
    H: asctile.ConstExpr[int],
    T: asctile.ConstExpr[int],
    U: asctile.ConstExpr[int],
):
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    g_gate = asctile.global_tensor(gate_idx, [T * H])
    g_value = asctile.global_tensor(value_idx, [T * H])

    gate_ub = asctile.copy_in(g_gate, [0], [T * H])
    value_ub = asctile.copy_in(g_value, [0], [T * H])

    tile_count = (R + T - 1) // T
    block = asctile.block_idx()
    block_count = asctile.block_num()

    for t in asctile.range(block, tile_count, block_count, unroll_factor=U):
        r = t * T
        rows = min(T, R - r)
        slab_len = 2 * T * H
        real_len = 2 * rows * H
        slab = asctile.copy_in(
            gx, [2 * r * H], [slab_len], real_shape=[real_len], pad_value=0.0
        )
        gate = asctile.reindex(slab, gate_ub)
        value = asctile.reindex(slab, value_ub)

        gate_f = gate.to(asctile.float32)
        value_f = value.to(asctile.float32)

        sig = gate_f / (1.0 + asctile.exp(-gate_f))
        out_f = sig * value_f

        out_tile = out_f.to(gate.dtype)
        out_len = rows * H
        asctile.copy_out(out_tile, gy, [r * H], real_shape=[out_len])


# ---------------------------------------------------------------------------
# Kernel 2: strided rank-two row tiles (two input loads, one store)
# ---------------------------------------------------------------------------
@asctile.jit(reuse_alloc=2)
def _swiglu_rows(
    x: asctile.GlobalAddress,
    y: asctile.GlobalAddress,
    R: asctile.ConstExpr[int],
    H: asctile.ConstExpr[int],
    T: asctile.ConstExpr[int],
    W: asctile.ConstExpr[int],
    U: asctile.ConstExpr[int],
):
    gx = asctile.global_tensor(x, [R, 2 * H])
    gy = asctile.global_tensor(y, [R, H])

    tile_count = (R + T - 1) // T
    block = asctile.block_idx()
    block_count = asctile.block_num()

    for t in asctile.range(block, tile_count, block_count, unroll_factor=U):
        r = t * T
        rows = min(T, R - r)
        gate_tile = asctile.copy_in(
            gx, [r, 0], [T, W], real_shape=[rows, H], pad_value=0.0
        )
        value_tile = asctile.copy_in(
            gx, [r, H], [T, W], real_shape=[rows, H], pad_value=0.0
        )

        gate_f = gate_tile.to(asctile.float32)
        value_f = value_tile.to(asctile.float32)

        sig = gate_f / (1.0 + asctile.exp(-gate_f))
        out_f = sig * value_f

        out_tile = out_f.to(gate_tile.dtype)
        asctile.copy_out(out_tile, gy, [r, 0], real_shape=[rows, H])


# ---------------------------------------------------------------------------
# Kernel 3: flat contiguous chunk streams
# ---------------------------------------------------------------------------
@asctile.jit(reuse_alloc=2)
def _swiglu_flat(
    x: asctile.GlobalAddress,
    y: asctile.GlobalAddress,
    R: asctile.ConstExpr[int],
    H: asctile.ConstExpr[int],
    T: asctile.ConstExpr[int],
    U: asctile.ConstExpr[int],
):
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])

    whole = H // T
    nfull = R * whole
    blocks = asctile.block_num()
    block = asctile.block_idx()

    q = nfull // blocks
    rem = nfull % blocks
    begin = q * block + min(block, rem)
    end = q * (block + 1) + min(block + 1, rem)

    whole_safe = max(1, whole)
    for t in asctile.range(begin, end, unroll_factor=U):
        row = t // whole_safe
        col = (t % whole_safe) * T
        gate_tile = asctile.copy_in(gx, [row * 2 * H + col], [T])
        value_tile = asctile.copy_in(gx, [row * 2 * H + H + col], [T])

        gate_f = gate_tile.to(asctile.float32)
        value_f = value_tile.to(asctile.float32)

        sig = gate_f / (1.0 + asctile.exp(-gate_f))
        out_f = sig * value_f

        out_tile = out_f.to(gate_tile.dtype)
        asctile.copy_out(out_tile, gy, [row * H + col])

    if H % T:
        valid = H % T
        col = whole * T
        for row in asctile.range(block, R, blocks):
            gate_tile = asctile.copy_in(
                gx, [row * 2 * H + col], [T], real_shape=[valid], pad_value=0.0
            )
            value_tile = asctile.copy_in(
                gx, [row * 2 * H + H + col], [T], real_shape=[valid], pad_value=0.0
            )

            gate_f = gate_tile.to(asctile.float32)
            value_f = value_tile.to(asctile.float32)

            sig = gate_f / (1.0 + asctile.exp(-gate_f))
            out_f = sig * value_f

            out_tile = out_f.to(gate_tile.dtype)
            asctile.copy_out(out_tile, gy, [row * H + col], real_shape=[valid])


# ---------------------------------------------------------------------------
# Host-side caches and geometry helpers
# ---------------------------------------------------------------------------
_SLAB_INDEX_CACHE = {}


def _dtype_size(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    return 4


def _dtype_label(dtype):
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return "fp32"


def _next_pow2_floor(value):
    if value <= 1:
        return 1
    return 1 << (value.bit_length() - 1)


def _align_up(value, align):
    return ((value + align - 1) // align) * align


def _get_slab_indices(device, H, T):
    key = (str(device), H, T)
    cached = _SLAB_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    total = T * H
    k = torch.arange(total, dtype=torch.int32, device=device)
    gate_idx = 2 * H * (k // H) + (k % H)
    value_idx = gate_idx + H
    _SLAB_INDEX_CACHE[key] = (gate_idx, value_idx)
    return gate_idx, value_idx


# ---------------------------------------------------------------------------
# Branch selection: short-half BF16 slab
# ---------------------------------------------------------------------------
def _try_slab(R, H, itemsize, ub_bytes, aivs):
    if R <= 0 or H <= 0:
        return None
    if H * itemsize > 32:
        return None
    payload_bytes_per_row = 2 * H * itemsize

    dma_rows = 16384 // max(1, payload_bytes_per_row)
    ub_rows_u1 = (ub_bytes // 2) // max(1, 24 * H)
    ub_rows_u2 = (ub_bytes // 2) // max(1, 28 * H)
    useful_rows = (R + aivs - 1) // aivs

    bound = min(dma_rows, ub_rows_u1, useful_rows)
    if bound < 1:
        return None

    T = _next_pow2_floor(bound)
    if T < 1:
        return None

    # byte-alignment checks for the selected tile
    if (2 * T * H * itemsize) % 32 != 0:
        return None
    if (T * H * itemsize) % 32 != 0:
        return None
    if (4 * T * H) % 32 != 0:
        return None

    tile_count = (R + T - 1) // T
    U = 1
    if ub_rows_u2 >= T and T >= 1:
        if (tile_count + aivs - 1) // aivs >= 2:
            U = 2

    blocks = min(aivs, max(1, tile_count))
    return {
        "kind": "slab",
        "T": T,
        "U": U,
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Branch selection: strided rank-two row tiles
# ---------------------------------------------------------------------------
def _try_rows(R, H, itemsize, ub_bytes, aivs):
    if R <= 0 or H <= 0:
        return None
    payload_per_row = H * itemsize
    if payload_per_row <= 0:
        return None

    # Smallest power-of-two height with >= 8KiB useful payload (min 1 row).
    target = 8192
    T = max(1, target // payload_per_row)
    T = _next_pow2_floor(T)
    if T < 1:
        T = 1

    align_elems = max(1, 32 // itemsize)
    W = _align_up(H, align_elems)

    # UB coefficient per unroll
    def coeff(unroll, dtype_label):
        if unroll == 2:
            if dtype_label == "fp16":
                return 20
            if dtype_label == "bf16":
                return 32
            return 28
        if dtype_label == "fp16":
            return 16
        if dtype_label == "bf16":
            return 24
        return 16

    return T, W, coeff


def _select_rows(R, H, itemsize, dtype_label, ub_bytes, aivs):
    info = _try_rows(R, H, itemsize, ub_bytes, aivs)
    if info is None:
        return None
    T, W, coeff = info

    while T >= 1:
        if T > R:
            T //= 2
            continue
        lane = T * W
        c1 = coeff(1, dtype_label)
        c2 = coeff(2, dtype_label)
        need1 = lane * c1
        need2 = lane * c2
        tile_count = (R + T - 1) // T
        avg_tiles = (tile_count + aivs - 1) // aivs if aivs > 0 else 1

        if need1 <= ub_bytes:
            if need2 <= ub_bytes and avg_tiles >= 2:
                U = 2
            else:
                U = 1
            blocks = min(aivs, max(1, tile_count))
            return {
                "kind": "rows",
                "T": T,
                "W": W,
                "U": U,
                "blocks": blocks,
            }
        T //= 2

    return None


# ---------------------------------------------------------------------------
# Branch selection: flat chunk streams
# ---------------------------------------------------------------------------
def _select_flat(R, H, itemsize, ub_bytes, aivs):
    if R <= 0 or H <= 0:
        return None
    # Budget ~16KiB worth of elements per chunk, bounded by UB.
    max_elems_dma = 16384 // max(1, itemsize)
    max_elems_ub = ub_bytes // max(1, 8 * 4)  # gate+value fp32 live ~ 8 bytes/elt
    T = max(1, min(H, max_elems_dma, max_elems_ub))
    T = _next_pow2_floor(T)
    if T < 1:
        T = 1
    if T > H:
        T = H

    whole = H // T
    tail = 1 if (H % T) else 0
    live_fulls = R * whole
    live_tail = R * tail
    live_total = live_fulls + live_tail
    if live_total <= 0:
        return None

    tile_count = live_fulls + live_tail
    U = 1
    if tile_count >= 2 * max(1, aivs):
        U = 2
    blocks = min(aivs, max(1, tile_count))
    return {
        "kind": "flat",
        "T": T,
        "U": U,
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Public operator
# ---------------------------------------------------------------------------
def swi_glu(input, dim=-1):
    ensure_npu_platform()

    if not torch.is_tensor(input):
        raise TypeError("swi_glu expects a torch.Tensor input")

    ndim = input.dim()
    if ndim == 0:
        raise ValueError("swi_glu requires at least 1 dimension")

    d = dim if dim >= 0 else dim + ndim
    if d < 0 or d >= ndim:
        raise IndexError(f"dim {dim} out of range for {ndim}D tensor")

    shape = list(input.shape)
    split_size = shape[d]
    if split_size % 2 != 0:
        raise ValueError(
            f"split dim {d} has odd size {split_size}; must be even"
        )
    if split_size == 0:
        out_shape = list(shape)
        out_shape[d] = 0
        return input.new_empty(out_shape)

    half = split_size // 2

    R = 1
    for i in range(d):
        R *= shape[i]
    T_after = 1
    for i in range(d + 1, ndim):
        T_after *= shape[i]
    H = half * T_after

    out_shape = list(shape)
    out_shape[d] = half
    out = input.new_empty(out_shape)

    source = input if input.is_contiguous() else input.contiguous()

    ub_bytes = query_ub_bytes()
    aivs = query_aivs()

    itemsize = _dtype_size(source.dtype)
    dtype_label = _dtype_label(source.dtype)

    plan = None
    if dtype_label == "bf16":
        plan = _try_slab(R, H, itemsize, ub_bytes, aivs)

    if plan is None:
        plan = _select_rows(R, H, itemsize, dtype_label, ub_bytes, aivs)

    if plan is None:
        plan = _select_flat(R, H, itemsize, ub_bytes, aivs)

    if plan is None:
        # Last resort: single-element flat chunks.
        plan = {
            "kind": "flat",
            "T": 1,
            "U": 1,
            "blocks": max(1, min(aivs, max(1, R * H))),
        }

    x_flat = source.reshape(-1)
    y_flat = out.reshape(-1)

    kind = plan["kind"]

    if kind == "slab":
        T = plan["T"]
        U = plan["U"]
        tile_count = (R + T - 1) // T
        work_items = tile_count
        blocks = plan["blocks"]
        gate_idx, value_idx = _get_slab_indices(x_flat.device, H, T)
        _swiglu_slab[vector_blocks(work_items)](
            x_flat, y_flat, gate_idx, value_idx, R, H, T, U
        )
        return out

    if kind == "rows":
        T = plan["T"]
        W = plan["W"]
        U = plan["U"]
        tile_count = (R + T - 1) // T
        work_items = tile_count
        _swiglu_rows[vector_blocks(work_items)](
            x_flat, y_flat, R, H, T, W, U
        )
        return out

    # flat
    T = plan["T"]
    U = plan["U"]
    whole = H // T
    tail = 1 if (H % T) else 0
    work_items = R * whole + R * tail
    if work_items <= 0:
        work_items = 1
    _swiglu_flat[vector_blocks(work_items)](
        x_flat, y_flat, R, H, T, U
    )
    return out


__all__ = ["swi_glu"]
