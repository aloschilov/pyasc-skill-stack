from asc.experimental import asctile


@asctile.jit(always_compile=False)
def swiglu_rows(x: asctile.GlobalAddress,
                y: asctile.GlobalAddress,
                R: int,
                H: int,
                TR: asctile.ConstExpr[int],
                HC: asctile.ConstExpr[int],
                U: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x, [R, 2 * H])
    y_gm = asctile.global_tensor(y, [R, H])

    nr = (R + TR - 1) // TR
    nc = (H + HC - 1) // HC
    ntiles = nr * nc

    for t in asctile.range(asctile.block_idx(), ntiles, asctile.block_num(),
                           unroll_factor=U):
        rt_idx = t // nc
        ct_idx = t % nc
        r0 = rt_idx * TR
        c0 = ct_idx * HC
        r_valid = min(TR, R - r0)
        c_valid = min(HC, H - c0)

        gate = asctile.copy_in(x_gm, [r0, c0], [TR, HC],
                               real_shape=[r_valid, c_valid], pad_value=0.0)
        val = asctile.copy_in(x_gm, [r0, H + c0], [TR, HC],
                              real_shape=[r_valid, c_valid], pad_value=0.0)

        gatef = asctile.cast(gate, asctile.float32)
        valf = asctile.cast(val, asctile.float32)

        e = asctile.exp(asctile.mul(gatef, -1.0))
        s = asctile.div(1.0, asctile.add(e, 1.0))
        res = asctile.mul(asctile.mul(gatef, s), valf)

        out_t = res.to(gate.dtype)
        asctile.copy_out(out_t, y_gm, [r0, c0],
                         real_shape=[r_valid, c_valid])
from asc.experimental import asctile


@asctile.jit(always_compile=False)
def swiglu_slab(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, i1: asctile.GlobalAddress, R: int, H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])
    gi1 = asctile.global_tensor(i1, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H])
    idx1 = asctile.copy_in(gi1, [0], [TR * H])

    ntiles = (R + TR - 1) // TR

    # Grid-stride loop over row tiles
    for t in asctile.range(asctile.block_idx(), ntiles, asctile.block_num(), unroll_factor=U):
        r0 = t * TR
        valid = min(TR, R - r0)

        # Load one contiguous slab (gate+value), bounded by valid row count
        slab = asctile.copy_in(gx, [2 * r0 * H], [2 * TR * H],
                               real_shape=[2 * valid * H], pad_value=0.0)

        # Gather gate / value elements via the cached indices
        g = asctile.reindex(slab, idx0)
        v = asctile.reindex(slab, idx1)

        # FP32 math, cast back to input dtype
        out_dtype = slab.dtype
        gf = g.to(asctile.float32)
        vf = v.to(asctile.float32)

        # SwiGLU: (gate / (1 + exp(-gate))) * value
        res = (gf / (asctile.exp(-gf) + 1.0)) * vf
        out = res.to(out_dtype)

        # Store at row offset, bounded by valid row count
        asctile.copy_out(out, gy, [r0 * H], real_shape=[valid * H])
from asc.experimental import asctile


@asctile.jit(always_compile=False)
def swiglu_flat(x: asctile.GlobalAddress, y: asctile.GlobalAddress, R: int, H: int, T: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    ncol = asctile.ceildiv(H, T)
    total = R * ncol

    for t in asctile.range(asctile.block_idx(), total, asctile.block_num(), unroll_factor=U):
        row = t // ncol
        col = (t % ncol) * T

        valid = min(T, H - col)

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

        out_dtype = gate_tile.dtype
        out_tile = out_f.to(out_dtype)

        asctile.copy_out(out_tile, gy, [row * H + col], real_shape=[valid])
@asctile.jit
def _swiglu_pair_groups(xptr: asctile.GlobalAddress, yptr: asctile.GlobalAddress,
                        rows: int, groups: int, tile: asctile.ConstExpr[int]):
    x = asctile.global_tensor(xptr, [2 * rows])
    y = asctile.global_tensor(yptr, [rows])
    for group in asctile.range(asctile.block_idx(), groups, asctile.block_num()):
        first = group * tile
        valid = min(tile, rows - first)
        flat = asctile.copy_in(x, [2 * first], [2 * tile],
                               real_shape=[2 * valid], pad_value=0.0)
        # A pair tile is [16, tile // 16, 2]. Move parity to the leading axis.
        columns = flat.reshape(16, tile // 8).transpose(1, 0)
        planes = columns.reshape(tile // 16, 32).transpose(1, 0).reshape(2, tile)
        xy = asctile.cast(planes, asctile.float32)
        plane_id = asctile.concat(asctile.full([1, tile], 0.0, asctile.float32),
                                  asctile.full([1, tile], 1.0, asctile.float32))
        is_gate = plane_id < 0.5
        gate = asctile.reduce_sum(asctile.where(is_gate, xy, 0.0), 0)
        value = asctile.reduce_sum(asctile.where(is_gate, 0.0, xy), 0)
        sigmoid = asctile.div(1.0, asctile.add(asctile.exp(-gate), 1.0))
        result = asctile.cast(gate * sigmoid * value, asctile.bfloat16)
        asctile.copy_out(result.reshape(tile), y, [first], real_shape=[valid])

def select(rows, half, itemsize, is_bf16):
    """SwiGLU split-axis route selection for UB=253952 B, 72 cores.

    Routes (chosen from layout/dtype only, never case ID or outer shape):
      packed_pairs  -- bf16 [R,2]: neighbor (gate,value) pairs on last axis
      index_gather  -- very short half-rows: two reusable int32 index tiles
      batched_rows  -- short half-rows: 2D tile, 32B-aligned physical width
      flat_chunks   -- long contiguous half-rows split into flat chunks
    """
    UB = 253952          # usable unified-buffer bytes
    CORES = 72           # independent tiles available
    row_bytes = half * itemsize

    def align32_down_elems(n):
        # Round an element count down to a 32-byte boundary (>=1 lane group).
        unit = max(1, (32 // itemsize))
        return max(unit, (n // unit) * unit)

    # Packed (gate, value) neighbor pairs: bf16 final axis of extent two.
    # Guard matches dtype + normalized axis semantics (half == 1 here means
    # shape[d] == 2, not extent of d). One pair per lane, gate exp in-tile.
    # Packed transpose on this pin fits 2048 pairs max.
    if is_bf16 and itemsize == 2 and half == 1:
        pairs = min(rows, 2048)
        return ("packed_pairs", pairs, 1)

    # Very short halves: gather complete rows via two local int32 index tiles.
    # Budget: two data half-tiles + two int32 index tiles (4 bytes/index).
    if row_bytes <= 32:
        per_block = max(1, min(rows, (UB // 4) // (row_bytes + 4)))
        per_block = min(per_block, 2 * CORES)   # bound block height vs grid
        return ("index_gather", per_block, 1)

    # Short half-rows: batch into 2D tiles targeting ~12 KiB per half-tile.
    # Physical column alignment is the caller's job; we return row count only,
    # with both height and logical width bounded by the tile budget.
    if row_bytes <= 3072:
        height = max(1, 12288 // ((row_bytes + 31) // 32 * 32))
        height = min(height, rows, CORES)
        return ("batched_rows", height, 1)

    # Long contiguous half-row: flat chunk stream, gate/value offsets H apart.
    # Flat unroll 2 fits 10560 elements max on this pin. Tile is an element
    # count, aligned down to 32 bytes (physical alignment applied by caller).
    chunk_elems = ((row_bytes + 31) // 32) * 32 // itemsize
    chunk = align32_down_elems(min(10560, chunk_elems))
    return ("flat_chunks", chunk, 2)
ROUTES = {"packed_pairs":"packed", "index_gather":"reindex", "batched_rows":"2d", "flat_chunks":"flat"}
from functools import lru_cache
from math import prod

import torch
from ._pyasc_runtime import ensure_npu_platform, vector_blocks


@lru_cache(maxsize=32)
def gather_indices(half, tile, device):
    lane = torch.arange(tile * half, device=device, dtype=torch.int32)
    gate = (lane // half) * (2 * half) + lane % half
    return gate, gate + half


def swi_glu(input, dim=-1):
    shape = list(input.shape)
    if not -len(shape) <= dim < len(shape):
        raise IndexError(dim)
    dim %= len(shape)
    if shape[dim] % 2:
        raise ValueError("SwiGLU requires equal halves")
    shape[dim] //= 2
    rows, half = prod(shape[:dim]), prod(shape[dim:])
    out = input.new_empty(shape)
    if not out.numel():
        return out
    ensure_npu_platform()
    itemsize = input.element_size()
    pair = input.dtype == torch.bfloat16 and dim == len(shape) - 1 and half == 1
    layout, tile, unroll = select(rows, half, itemsize, pair)
    route = ROUTES[layout]
    x, y = input.contiguous().view(-1), out.view(-1)
    groups = (rows + tile - 1) // tile
    if route == "flat":
        work = rows * ((half + tile - 1) // tile)
        swiglu_flat[vector_blocks(work)](x, y, rows, half, tile, unroll, reuse_alloc=2)
    elif route == "2d":
        align = 32 // itemsize
        width = (half + align - 1) // align * align
        swiglu_rows[vector_blocks(groups)](x, y, rows, half, tile, width, unroll, reuse_alloc=2)
    elif route == "reindex":
        i0, i1 = gather_indices(half, tile, x.device)
        swiglu_slab[vector_blocks(groups)](x, y, i0, i1, rows, half, tile, unroll, reuse_alloc=2)
    elif route == "packed":
        _swiglu_pair_groups[vector_blocks(groups)](x, y, rows, groups, tile)
    else:
        raise ValueError(layout)
    return out
