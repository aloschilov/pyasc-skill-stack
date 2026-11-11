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

def select(rows, half, itemsize, is_pair, is_bf16, ub_bytes, aivs):
    """SwiGLU split-axis route selection with live UB/AIV capacities.

    Routes (chosen from layout/dtype only, never case ID or outer shape):
      packed_pairs  -- bf16 [R,2]: neighbor (gate,value) pairs on last axis
      index_gather  -- short half-rows: public reindex gather, two cached
                       int32 index vectors, DMA-capped power-of-two height
      batched_rows  -- short half-rows (or bounded 2D fallback): 2D tile,
                       32B-aligned physical width
      flat_chunks   -- long contiguous half-rows split into flat chunks

    On the archive-bundled 950PR public build, dense BF16 pairs (half == 1)
    are dispatched through index_gather rather than the legacy packed_pairs
    early return.

    `ub_bytes` is the effective usable unified-buffer capacity (any fixed
    workspace already subtracted); `aivs` is the active AIV count.
    """
    UB = max(1, ub_bytes)          # effective usable unified-buffer bytes
    CORES = max(1, aivs)           # active independent tiles / AIVs
    half_bytes = half * itemsize   # gathered half-row bytes per row
    in_row_bytes = 2 * half_bytes  # contiguous input bytes per full row

    def align32_down_elems(n):
        # Round an element count down to a 32-byte boundary (>=1 lane group).
        unit = max(1, (32 // itemsize))
        return max(unit, (n // unit) * unit)

    def bounded_2d_fallback():
        # Bounded 2D tile: ~12 KiB per half-tile, height capped by grid.
        height = max(1, 12288 // ((half_bytes + 31) // 32 * 32))
        height = min(height, max(1, rows), CORES)
        return ("batched_rows", height, 1)

    # Dense BF16 pairs intentionally fall through to the short-half gather
    # branch below on this archive-bundled 950PR public build; the legacy
    # packed-transpose early return is omitted so they share the same
    # qualified reindex selector as other short BF16 halves.
    if half_bytes <= 32:
        if is_bf16 and itemsize == 2:
            # Eligible short-half-row BF16 gather: bounded row geometry.
            # Bounds (all must hold):
            #   DMA target 16 KiB  -> 16384 // (2*H*s) rows per input DMA
            #   single-iteration live cost 24*H bytes/row -> UB // (24*H)
            #   useful rows per AIV -> ceil(R / active_AIVs)
            dma_cap = max(0, 16384 // in_row_bytes) if in_row_bytes else 0
            ub_cap = UB // (24 * half) if half > 0 else 0
            aiv_cap = -(-max(1, rows) // CORES)  # ceil
            bound = min(dma_cap, ub_cap, aiv_cap)

            if bound >= 1:
                # Largest power-of-two row count within all three bounds.
                t = 1 << (bound.bit_length() - 1)

                # Byte-alignment guard: physical input tile, gathered output
                # tile and both int32 index vectors must each be 32B-aligned.
                # If the largest power-of-two tile is misaligned, every
                # smaller power of two carries fewer factors of two and also
                # fails, so the bounded row fallback is the only recourse.
                in_tile_bytes = 2 * t * half * itemsize   # input DMA bytes
                out_tile_bytes = t * half * itemsize      # gathered output
                idx_tile_bytes = 4 * t * half             # int32 index vector
                if (in_tile_bytes % 32 == 0
                        and out_tile_bytes % 32 == 0
                        and idx_tile_bytes % 32 == 0):
                    tiles = -(-max(1, rows) // t)  # ceil(R / T)

                    # Prefer unroll 2: two-iteration footprint (28*H/row) fits
                    # UB and >= two tiles per active block on average.
                    if 28 * half * t <= UB and tiles >= 2 * CORES:
                        return ("index_gather", t, 2)
                    # Otherwise unroll 1 at this height (footprint fits by
                    # construction of the bound).
                    if 24 * half * t <= UB:
                        return ("index_gather", t, 1)
                # Misaligned tile (or neither unroll fits): fall through to
                # the bounded 2D row fallback below.

            # No aligned tile fits: bounded 2D fallback.
            return bounded_2d_fallback()

        # Non-BF16 short halves: gather complete rows via two local int32
        # index tiles. Budget: two data half-tiles + two int32 index tiles.
        per_block = max(1, min(max(1, rows), (UB // 4) // (half_bytes + 4)))
        per_block = min(per_block, 2 * CORES)  # bound block height vs grid
        return ("index_gather", per_block, 1)

    # Short half-rows (wider): batch into 2D tiles targeting ~12 KiB per
    # half-tile. Physical column alignment is the caller's job.
    if half_bytes <= 3072:
        return bounded_2d_fallback()

    # Long contiguous half-row: flat chunk stream, gate/value offsets H apart.
    # Flat unroll 2 fits 10560 elements max on this pin. Tile is an element
    # count, aligned down to 32 bytes (physical alignment applied by caller).
    chunk_elems = ((half_bytes + 31) // 32) * 32 // itemsize
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


from ._pyasc_runtime import query_aivs, query_ub_bytes
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
    ub_bytes = query_ub_bytes()
    aivs = query_aivs()
    itemsize = input.element_size()
    is_pair = dim == len(shape) - 1 and half == 1
    is_bf16 = input.dtype == torch.bfloat16
    layout, tile, unroll = select(rows, half, itemsize, is_pair, is_bf16, ub_bytes, aivs)
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
