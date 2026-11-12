from asc.experimental import asctile


@asctile.jit(always_compile=False)
def swiglu_whole_row(x:asctile.GlobalAddress,y:asctile.GlobalAddress,R:asctile.ConstExpr[int],H:asctile.ConstExpr[int],U:asctile.ConstExpr[int],N:asctile.ConstExpr[int]):
    gx=asctile.global_tensor(x,[2*R*H]);gy=asctile.global_tensor(y,[R*H]);b=asctile.block_idx()
    for step in asctile.range(R//N,unroll_factor=U):
        row=step*N+b
        gate=asctile.copy_in(gx,[row*2*H],[H]);val=asctile.copy_in(gx,[row*2*H+H],[H])
        gf=gate.to(asctile.float32);vf=val.to(asctile.float32)
        out=((gf/(asctile.exp(-gf)+1.0))*vf).to(gate.dtype)
        asctile.copy_out(out,gy,[row*H])


@asctile.jit(always_compile=False)
def swiglu_seed_waves(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    seed=asctile.copy_in(gi0,[0],[TR*H//2])
    idx0=asctile.concat(seed,seed+TR*H)
    idx1 = idx0 + H

    ntiles = (R + TR - 1) // TR

    # Grid-stride loop over row tiles
    block=asctile.block_idx()
    n=asctile.block_num()
    q=ntiles//n
    rem=ntiles-q*n
    lo=q*block+min(block,rem)
    hi=lo+q+min(1,max(0,rem-block))
    for t in asctile.range(lo,hi,unroll_factor=U):
        r0 = t * TR
        valid = min(TR, R - r0)

        # Load one contiguous slab (gate+value), bounded by valid row count
        slab = asctile.copy_in(gx, [2 * r0 * H], [2 * TR * H],
                               real_shape=[2 * valid * H], pad_value=0.0)

        # Gather gate / value elements via the cached indices
        g = asctile.reindex(slab,idx0).to(asctile.float32)
        denom = asctile.exp(-g)+1.0
        activated = g/denom
        v = asctile.reindex(slab,idx1).to(asctile.float32)
        out = (activated*v).to(slab.dtype)
        # Store at row offset, bounded by valid row count
        asctile.copy_out(out, gy, [r0 * H], real_shape=[valid * H])

@asctile.jit(always_compile=False)
def swiglu_column_stripe(x: asctile.GlobalAddress, y: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], T: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    ncol = asctile.ceildiv(H, T)
    total = R * ncol

    for t in asctile.range(asctile.block_idx(), total, asctile.block_num(), unroll_factor=U):
        row = t % R
        col = (t // R) * T

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

@asctile.jit(always_compile=False)
def swiglu_selected_compact(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    span=TR*H//2
    seed=asctile.copy_in(gi0,[0],[span])
    pair=asctile.concat(seed,seed+2*span)
    idx0=pair
    idx1 = idx0 + H

    ntiles = (R + TR - 1) // TR

    # Grid-stride loop over row tiles
    b=asctile.block_idx()
    n=asctile.block_num()
    first=(b%2)*(n//2) + b//2
    for t in asctile.range(first,ntiles,n,unroll_factor=U):
        r0 = t * TR
        valid = min(TR, R - r0)

        # Load one contiguous slab (gate+value), bounded by valid row count
        slab = asctile.copy_in(gx, [2 * r0 * H], [2 * TR * H])

        # Gather gate / value elements via the cached indices
        g=asctile.reindex(slab,idx0)
        act=g/(1.0+asctile.exp(-g))
        v=asctile.reindex(slab,idx1)
        out=act*v
        # Store at row offset, bounded by valid row count
        asctile.copy_out(out, gy, [r0 * H])

@asctile.jit(always_compile=False)
def swiglu_selected_pairs(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H])
    idx1 = idx0 + H

    ntiles = (R + TR - 1) // TR

    # Grid-stride loop over row tiles
    block=asctile.block_idx()
    n=asctile.block_num()
    first=(block%4)*(n//4)+block//4
    for t in asctile.range(first, ntiles, n, unroll_factor=U):
        r0 = t * TR
        valid = min(TR, R - r0)

        # Load one contiguous slab (gate+value), bounded by valid row count
        slab = asctile.copy_in(gx, [2 * r0 * H], [2 * TR * H],
                               real_shape=[2 * valid * H], pad_value=0.0)

        # Gather gate / value elements via the cached indices
        g = asctile.reindex(slab,idx0).to(asctile.float32)
        denom = asctile.exp(-g)+1.0
        activated = g/denom
        v = asctile.reindex(slab,idx1).to(asctile.float32)
        out = (activated*v).to(slab.dtype)
        # Store at row offset, bounded by valid row count
        asctile.copy_out(out, gy, [r0 * H], real_shape=[valid * H])

@asctile.jit(always_compile=False)
def swiglu_local_waves(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H])
    idx1 = idx0 + H

    ntiles = (R + TR - 1) // TR

    # Grid-stride loop over row tiles
    block=asctile.block_idx()
    n=asctile.block_num()
    q=ntiles//n
    rem=ntiles-q*n
    lo=q*block+min(block,rem)
    hi=lo+q+min(1,max(0,rem-block))
    for t in asctile.range(lo,hi,unroll_factor=U):
        r0 = t * TR
        valid = min(TR, R - r0)

        # Load one contiguous slab (gate+value), bounded by valid row count
        slab = asctile.copy_in(gx, [2 * r0 * H], [2 * TR * H],
                               real_shape=[2 * valid * H], pad_value=0.0)

        # Gather gate / value elements via the cached indices
        g = asctile.reindex(slab,idx0).to(asctile.float32)
        denom = asctile.exp(-g)+1.0
        activated = g/denom
        v = asctile.reindex(slab,idx1).to(asctile.float32)
        out = (activated*v).to(slab.dtype)
        # Store at row offset, bounded by valid row count
        asctile.copy_out(out, gy, [r0 * H], real_shape=[valid * H])

@asctile.jit(always_compile=False)
def swiglu_flat_edge(x:asctile.GlobalAddress,y:asctile.GlobalAddress,H:asctile.ConstExpr[int],T:asctile.ConstExpr[int],U:asctile.ConstExpr[int],N:asctile.ConstExpr[int],E:asctile.ConstExpr[int]):
    gx=asctile.global_tensor(x,[2*H]);gy=asctile.global_tensor(y,[H])
    block=asctile.block_idx()
    waves=H//(N*T)
    if waves>0:
        for step in asctile.range(waves,unroll_factor=U):
            col=(step*N+block)*T
            g=asctile.copy_in(gx,[col],[T]);v=asctile.copy_in(gx,[H+col],[T])
            gf=g.to(asctile.float32);vf=v.to(asctile.float32)
            out=((gf/(1.0+asctile.exp(-gf)))*vf).to(g.dtype)
            asctile.copy_out(out,gy,[col])
    if H%(N*T):
        tail=H%(N*T)
        off=block*E
        if off<tail:
            col=waves*N*T+off
            count=min(E,tail-off)
            g=asctile.copy_in(gx,[col],[E],real_shape=[count],pad_value=0.0)
            v=asctile.copy_in(gx,[H+col],[E],real_shape=[count],pad_value=0.0)
            gf=g.to(asctile.float32);vf=v.to(asctile.float32)
            out=((gf/(1.0+asctile.exp(-gf)))*vf).to(g.dtype)
            asctile.copy_out(out,gy,[col],real_shape=[count])

@asctile.jit(always_compile=False)
def swiglu_aligned_rows(x: asctile.GlobalAddress,
                y: asctile.GlobalAddress,
                R: asctile.ConstExpr[int],
                H: asctile.ConstExpr[int],
                TR: asctile.ConstExpr[int],
                HC: asctile.ConstExpr[int],
                U: asctile.ConstExpr[int], N: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x, [R,2*H])
    y_gm = asctile.global_tensor(y, [R,H])
    block = asctile.block_idx()
    waves = R // (TR*N)
    if waves > 0:
        for step in asctile.range(waves,unroll_factor=U):
            r0=(step*N+block)*TR
            gate = asctile.copy_in(x_gm,[r0,0],[TR,HC])
            value = asctile.copy_in(x_gm,[r0,H],[TR,HC])
            g = gate.to(asctile.float32)
            v = value.to(asctile.float32)
            result = ((g/(1.0+asctile.exp(-g)))*v).to(gate.dtype)
            asctile.copy_out(result,y_gm,[r0,0])
    if R % (TR*N):
        r0=(waves*N+block)*TR
        if r0 < R:
            gate = asctile.copy_in(x_gm,[r0,0],[TR,HC],real_shape=[min(TR,R-r0),H],pad_value=0.0)
            value = asctile.copy_in(x_gm,[r0,H],[TR,HC],real_shape=[min(TR,R-r0),H],pad_value=0.0)
            g = gate.to(asctile.float32)
            v = value.to(asctile.float32)
            result = ((g/(1.0+asctile.exp(-g)))*v).to(gate.dtype)
            asctile.copy_out(result,y_gm,[r0,0],real_shape=[min(TR,R-r0),H])

@asctile.jit(always_compile=False)
def swiglu_wave_slab(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, i1: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int], N: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])
    gi1 = asctile.global_tensor(i1, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H])
    idx1 = idx0 + H

    block=asctile.block_idx()
    full=R//TR
    waves=full//N
    if waves>0:
        for step in asctile.range(waves,unroll_factor=U):
            r0=(step*N+block)*TR
            slab=asctile.copy_in(gx,[2*r0*H],[2*TR*H])
            g=asctile.reindex(slab,idx0).to(asctile.float32)
            activated=g/(asctile.exp(-g)+1.0)
            v=asctile.reindex(slab,idx1).to(asctile.float32)
            out=(activated*v).to(slab.dtype)
            asctile.copy_out(out,gy,[r0*H])
    if full%N:
        if block < full%N:
            r0=(waves*N+block)*TR
            slab=asctile.copy_in(gx,[2*r0*H],[2*TR*H])
            g=asctile.reindex(slab,idx0).to(asctile.float32)
            activated=g/(asctile.exp(-g)+1.0)
            v=asctile.reindex(slab,idx1).to(asctile.float32)
            out=(activated*v).to(slab.dtype)
            asctile.copy_out(out,gy,[r0*H])
    if R%TR:
        if block == full%N:
            r0=full*TR
            slab=asctile.copy_in(gx,[2*r0*H],[2*TR*H],real_shape=[2*H*(R%TR)],pad_value=0.0)
            g=asctile.reindex(slab,idx0).to(asctile.float32)
            activated=g/(asctile.exp(-g)+1.0)
            v=asctile.reindex(slab,idx1).to(asctile.float32)
            out=(activated*v).to(slab.dtype)
            asctile.copy_out(out,gy,[r0*H],real_shape=[H*(R%TR)])

@asctile.jit(always_compile=False)
def swiglu_full_slab(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H])
    idx1 = idx0 + H

    ntiles = (R + TR - 1) // TR

    # Grid-stride loop over row tiles
    b=asctile.block_idx()
    n=asctile.block_num()
    first=(b%2)*(n//2) + b//2
    for t in asctile.range(first,ntiles,n,unroll_factor=U):
        r0 = t * TR
        valid = min(TR, R - r0)

        # Load one contiguous slab (gate+value), bounded by valid row count
        slab = asctile.copy_in(gx, [2 * r0 * H], [2 * TR * H])

        # Gather gate / value elements via the cached indices
        g=asctile.reindex(slab,idx0)
        act=g/(1.0+asctile.exp(-g))
        v=asctile.reindex(slab,idx1)
        out=act*v
        # Store at row offset, bounded by valid row count
        asctile.copy_out(out, gy, [r0 * H])

@asctile.jit(always_compile=False)
def swiglu_permuted_pairs(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H])
    idx1 = idx0 + H

    ntiles = (R + TR - 1) // TR

    # Grid-stride loop over row tiles
    block=asctile.block_idx()
    n=asctile.block_num()
    first=(block%2)*(n//2)+block//2
    for t in asctile.range(first, ntiles, n, unroll_factor=U):
        r0 = t * TR
        valid = min(TR, R - r0)

        # Load one contiguous slab (gate+value), bounded by valid row count
        slab = asctile.copy_in(gx, [2 * r0 * H], [2 * TR * H],
                               real_shape=[2 * valid * H], pad_value=0.0)

        # Gather gate / value elements via the cached indices
        g = asctile.reindex(slab,idx0).to(asctile.float32)
        denom = asctile.exp(-g)+1.0
        activated = g/denom
        v = asctile.reindex(slab,idx1).to(asctile.float32)
        out = (activated*v).to(slab.dtype)
        # Store at row offset, bounded by valid row count
        asctile.copy_out(out, gy, [r0 * H], real_shape=[valid * H])

@asctile.jit(always_compile=False)
def swiglu_direct_slab(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H])
    idx1 = idx0 + H

    ntiles = (R + TR - 1) // TR

    # Grid-stride loop over row tiles
    for t in asctile.range(asctile.block_idx(), ntiles, asctile.block_num(), unroll_factor=U):
        r0 = t * TR
        valid = min(TR, R - r0)

        # Load one contiguous slab (gate+value), bounded by valid row count
        slab = asctile.copy_in(gx, [2 * r0 * H], [2 * TR * H],
                               real_shape=[2 * valid * H], pad_value=0.0)

        # Gather gate / value elements via the cached indices
        g = asctile.reindex(slab,idx0).to(asctile.float32)
        denom = asctile.exp(-g)+1.0
        activated = g/denom
        v = asctile.reindex(slab,idx1).to(asctile.float32)
        out = (activated*v).to(slab.dtype)
        # Store at row offset, bounded by valid row count
        asctile.copy_out(out, gy, [r0 * H], real_shape=[valid * H])

@asctile.jit(always_compile=False)
def swiglu_inline_tail(x: asctile.GlobalAddress, y: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], T: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    ncol = (H + T - 1) // T
    total = R * ncol

    nb = asctile.block_num()
    bi = asctile.block_idx()
    q = total // nb
    r = total - q * nb
    first = q * bi + min(bi, r)
    end = q * (bi + 1) + min(bi + 1, r)
    for t in asctile.range(first, end, unroll_factor=U):
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

@asctile.jit(always_compile=False)
def swiglu_late_slab(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, i1: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])
    gi1 = asctile.global_tensor(i1, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H]).to(asctile.int32)
    idx1 = idx0 + H

    ntiles = (R + TR - 1) // TR

    # Grid-stride loop over row tiles
    for t in asctile.range(asctile.block_idx(), ntiles, asctile.block_num(), unroll_factor=U):
        r0 = t * TR
        valid = min(TR, R - r0)

        # Load one contiguous slab (gate+value), bounded by valid row count
        slab = asctile.copy_in(gx, [2 * r0 * H], [2 * TR * H],
                               real_shape=[2 * valid * H], pad_value=0.0)

        # Gather gate / value elements via the cached indices
        g = asctile.reindex(slab,idx0).to(asctile.float32)
        denom = asctile.exp(-g)+1.0
        activated = g/denom
        v = asctile.reindex(slab,idx1).to(asctile.float32)
        out = (activated*v).to(slab.dtype)
        # Store at row offset, bounded by valid row count
        asctile.copy_out(out, gy, [r0 * H], real_shape=[valid * H])

@asctile.jit(always_compile=False)
def swiglu_static_slab(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, i1: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int], N: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])
    gi1 = asctile.global_tensor(i1, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H]).to(asctile.int32)
    idx1 = idx0 + H

    ntiles = (R + TR - 1) // TR

    K = (ntiles + N - 1) // N
    block = asctile.block_idx()
    for step in asctile.range(K,unroll_factor=U):
        t = step*N+block
        r0 = min(t*TR,R-1)
        valid = max(0,min(TR,R-t*TR))
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

@asctile.jit(always_compile=False)
def swiglu_short_rows(x: asctile.GlobalAddress,
                y: asctile.GlobalAddress,
                R: asctile.ConstExpr[int],
                H: asctile.ConstExpr[int],
                TR: asctile.ConstExpr[int],
                HC: asctile.ConstExpr[int],
                U: asctile.ConstExpr[int], N: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x, [R, 2 * H])
    y_gm = asctile.global_tensor(y, [R, H])

    block = asctile.block_idx()
    whole_waves = R // (TR*N)
    if whole_waves > 0:
        for step in asctile.range(whole_waves,unroll_factor=U):
            r0 = (step*N + block)*TR
            gate = asctile.copy_in(x_gm, [r0, 0], [TR, HC],
                                   real_shape=[TR, H], pad_value=0.0)
            val = asctile.copy_in(x_gm, [r0, H + 0], [TR, HC],
                                  real_shape=[TR, H], pad_value=0.0)

            gatef = asctile.cast(gate, asctile.float32)
            valf = asctile.cast(val, asctile.float32)

            # Direct division: build the denominator from the gate, divide the
            # gate by it, multiply by the value and cast once.  This avoids the
            # extra live FP32 tile of the reciprocal-then-multiply form; the
            # rounding order changes so the golden must be compared with the
            # task's tolerance.
            d = asctile.add(asctile.exp(asctile.mul(gatef, -1.0)), 1.0)
            res = asctile.mul(asctile.div(gatef, d), valf)

            out_t = res.to(gate.dtype)
            asctile.copy_out(out_t, y_gm, [r0, 0],
                             real_shape=[TR, H])
    if R % (TR*N):
        rr = (whole_waves*N + block)*TR
        if rr < R:
            gate = asctile.copy_in(x_gm, [rr, 0], [TR, HC],
                                   real_shape=[min(TR,R-rr), H], pad_value=0.0)
            val = asctile.copy_in(x_gm, [rr, H + 0], [TR, HC],
                                  real_shape=[min(TR,R-rr), H], pad_value=0.0)

            gatef = asctile.cast(gate, asctile.float32)
            valf = asctile.cast(val, asctile.float32)

            # Direct division: build the denominator from the gate, divide the
            # gate by it, multiply by the value and cast once.  This avoids the
            # extra live FP32 tile of the reciprocal-then-multiply form; the
            # rounding order changes so the golden must be compared with the
            # task's tolerance.
            d = asctile.add(asctile.exp(asctile.mul(gatef, -1.0)), 1.0)
            res = asctile.mul(asctile.div(gatef, d), valf)

            out_t = res.to(gate.dtype)
            asctile.copy_out(out_t, y_gm, [rr, 0],
                             real_shape=[min(TR,R-rr), H])

@asctile.jit(always_compile=False)
def swiglu_compact_slab(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, i1: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])
    gi1 = asctile.global_tensor(i1, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H])
    idx1 = idx0 + H

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

@asctile.jit(always_compile=False)
def swiglu_striped(x: asctile.GlobalAddress, y: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], T: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
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

@asctile.jit(always_compile=False)
def swiglu_rows(x: asctile.GlobalAddress,
                y: asctile.GlobalAddress,
                R: asctile.ConstExpr[int],
                H: asctile.ConstExpr[int],
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

        # Direct division: build the denominator from the gate, divide the
        # gate by it, multiply by the value and cast once.  This avoids the
        # extra live FP32 tile of the reciprocal-then-multiply form; the
        # rounding order changes so the golden must be compared with the
        # task's tolerance.
        d = asctile.add(asctile.exp(asctile.mul(gatef, -1.0)), 1.0)
        res = asctile.mul(asctile.div(gatef, d), valf)

        out_t = res.to(gate.dtype)
        asctile.copy_out(out_t, y_gm, [r0, c0],
                         real_shape=[r_valid, c_valid])
from asc.experimental import asctile


@asctile.jit(always_compile=False)
def swiglu_slab(x: asctile.GlobalAddress, y: asctile.GlobalAddress, i0: asctile.GlobalAddress, i1: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], TR: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    # Rank-1 global tensors
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    gi0 = asctile.global_tensor(i0, [TR * H])
    gi1 = asctile.global_tensor(i1, [TR * H])

    # Copy each cached int32 index tensor ONCE, full length, no padding
    idx0 = asctile.copy_in(gi0, [0], [TR * H]).to(asctile.int32)
    idx1 = idx0 + H

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
def swiglu_flat(x: asctile.GlobalAddress, y: asctile.GlobalAddress, R: asctile.ConstExpr[int], H: asctile.ConstExpr[int], T: asctile.ConstExpr[int], U: asctile.ConstExpr[int]):
    gx = asctile.global_tensor(x, [2 * R * H])
    gy = asctile.global_tensor(y, [R * H])
    whole = H // T
    nfull = R * whole
    blocks = asctile.block_num()
    block = asctile.block_idx()
    q = nfull // blocks
    rem = nfull % blocks
    begin = q*block + min(block,rem)
    end = q*(block+1) + min(block+1,rem)
    for t in asctile.range(begin,end,unroll_factor=U):
        row = t // max(1,whole)
        col = (t % max(1,whole))*T
        gate_tile = asctile.copy_in(
            gx, [row * 2 * H + col], [T]
        )
        value_tile = asctile.copy_in(
            gx, [row * 2 * H + H + col], [T]
        )


        gate_f = gate_tile.to(asctile.float32)
        value_f = value_tile.to(asctile.float32)

        sig = gate_f / (1.0 + asctile.exp(-gate_f))
        out_f = sig * value_f

        out_dtype = gate_tile.dtype
        out_tile = out_f.to(out_dtype)

        asctile.copy_out(out_tile, gy, [row * H + col])
    if H % T:
        for row in asctile.range(block, R, blocks):
            col = whole*T
            valid = H % T
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

def row_geometry(rows, half, itemsize, ub_bytes, aivs, is_bf16):
    """Pick (height, unroll) for the public 950PR strided SwiGLU 2D row copy.

    The kernel reads two half-rows of `half` contiguous elements of
    `itemsize` bytes with strided copies and writes one half-row
    (reuse_alloc=2), widening 16-bit inputs to FP32 and using the direct
    division form (gate / sigmoid(gate)) * value.  `ub_bytes` is the
    single-iteration UB budget with any fixed workspace already subtracted.
    `is_bf16` distinguishes BF16 from FP16 for the per-lane coefficients.
    Raises ValueError when a single row cannot fit the iteration budget
    (caller must use its own UB-qualified column-chunk fallback); small
    gathers use the separate fallback and never reach here.
    """
    if rows < 1 or half < 1 or itemsize < 1 or ub_bytes < 1 or aivs < 1:
        raise ValueError("row_geometry expects positive dimensions")
    if not isinstance(is_bf16, bool):
        raise ValueError("row_geometry expects is_bf16 to be a bool")

    # Useful payload of one half-row (one input descriptor).  The copy's
    # real shape still uses `half` elements; only allocation checks see
    # the padded physical width.
    payload = half * itemsize

    # Physical width W: `half` rounded up to a multiple of 32/s elements so
    # the physical row width is 32-byte aligned.
    if itemsize <= 32 and 32 % itemsize == 0:
        quantum = 32 // itemsize
    else:
        quantum = 1
    width = ((half + quantum - 1) // quantum) * quantum

    # Per-iteration UB cost: bytes per physical lane, lane count being
    # width * height.  reuse_alloc=2 budgets: 16 B/lane at unroll 1.
    # At unroll 2 the direct-division FP16 row kernel measured 20 B/lane
    # on the archive-bundled 950PR public build (163,840 bytes for 8,192
    # lanes); the reciprocal form used 24.  FP16 uses the qualified 20
    # B/lane.  FP32 keeps the 28 B/lane upper bound and BF16 the
    # conservative 32 B/lane until their direct-division emitted forms
    # have been qualified; verify the compiled allocation on each new
    # kernel/build before reusing a smaller coefficient for dispatch.
    lane_unroll1 = 16
    if itemsize >= 4:
        lane_unroll2 = 28
    elif is_bf16:
        lane_unroll2 = 32
    else:
        lane_unroll2 = 20

    def footprint(height, unroll):
        per_lane = lane_unroll1 if unroll == 1 else lane_unroll2
        return per_lane * width * height

    # Smallest power-of-two height giving at least 8 KiB of useful
    # half-tile payload; a whole row is the minimum.  When a row is
    # smaller than 8 KiB this still yields < 16 KiB of useful payload.
    t = 1
    while t * payload < 8192:
        t *= 2

    # Halve T until it fits the row count and the single-iteration UB
    # budget at unroll 1.
    while t > 1 and (t > rows or footprint(t, 1) > ub_bytes):
        t //= 2

    if footprint(t, 1) > ub_bytes:
        # Even one row overflows the iteration budget; the row kernel
        # cannot fit.  The column-chunk fallback must qualify its own
        # chunk size against the UB.
        raise ValueError(
            "swiglu row kernel cannot fit: one row needs %d UB bytes for a "
            "single iteration but only %d are available"
            % (footprint(1, 1), ub_bytes)
        )

    # At this height, take unroll 2 only if its footprint fits and the
    # grid supplies at least two tiles per active AIV on average
    # (active AIVs = min(available, ceil(rows/T))); the grid-stride loop
    # owns the final partial tile.  Otherwise keep unroll 1.
    tiles = -(-rows // t)
    active = min(aivs, tiles)
    if footprint(t, 2) <= ub_bytes and tiles >= 2 * active:
        return t, 2
    return t, 1

def previous_select(rows, half, itemsize, is_pair, is_bf16, ub_bytes, aivs):
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

    if half_bytes <= 3072:
        tile, unroll = row_geometry(rows, half, itemsize, ub_bytes, aivs, is_bf16)
        return ("batched_rows", tile, unroll)

    # Long contiguous half-row: flat chunk stream, gate/value offsets H apart.
    # Flat unroll 2 fits 10560 elements max on this pin. Tile is an element
    # count, aligned down to 32 bytes (physical alignment applied by caller).
    chunk_elems = ((half_bytes + 31) // 32) * 32 // itemsize
    chunk = align32_down_elems(min(10560, chunk_elems))
    return ("flat_chunks", chunk, 2)

def fp16_wave_height(rows, half, ub_bytes, aivs, base_tile, base_unroll):
    # Existing height is returned whenever the rule does not apply.
    if rows <= 0 or half <= 0 or ub_bytes <= 0 or aivs <= 0 or base_tile <= 0:
        return base_tile
    if base_unroll != 2:
        return base_tile

    # FP16 element is 2 bytes; round the half-row width up to a 32-byte
    # DMA quantum (16 FP16 elements per 32 bytes).
    QUANTUM = 16
    aligned_half = ((half + QUANTUM - 1) // QUANTUM) * QUANTUM
    half_bytes = aligned_half * 2

    # Largest power-of-two grid no larger than the available AIVs.
    pow2_grid = 1
    while pow2_grid * 2 <= aivs:
        pow2_grid *= 2

    # Already an exact multiple of base_tile * pow2_grid: keep existing height.
    unit = base_tile * pow2_grid
    if half in (aligned_half,) and unit > 0 and rows % unit == 0:
        return base_tile

    # Rows per tile bounded by a 12 KiB half-tile DMA budget.
    DMA_BUDGET = 12 * 1024
    dma_rows = DMA_BUDGET // half_bytes

    # Rows per tile bounded by 20 bytes of UB per physical element.
    ub_elems = ub_bytes // 20
    ub_rows = ub_elems // aligned_half if aligned_half > 0 else 0

    capacity = dma_rows if dma_rows < ub_rows else ub_rows

    # Bounds below two rows: keep the old geometry.
    if capacity < 2:
        return base_tile

    # Inputs shorter than two complete waves at this bound: keep old geometry.
    if rows < 2 * aivs * capacity:
        return base_tile

    # Fewest full-grid waves covering the rows at this capacity.
    waves = (rows + aivs * capacity - 1) // (aivs * capacity)

    # Divide rows among the waves, rounding the new height up.
    new_height = (rows + waves * aivs - 1) // (waves * aivs)

    # Return only if it fits both bounds (it cannot exceed capacity by
    # construction, but guard explicitly).
    if new_height <= dma_rows and new_height <= ub_rows:
        return new_height
    return base_tile


def choose_even_grid(groups, capacity):
    if groups <= 0 or capacity <= 0:
        return 0
    available = groups if groups < capacity else capacity
    waves = groups // available
    if waves < 4:
        return available
    low = (7 * available + 7) // 8
    candidate = available - (available % 2)
    while candidate >= low:
        if candidate > 0 and groups % candidate == 0:
            return candidate
        candidate -= 2
    return available


def paired_bf16_width(rows, half, tile, aivs, ub):
    if min(tile, aivs, ub) <= 0 or not 1 < rows < aivs:
        return tile
    if not aivs * tile < half <= 2 * aivs * tile:
        return tile
    chunks = 2 * aivs
    width = ((half + chunks * 16 - 1) // (chunks * 16)) * 16
    if width * 20 <= ub and (half + width - 1) // width == chunks:
        return width
    return tile


def whole_row_grid(rows, half, tile, itemsize, cap, ub_bytes):
    if itemsize != 4 or half != tile or cap < 1 or half % 8:
        return 0
    if not 8192 <= half * itemsize <= 16384 or not 4 * cap <= rows < 8 * cap:
        return 0
    if 2 * rows * half * itemsize > 64 * 1024**2 or 48 * tile > ub_bytes:
        return 0
    grid = choose_even_grid(rows, cap)
    return grid if rows % (4 * grid) == 0 else 0


def select(rows, half, itemsize, is_pair, is_bf16, ub_bytes, aivs):
    route, t, u = previous_select(rows, half, itemsize, is_pair, is_bf16, ub_bytes, aivs)
    size = half * itemsize
    if route=='index_gather' and is_bf16 and half==1:
        # A DMA-sized pair slab can cover this input in four compact waves.
        grid = 1 << (aivs.bit_length() - 1)
        count = (rows + t - 1) // t
        if u == 2 and 3 * grid < count <= 4 * grid and 28*t <= ub_bytes:
            return route, t, u
        quantum=32//itemsize
        waves=4
        h=((rows+waves*aivs-1)//(waves*aivs)+quantum-1)//quantum*quantum
        if t//2<=h<=t and h*28<=ub_bytes and 2*h<=32768 and rows>=2*h*aivs:
            return route,h,2
    if route == 'index_gather' and is_bf16 and half > 1:
        quantum = 32//itemsize
        wave = (rows+2*aivs-1)//(2*aivs)
        wave = (wave+quantum-1)//quantum*quantum
        if wave and wave*half*28 <= ub_bytes and wave <= 2*t:
            return route,wave,2
    if route == 'batched_rows' and itemsize == 2 and not is_bf16 and half % 16 and u == 2 and t >= 2:
        w = (half + 15) // 16 * 16
        small = t//2
        if 20*w*small <= ub_bytes and rows >= 2*aivs*small:
            return route, fp16_wave_height(rows, half, ub_bytes, aivs, small, 2), 2
    if route == 'batched_rows' and itemsize == 4 and half < 64 and half % 8 and u == 2:
        if 32*t*half <= ub_bytes and 2*t*half <= 32768 and t*half % 16 == 0:
            return 'compact_slab',t,u
    if route != 'flat_chunks':
        return route,t,u
    if whole_row_grid(rows, half, t, itemsize, aivs, ub_bytes):
        return 'whole_rows', t, 4
    if itemsize == 4 and 8192 <= size <= 16384 and rows >= 8*aivs:
        h = max(1,16384//size)
        if h*((half+7)//8*8)*36 <= ub_bytes:
            return 'batched_rows',h,3
    if size <= 32768 and itemsize == 2 and half == t and t*32 <= ub_bytes:
        return route,t,4
    if size > 32768:
        traffic = 2*rows*size
        if traffic > 64*1024**2 and ub_bytes >= 8192*(30 if itemsize==2 else 24):
            return route,8192,(3 if itemsize==2 else 2)
        if traffic <= 64*1024**2 and ub_bytes >= 24*8192:
            if is_bf16 and 1 < rows < aivs:
                width = paired_bf16_width(rows, half, 8192, aivs, ub_bytes)
                if width != 8192:
                    return 'column_stream',width,2
            if is_bf16 and rows==1 and half>2*aivs*8192:
                waves=2*((half+2*aivs*8192-1)//(2*aivs*8192))
                chunks=waves*aivs
                lanes=((half+chunks-1)//chunks+15)//16*16
                if 24*lanes <= ub_bytes:
                    return route,lanes,2
            if rows==1 and itemsize==2 and not is_bf16:
                target=10240//itemsize
                pairs=(half + 2*aivs*target-1)//(2*aivs*target)
                n=2*aivs*max(1,pairs)
                unit=32//itemsize
                lanes=((half+n-1)//n+unit-1)//unit*unit
                if lanes*20 <= ub_bytes:
                    return route,lanes,2
            if itemsize==4 and rows==1:
                unit=32//itemsize
                wanted=max(1,(half+8192*aivs-1)//(8192*aivs))
                waves=1 << (wanted-1).bit_length()
                lanes=half//(waves*aivs)//unit*unit
                rest=half-waves*aivs*lanes
                if lanes>=unit and lanes*24<=ub_bytes and rest<=unit*aivs:
                    return route,lanes,2
                chunks=(half+8191)//8192
                chunks=(chunks+aivs-1)//aivs*aivs
                unit=32//itemsize
                t=((half+chunks-1)//chunks+unit-1)//unit*unit
                return route,t,2
            return route,min(t,8192),2
    return route,t,u

ROUTES = {"whole_rows":"flat", "column_stream":"flat", "packed_pairs":"packed", "index_gather":"reindex", "batched_rows":"2d", "flat_chunks":"flat"}
from functools import lru_cache
from math import prod

import torch
from ._pyasc_runtime import ensure_npu_platform, vector_blocks


@lru_cache(maxsize=32)
def gather_indices(half, tile, device):
    lane = torch.arange(tile * half, device=device, dtype=torch.int32)
    gate = (lane // half) * (2 * half) + lane % half
    return (gate.to(torch.int16) if half == 1 and 2*half*tile <= 32768 else gate), gate


from ._pyasc_runtime import query_aivs, query_ub_bytes
@lru_cache(maxsize=32)
def small_indices(h,t,device):
    lanes=torch.arange(t*h,device=device,dtype=torch.int32)
    g=((lanes//h)*(2*h)+lanes%h).to(torch.int16)
    return g,g

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
    route = 'compact' if layout == 'compact_slab' else ROUTES[layout]
    x, y = input.contiguous().view(-1), out.view(-1)
    groups = (rows + tile - 1) // tile
    if route == "flat":
        work = rows * ((half + tile - 1) // tile)
        blocks = vector_blocks(work)
        traffic = 2*rows*half*itemsize
        if rows == 1 and itemsize == 2 and traffic > 64*1024**2:
            blocks = 1 << (blocks.bit_length()-1)
        elif is_bf16 and half==tile and unroll>=4:
            aligned_grid=1 << (blocks.bit_length()-1)
            if work >= 8*blocks and work % aligned_grid == 0:
                blocks=aligned_grid
        if layout == 'whole_rows':
            row_blocks = whole_row_grid(rows, half, tile, itemsize, blocks, ub_bytes)
            if row_blocks:
                swiglu_whole_row[row_blocks](x,y,rows,half,unroll,row_blocks,opt_level=3,reuse_alloc=2)
                return out
            unroll = 2
        flat_fn = swiglu_flat
        if itemsize == 4 and (half == tile or (rows > aivs and half%tile)):
            flat_fn = swiglu_inline_tail
        elif traffic <= 64*1024**2 and half > tile:
            flat_fn = swiglu_inline_tail if itemsize==2 and (not is_bf16 or (rows==1 and half>2*aivs*8192)) else swiglu_striped
        if layout == "column_stream":
            flat_fn=swiglu_column_stripe
        unit=32//itemsize
        rem=half%(tile*blocks)
        if rows==1 and itemsize==4 and tile*blocks<=half and rem<=unit*blocks:
            swiglu_flat_edge[blocks](x,y,half,tile,unroll,blocks,unit,reuse_alloc=2)
        else:
            if itemsize == 4 and half == tile and rows > aivs and traffic <= 64*1024**2:
                flat_fn[blocks](x,y,rows,half,tile,unroll,opt_level=3,reuse_alloc=2)
            else:
                flat_fn[blocks](x,y,rows,half,tile,unroll,reuse_alloc=2)
    elif route == "2d":
        align = 32 // itemsize
        width = (half + align - 1) // align * align
        blocks = vector_blocks(groups)
        exact_blocks=1 << (blocks.bit_length()-1)
        if input.dtype==torch.float16 and width==half and rows%tile==0 and groups%exact_blocks==0 and groups>=4*exact_blocks and unroll>=2:
            swiglu_aligned_rows[exact_blocks](x,y,rows,half,tile,width,unroll,exact_blocks,reuse_alloc=2)
        elif itemsize == 4 and groups <= blocks:
            swiglu_short_rows[blocks](x,y,rows,half,tile,width,unroll,blocks,reuse_alloc=2)
        else:
            if itemsize==4 and width==half and 8192 <= half*itemsize <= 16384 and rows >= 8*aivs:
                blocks=choose_even_grid(groups,blocks)
                swiglu_rows[blocks](x,y,rows,half,tile,width,unroll,opt_level=3,reuse_alloc=2)
            else:
                swiglu_rows[blocks](x,y,rows,half,tile,width,unroll,reuse_alloc=2)
    elif route == 'compact':
        i0,i1 = small_indices(half,tile,x.device)
        if rows%tile==0 and vector_blocks(groups)%2==0:
            full_fn = swiglu_full_slab
            full_fn[vector_blocks(groups)](x,y,i0,rows,half,tile,unroll,reuse_alloc=2)
        else:
            swiglu_compact_slab[vector_blocks(groups)](x,y,i0,i1,rows,half,tile,unroll,reuse_alloc=2)
    elif route == "reindex":
        blocks = vector_blocks(groups)
        if is_bf16 and 2*tile*half <= 32768 and tile*half % 16 == 0:
            i0,i1=small_indices(half,tile,x.device)
            if half > 1 and groups > blocks and groups <= 2 * blocks and unroll == 2:
                wave_fn = swiglu_local_waves
                wave_fn[blocks](x,y,i0,rows,half,tile,unroll,reuse_alloc=2)
            elif half>1:
                swiglu_wave_slab[blocks](x,y,i0,i1,rows,half,tile,unroll,blocks,reuse_alloc=2)
            else:
                compact = 1 << (blocks.bit_length()-1)
                if unroll == 2 and 3*compact < groups <= 4*compact:
                    blocks = compact
                pair_fn = swiglu_permuted_pairs if blocks==compact and blocks%2==0 and 3*compact < groups <= 4*compact else swiglu_direct_slab
                if pair_fn is swiglu_permuted_pairs:
                    pair_fn[blocks](x,y,i0,rows,half,tile,unroll,opt_level=3,reuse_alloc=2)
                else:
                    pair_fn[blocks](x,y,i0,rows,half,tile,unroll,reuse_alloc=2)
        elif is_bf16 and half == 1:
            i0,i1=gather_indices(half,tile,x.device)
            swiglu_late_slab[blocks](x,y,i0,i1,rows,half,tile,unroll,reuse_alloc=2)
        elif is_bf16:
            i0,i1=gather_indices(half,tile,x.device)
            swiglu_static_slab[blocks](x,y,i0,i1,rows,half,tile,unroll,blocks,reuse_alloc=2)
        else:
            i0,i1=gather_indices(half,tile,x.device)
            swiglu_slab[blocks](x,y,i0,i1,rows,half,tile,unroll,reuse_alloc=2)
    elif route == "packed":
        _swiglu_pair_groups[vector_blocks(groups)](x, y, rows, groups, tile)
    else:
        raise ValueError(layout)
    return out
