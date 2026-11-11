# Row geometry for public reindex gathers

For a contiguous `[R, 2H]` split, one input DMA brings `2*T*H*s` bytes
into UB, where `T` counts rows and `s` is the input element size. Two cached
int32 index vectors of length `T*H` select the halves. Include both vectors,
casts, arithmetic temporaries and software-pipeline buffers in the allocation.
Read the effective UB capacity and available AIVs after platform selection.

## BF16 SwiGLU on the archive-bundled 950PR PyASC build

Use this geometry inside the short-half gather branch (`H*s <= 32` bytes).
It also covers contiguous BF16 pairs (`H=1`, normalized final axis of length
two). On this public build with cached int32 indices, let dense pairs reach
the same gather selector as other short BF16 halves. An early return for
the legacy transpose specialization would hide the eligible gather. Reuse
the existing gather selector and kernel. Keep the existing bounded row
fallback when alignment or capacity declines a tile. Preserve the established
2D and flat routes for wider halves.

With `reuse_alloc=2`, aligned BF16 half-rows of one and seven elements used
24 bytes per output lane at unroll 1 and 28 at unroll 2. These are compiled live-buffer costs. Recheck allocations
when applying this estimate to a different specialization or compiler.
Each row has `H` output lanes. For `T` rows the two footprints in bytes are
`24*T*H` and `28*T*H`. The corresponding costs per row are `24*H` and
`28*H`. Subtract any fixed workspace from available UB before division.

For a positive row count and active vector count `A`, compute the candidate
row bound as `min(16384 // (2*H*s), available_ub // (24*H), ceil(R/A))`.
Return the caller's existing public gather route tag for either unroll.
Test dispatch at a smaller UB that fits only one iteration and at another
that forces a shorter tile. A single full-capacity test can hide a unit error.

1. Target up to 16 KiB per contiguous input DMA on this build. Start with the
   largest power-of-two row count within all three bounds. Use the DMA target
   divided by input bytes per row, the effective UB divided by the single-
   iteration cost per row, and `ceil(R / active_AIVs)` useful rows per AIV.
   Keep the physical input, gathered output and int32 index tile lengths
   aligned to 32 bytes. Check the actual byte lengths before returning either
   gather specialization: `(2*T*H*s) % 32 == (T*H*s) % 32 == (4*T*H) % 32 == 0`.
   If the largest power-of-two tile fails, all smaller powers also fail.
   Use the row fallback in that case. Allow the final valid row count to be smaller.
   Exercise the fallback with `R=A` and a seven-element BF16 half-row:
   the one-row candidate has only 14 output bytes.
2. At this height prefer unroll 2 when the two-iteration footprint fits UB
   and there are at least two tiles per active block on average. Otherwise
   try unroll 1 at that height. If no aligned tile fits, use the existing
   bounded row fallback. Grid size is `min(active_AIVs, ceil(R/T))`.
3. A device with more rows of work per block can support taller tiles. A cap
   proportional to the *number of cores* confuses block count with tile
   height. Compare compiled UB, actual launches, warmed device medians and
   the full golden before promoting a specialization.

Account for any fixed workspace or alignment overhead reported by the
compiler. For other dtypes, kernels and runtimes, supply their own compiled
footprints and measure the suitable DMA target and unroll.
