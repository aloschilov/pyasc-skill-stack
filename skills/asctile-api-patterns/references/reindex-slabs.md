# Row geometry for public reindex gathers

For a contiguous `[R, 2H]` split, one input DMA brings `2*T*H*s` bytes
into UB, where `T` counts rows and `s` is the input element size. The general BF16 gather uses a cached int32 gate vector of length `T*H` and derives the value indices in UB. Dense BF16
pairs can use one signed-int16 vector as described below. Include live indices,
casts, arithmetic temporaries and software-pipeline buffers in the allocation.
Read the effective UB capacity and available AIVs after platform selection.

## BF16 SwiGLU on the archive-bundled 950PR PyASC build

Use this geometry inside the short-half gather branch (`H*s <= 32` bytes).
It also covers contiguous BF16 pairs (`H=1`, normalized final axis of length
two). On this public build, let dense pairs reach
the same gather selector as other short BF16 halves. An early return for
the legacy transpose specialization would hide the eligible gather. For a new module, define a public slab kernel and call it from the
host selector. View input and output as rank-one descriptors. Cache the
input-independent gate table by dtype of the indices, device, H and T.
Pass its pointer as the third tensor argument and load it once per block
before the tile loop. Use `ig = asctile.global_tensor(gate_idx_ptr, [T*H])`,
`gate_idx_ub = asctile.copy_in(ig, [0], [T*H])`, and
`value_idx_ub = gate_idx_ub + H`. For a slab starting at row r, copy
`2*T*H` input lanes at offset `2*r*H`, with true extent
`2*min(T,R-r)*H` and zero padding. Use `asctile.reindex(slab, gate_idx_ub)`
and `asctile.reindex(slab, value_idx_ub)` before widening each gathered
half. Store the resulting contiguous output at `r*H`, with true extent
`min(T,R-r)*H`. The final partial tile belongs to the same launch.

For ordinary dense pairs (`H == 1`) on the 950PR wheel, select a single
signed-int16 gate table when `2*T-1 <= 32767` and `2*T` index bytes
are DMA-aligned. Construct `2*k` in int32 before converting to int16 on
the caller's device. Cache only that table. Pass input, output and the
gate table as the three tensor arguments. Load the entire gate table once
per block before the tile loop and compute `value_idx = gate_idx + 1` in UB.
In a JIT shared with other half-widths write `value_idx = gate_idx + H`.
Adding one is valid inside a dedicated H=1 specialization. Exercise the ordinary
H=2 route with 2048 rows of [1, 2, 4, 8], whose value lanes are [4, 8].
Pass the int16 indices directly to public reindex. Gather and widen the gate,
compute its activation, then gather and widen the value and multiply. Keep
the grid, bounded DMA extents and existing UB admission bounds. If the
index bound or alignment fails, use the general int32 path. Check emitted
allocation when changing tile geometry or compiler. At T4096/U2/N72,
two matched runs took 6.301 and 6.273 µs with this kernel. Loading both
int32 tables at the same geometry took 6.789 and 6.851 µs.

Once the ordinary selector has admitted a positive tile, build one table
with the appropriate index dtype. Dispatch both dtypes to the ordinary
slab JIT. On the 72-AIV fixture `R=2048, H=2`, the ordinary selector
returns T16 and the launch takes an int32 gate vector on 72 blocks. The
first four indices are [0, 1, 4, 5] and their value indices are [2, 3, 6, 7].

```python
def dense_pair_gate_index(half, tile, device, cache):
    if half != 1 or tile < 1 or 2 * tile - 1 > 32767 or (2 * tile) % 32:
        return None
    key = ("dense_pair_i16", device, tile)
    if key not in cache:
        k = torch.arange(tile, device=device, dtype=torch.int32)
        cache[key] = (2 * k).to(torch.int16)
    return cache[key]


def ordinary_gate_index(half, tile, device, cache):
    gate_index = dense_pair_gate_index(half, tile, device, cache)
    if gate_index is not None:
        return gate_index
    key = ("general_i32", device, half, tile)
    if key not in cache:
        k = torch.arange(tile * half, dtype=torch.int32, device=device)
        cache[key] = 2 * half * (k // half) + k % half
    return cache[key]
```

Declare the device function itself with `@asctile.jit(reuse_alloc=2)`. The
selector below is paired with this allocation policy. Merely writing
`@asctile.jit(always_compile=False)` does not select allocation reuse.
Check the emitted decorator on the slab as well as the row kernel.
With `reuse_alloc=2`, aligned BF16 half-rows of one and seven elements used
24 bytes per output lane at unroll 1 and 28 at unroll 2. These are compiled live-buffer costs. Recheck allocations
when applying this estimate to a different specialization or compiler.
Each row has `H` output lanes. For `T` rows the two footprints in bytes are
`24*T*H` and `28*T*H`. The corresponding costs per row are `24*H` and
`28*H`. Subtract any fixed workspace from available UB before division.

For a positive row count and active vector count `A`, compute the candidate
row bound as `min(16384 // (2*H*s), available_ub // (24*H), ceil(R/A))`.
Use a single division of bytes by bytes per row. For H=1 and s=2,
one physical input row is four bytes; 16,384 bytes contain 4,096
complete pairs. In the ordinary int32-index slab selector use:

```python
def ordinary_bf16_slab_tile(rows, half, aivs, usable_ub):
    if min(rows, half, aivs, usable_ub) <= 0 or 2 * half > 32:
        return 0
    bytes_per_input_row = 2 * half * 2
    by_dma = 16384 // bytes_per_input_row
    bound = min(by_dma, usable_ub // (24 * half), (rows + aivs - 1) // aivs)
    if bound < 1:
        return 0
    tile = 1 << (bound.bit_length() - 1)
    lanes = tile * half
    return tile if (2 * lanes) % 32 == 0 else 0

def ordinary_slab_launch(rows, half, tile, aivs, usable_ub):
    """Size a gather launch after selecting a positive tile in rows."""
    tiles = (rows + tile - 1) // tile
    blocks = min(aivs, tiles)
    two_stage_bytes = 28 * tile * half
    unroll = 2 if tiles >= 2 * blocks and two_stage_bytes <= usable_ub else 1
    if half == 1 and unroll == 2 and 3 * blocks < tiles <= 4 * blocks:
        compact = 2 * ((tiles + 7) // 8)
        if compact <= blocks:
            blocks = compact
    return blocks, unroll
```

For BF16, alignment of `2*lanes` output bytes implies alignment of
the input and int32 table. At the selected height, two stages require
`28*lanes <= usable_ub` and at least two tiles per active block.
Here each tile owns up to T whole rows, including H=1. Count work with
`ceil(R/T)`. The flat column-stream count `R*(H//T)` gives zero for a
4096-row pair slab. Use `ordinary_slab_launch` at the actual gather call.
Trace both bytes and rows: for 1,000,003 pairs, 72 AIVs and 253,952
bytes the ordinary path reads 16,384 bytes at T4096/U2. For two-stage dense
pairs with more than three and at most four waves at the detected grid,
choose the smallest even grid covering the work in four waves. Use identity
block order and a block-stride loop. Block zero owns tile ordinals
`0, blocks, 2*blocks, ...`, block one starts at one. Write this in
the ordinary slab JIT itself:

```python
for t in asctile.range(asctile.block_idx(), count, asctile.block_num(),
                       unroll_factor=unroll):
    first = t * tile
    valid = min(tile, rows - first) * half
    # Read the packed input and write only the valid outputs.
```

The two-wave multi-lane JIT has a separate consecutive interval rule.
For 245 tiles the ordinary dense-pair rule yields
62 blocks. In three paired rounds the 62-block medians were 6.448, 6.462
and 6.800 µs, against 7.142, 6.867 and 6.817 µs on 72 blocks. On a
separate finite input of 911117 pairs, the same formula gave 56 blocks
for 223 tiles and won in all three rounds. A two-group permutation at
either grid did not improve the median across rounds on either shape. Dividing by
the number of halves a second time gave T2048 and an 8,192-byte read.

Return the caller's existing public gather route tag for either unroll.
Carry the returned unroll across the JIT boundary as a
`unroll: asctile.ConstExpr[int]` argument. In the row-slab loop set
`unroll_factor=unroll`. Verify this on the emitted loop and on its launch:
a fixed `unroll_factor=1` silently discarded the host's two-stage decision.
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
   bounded row fallback. Start with `min(active_AIVs, ceil(R/T))` blocks.
   Apply the qualified compact-grid rule below for eligible dense BF16 pairs.
3. A device with more rows of work per block can support taller tiles. A cap
   proportional to the *number of cores* confuses block count with tile
   height. Compare compiled UB, actual launches, warmed device medians and
   the full golden before promoting a specialization.

Account for any fixed workspace or alignment overhead reported by the
compiler. For other dtypes, kernels and runtimes, supply their own compiled
footprints and measure the suitable DMA target and unroll.

## Reducing the index reads

For a contiguous split, gate lane `k` indexes
`2*H*(k//H) + k%H`. The corresponding value is exactly `H` elements
later. Cache one gate-index vector per device, logical half-width and tile
height. Load it once per block before the tile loop and derive the value
indices with one integer add. Compare this with loading both int32
vectors. The input slab and output store keep their original dtype.

For dense BF16 pairs, try storing the gate indices as signed int16 when
the greatest index used for either half, `2*T*H-1`, is at most 32767.
Build the index in int32 first. Widen the cached vector to int32 in UB
before the add and both public `reindex` calls. Include this conversion and
the derived indices in the compiled UB check. Account for index DMA
alignment at the actual storage width. For a larger slab retain int32
storage. On Ascend950PR_9599 a 4,096-row tile with two pipeline stages
passed the full golden, including the 579-row tail of 1,000,003 BF16
pairs. Compare against the two-vector version in the same timing session.

The public front end of this wheel also accepts signed int16 indices
directly. Qualify that lowering with the actual source dtype. For
unaligned FP32 half-rows compare dense slab gather plus one contiguous
store with two strided reads plus a strided store. A tile of 64 rows with
50 FP32 elements per half-row, unroll 2, a cached int16 gate vector and a locally
derived value vector used 102,400 UB bytes, or 32 bytes per output lane,
and passed the full golden. For BF16, compare direct int16 reindex against widening the same cached
indices to int32. Keep the BF16 slab in its original dtype. Gather and
widen the gate, compute its activation, then gather and widen the value.
In a randomized three-round comparison on 1,000,003 pairs, the 4,096-row
direct-int16 variant had run medians of 6.150, 6.061 and 5.998 µs.
Casting the entire slab to FP32 before the two gathers at that same tile
gave 7.870, 7.022 and 7.092 µs. Each timed kernel passed the full golden.
Check the index range and index-DMA alignment for both halves before
applying this ordering to another short-row layout. Building full vectors by broadcasting a short pattern
and per-group offsets increased the device median for the tested
4,096-row BF16 pair tile.

## Two-wave BF16 half-rows

Use the default cached descriptor for a gate-index table shared by all
blocks. Each AIV reads the same small table; streaming input descriptors
and index-table descriptors have different reuse. At 928 rows per tile,
seven BF16 values per half and 72 AIVs, the default descriptors took
5.967, 6.560 and 6.016 µs in three randomized rounds on 950PR. Bypassing
only the index table took 15.983, 15.957 and 16.151 µs. Bypassing only
the streamed input took 7.103, 6.653 and 7.197 µs. All twelve runs,
including the both-bypass control, used the same addresses within each
tile, arithmetic, grid and full golden check.

In the generated host wrapper call the
[two-wave selector and kernel](../templates/two-wave-swiglu-slab.py)
before the generic DMA-limited slab. When the selector returns a positive
height, launch this kernel with the queried AIV count. Cache one full gate
index on the caller's device, keyed by device, half-width and height:
for `k = torch.arange(tile*half, device=x.device, dtype=torch.int32)`,
store `(k + half*(k//half)).to(torch.int16)`. Pass that tensor as the
third kernel argument. A zero return falls through to ordinary selection.

For resident Torch NPU input, the framework constructs the index on the
caller's current stream. Launch this two-wave gather on that stream using
the public wheel's pointer-width handle and the same block count. Scope
the device index cache to its producing stream or establish an event
dependency before reusing it across streams. CPU-staged inputs retain
the grid-only launch. Query the stream on every invocation.

On the 950PR, at T928/H7/N72, a matched restored-input comparison took
6.63 and 5.95 µs on the caller's stream, versus 7.82 and 7.62 µs with
a grid-only launch. Kernel source, grid, index and bounded tail were
identical. The analogous FP16 identity Exp comparison was mixed (3.70
and 4.05 µs on the caller's stream). Qualify other launch families
independently before predicting a timing gain.

On the qualified 950PR, compare the DMA-sized slab with two adjacent
tiles per AIV for BF16 half-rows wider than one element. The latter is
a separate candidate with a larger DMA. Compute
`T = align_up(ceil_div(R, 2*A), 16)` in rows. Select the two-wave
schedule when the resulting tile count is exactly `2*A`, all index
and DMA extents align and `2*T*H-1 <= 32767`. Use the cached full-length
signed-int16 gate index, derive the value index in UB, and use the
bounded contiguous interval per block described below, with unroll two.
Check the UB for this emitted kernel. At 928 by 7 its compiled requirement
was 142,912 bytes. Recheck the compiler allocation for other geometries
or index lowerings. Fall back to the conservative slab when it cannot
fit. Keep the single-lane pair selector separate.

For short BF16 half-rows wider than one element, try mapping two complete
waves onto the runtime AIV count. Round the proposed row height up to the
32-byte DMA quantum and recheck the compiled UB footprint, the signed index
limit and the actual number of tiles. Pass the actual launch count N to
the kernel whenever it participates in compile-time loop bounds. With
F = R // T complete tiles, each block handles F // N full tiles in a fixed
loop; blocks below F % N handle one more full tile. If R % T is nonzero,
block F % N alone processes the final partial tile using the true row count.
Unmasked fixed-size copies are valid on each complete tile. This schedule
also covers fewer than N tiles and a partial tile immediately after an exact
number of waves. On the 72-AIV 950PR, 133,221 BF16 rows of half-width seven
used a height of 928 and unroll two. Two independent control groups returned
6.003 and 6.211 µs with complete-output checks.

With between one and two tiles per AIV, compare the grid-stride schedule
with a contiguous interval per block at the same tile, unroll, and grid.
Divide the tile count M by the active block count N. For quotient q and
remainder r, block b starts at `b*q + min(b,r)` and owns q tiles plus one
when b < r. Use the valid row extent on the last load and store. For
133,221 BF16 rows with half-width seven, M=144 and N=72, so each AIV
processes two adjacent 928-row tiles. With a cached int16 index and unroll
two, the contiguous kernel used 142,912 bytes of UB and measured 6.089
and 6.018 µs in independent hardware controls. The existing grid-stride
kernel used 116,928 bytes. Its separate
full-wave loop and bounded tail measured 6.830 and 6.863 µs. The kernels also
differ in how they bound full tiles. Benchmark the entire kernel when
choosing the schedule for another shape.

When tuning a multi-lane BF16 slab near two waves, vary the launch count
at a fixed tile height before attributing a gain to rounding the tile.
For a proposed larger tile with M tiles and A available AIVs, when
A < M <= 2*A, also measure the even grid N = min(A, 2*ceil(M/4)).
Verify the compiled UB footprint, the signed index bound, and complete
output ownership by the existing contiguous-interval loop. On 950PR,
131 tiles of 1,024 rows and seven values per half took
6.282/6.103/6.783 µs at 66 blocks and 7.068/6.988/6.895 µs at 72,
with byte-identical emitted C++. The ordinary 928-row tile at 72 blocks
took 6.684/6.532/6.729 µs. At half-widths five and six the compact
1,024-row candidate beat the ordinary tile in three of four paired
rounds. Treat it as an alternative to qualify on the target. Use the
ordinary two-wave selector as the default until a full independent
generation confirms a replacement.

## Dense FP32 slabs

For dense FP32 slabs with no partial row tile, time an alternate first
tile per block while leaving the UB layout, block count and loop stride
fixed. On an even grid of N blocks, physical block b can start at
`(b % 2) * (N // 2) + b // 2` and advance by N tiles. The even block
IDs cover the first half of the starting offsets and the odd IDs cover
the second half exactly once. Keep the identity ordering available for
odd grids. A 64-row, two-stage tile with 100 FP32 inputs and 50 outputs per row took
15.589, 15.441 and 15.619 µs in separate control groups on the 72-AIV
950PR. The integrated candidate measured 15.500 µs. Qualify the
permutation independently for other layouts and row heights.

For a dense BF16 pair stream, include the original DMA-sized tile in
this comparison. Compute its row count from the input bytes, the
compiled UB footprint and useful work per AIV. When it yields just over
three and at most four waves on the largest power-of-two grid within the
device limit, time that grid with identity and even/odd block order.
The earlier 64-block control used even/odd ordering for two-stage dense
BF16 pairs with a bounded signed-int16 index. Follow the ordinary selector
above when its four-wave window fits the queried AIV count. Use identity order on a single block.
Keep the true input and output extents on the final tile. At 1,000,003
pairs, 4,096 rows per tile give 245 tiles. A 64-block launch has 53
blocks with four tiles and 11 with three. The interleaved order took
6.075 and 6.017 µs in two randomized fresh-process controls, versus
6.088 and 6.481 µs with identity order at identical geometry. The
device exposes 72 AIVs. The measured 64 is a launch choice specific to
this occupancy range.

For dense BF16 pairs in this four-wave regime, time a four-group order if the compact grid is divisible by four. Map physical block `b` to first tile `(b % 4) * (N // 4) + b // 4` and advance by `N`. The quotient selects the offset within a group and the remainder selects the group, so every tile has one owner. Retain the bounded input and output copies on the final tile. At 4,096 rows, two stages and 64 blocks, the four-group order took 6.066 and 6.197 µs in matched fresh processes, compared with 6.570 and 6.299 µs for the two-group order. Using consecutive tiles per block for this same single-lane geometry took 7.673 and 7.666 µs. Across three independent full suites, the four-group kernel took 6.735, 6.193 and 7.026 µs. The earlier two-group build took 6.567, 6.935 and 5.915 µs. At that fixed 64-block grid use two groups as the control for testing four groups. The multi-lane, two-wave rule above has a different work distribution.

For a complete FP32 slab of `T` rows and `H` lanes per half, try reading half the cached signed-int16 gate indices from GM. Require even `T`, a DMA-aligned seed of `S = T * H // 2` indices, and `2 * T * H <= 32768`. An index in row `r + T // 2` is exactly `T * H` elements past the corresponding index in row `r`. Build `gate = asctile.concat(seed, seed + 2 * S)` in UB and derive `value = gate + H`. Time both full kernels. At T = 64 and H = 50 on 950PR, halving the index read reduced the two control medians from 16.358 and 16.174 µs to 15.661 and 15.928 µs at the same 102,400-byte compiled allocation. In subsequent independent full suites the half-seed kernel took 15.821, 16.362 and 15.885 µs. The preceding full-seed build took 15.634, 15.801 and 15.689 µs. Keep this as a calibration option. For dense BF16 pairs, splitting the index into two or four segments won only one of two controls. Keep the full-table load for that layout pending evidence on the target.

For a two-wave, multi-lane BF16 slab, compare shortening the index transfer separately from fixing the loop trip count at two. At 928 rows and 7 values per half on 72 AIVs, loading 3,248 signed gate indices and reconstructing the other 3,248 in UB took 6.000 and 6.298 µs. The full-index baseline with the same bounded loop took 6.482 and 6.070 µs. Fixing the loop count at two and using the same seed took 6.677 and 7.289 µs. The seed construction needs even tile height, DMA-aligned half-index length, and an in-range signed offset for every value index. Preserve the final tile’s true extents. In three independent full-suite runs, the half-index seed took 5.976, 6.502 and 6.928 µs. An earlier full-index build with the same bounded interval and geometry took 6.034, 6.017 and 6.107 µs. Start this two-wave shape with the full index table. Select a shorter seed only after its gain reproduces in a full-suite repeat on the target.

For a single BF16 pair per row, making the tile count exactly four times the grid can increase latency. On 1,000,003 rows, 3,920 pairs per tile gives exactly 256 tiles at 64 blocks. An explicit four-step loop took 6.276 and 6.839 µs at unroll two, and 8.761 and 8.861 µs at unroll four. Keeping 4,096 rows and a bounded final wave took 5.990 and 5.911 µs in those controls. Evaluate complete-wave overhead alongside the larger tile’s shorter total loop.

Check which loop executes before attributing a change in time to unrolling.
For 133,221 rows of seven BF16 elements per half, T = 1,856 gives 71 full
tiles and one final tile of 1,445 rows on the 72-AIV 950PR. In the
full-wave schedule above, `R // (N*T)` is zero. Every block takes only
one of the two residual paths. The compiled allocation was 233,856 bytes
at both unroll settings, within the 253,952-byte UB. Two independent
fresh-process medians were 6.231 and 7.022 µs at unroll one, and 6.939
and 6.043 µs at unroll two. The timed samples do not establish a benefit
from changing the unroll on the unused loop. Keep the two-wave geometry
in the comparison and time a single-tile body if the one-wave geometry
fits the measured UB budget.


On the 950PR public build, distinguish a global row gather from local reindex after a contiguous GM read. For 52,224 FP32 rows with half-width 50, starting from a 64-row slab, two global gathers of entire half-rows took 77.930 and 77.382 µs. Reading a contiguous slab and separating the lanes with local indices took 15.699 and 15.806 µs in the matched controls. Masked, fixed-width strided copies took 16.080 and 16.511 µs. All paths passed the full numerical check. Use contiguous reads plus local reindex as the starting point for this narrow, unaligned regime.

For a dense BF16 pair slab covering nearly four waves on an even grid, also test the smallest even grid that preserves the four-wave bound. With `groups = ceil(rows / tile_rows)`, it is `2 * ceil(groups / 8)`. Keep the two-group block permutation and fixed compiler options for this comparison. For 245 tiles of 4,096 pairs, 62 blocks cover 248 positions in four waves. At 62 blocks the two medians were 5.930 and 5.922 µs. At 64 blocks they were 6.890 and 6.135 µs. The 66- and 72-block comparisons each exceeded 6.3 µs in at least one repeat. All of these runs used 90,112 bytes of UB and passed the full golden check. Start by testing a reduction of one block pair, and verify the choice in an independent complete suite.
