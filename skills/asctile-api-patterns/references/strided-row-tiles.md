# Height and unroll for strided SwiGLU row copies

Use this geometry for the existing public 950PR SwiGLU 2D kernel. It reads
two half-rows with strided copies and writes one half-row. The short-half
gather and long flat-chunk routes have different DMA and UB costs.

For a contiguous input split along normalized axis `d`, the output shape
halves only `shape[d]`. The physical half-row width is
`H = (shape[d] // 2) * prod(shape[d+1:])` and the row count is
`R = prod(shape[:d])`. Use H consistently in the dispatch heuristics,
kernel descriptors, bounds and offsets. Trace a non-final split axis
through the actual launch and verify `2*R*H == input.numel()`.
On input (8192, 16384) with dim=0, the output axis is 4096 but the JIT
half-row contains 67108864 elements. Passing 4096 as the JIT width leaves
most of the output unwritten.

For a logical half of H contiguous elements of s bytes, use H*s for the
useful payload of each row in one input descriptor. Set the physical
width W to H rounded up to a multiple of 32/s elements. Size the tile
from the useful payload and check local allocation using W including
padding. The copy's real shape still uses H.

When investigating a partial final tile, compare the emitted C++ with
separate fresh `PYASC_CACHE_DIR` and `PYASC_DUMP_PATH` directories per
variant. On the public 950PR build, a bounded 2D `copy_in` produced the
same conditional `Duplicate` of missing rows with implicit padding and
with `pad_value=0.0`. A final-block branch using the true compile-time
extent removed that fill and shortened its register calculation. Keep
the output bounded and time both versions at the same grid before
changing dispatch. In three rounds on each of two short FP32 geometries,
the exact terminal extent gave no consistent speedup.
[Generated C++ comparison and matched hardware controls](https://github.com/aloschilov/pyasc-skill-stack/pull/60#issuecomment-6859782820).

A partial output can also read a full input tile when its first row leaves
at least T rows in the tensor. Place the short output interval at the
front, then assign the remaining full tiles after it. Compare bounded and
full reads at those same row offsets, with each output row owned once.
On 950PR this removed both conditional fills in the two-input FP32
kernel. A one-row tail at T8/H509 reads seven extra input rows, and a
three-row tail at T4/H536 reads one extra. Neither shape gave a repeatable
benefit across public and finite controls. Keep the ordinary bounded
reads for these geometries.

For pointwise register math on planar UB tiles, flatten the physical extent
`T*W` only when gate, value and output have the same row pitch and the GM
store uses the logical bounds. For an FP32 span of at least 64 lanes with a
32-byte-aligned end, a final full vector may start at `T*W - 64`. If it
overlaps the preceding vector and output can alias either input, load both
input windows for both vectors before writing either result. At T=4 and
W=536 the last windows start at lanes 2048 and 2080 and overlap by 32.
This schedule works for an elementwise calculation that writes the same
value to each repeated lane. For a reduction, count each logical lane once.
Inspect the compiled allocation and compare a single register chain with
two interleaved chains at the same DMA bounds and grid. On 950PR the
paired version took 2.137 µs at T4/W536 and 2.598 µs at T8/W512. It
beat the high-level body in two of three rounds on each public and finite
fixture. Use a paired control before specializing another shape.

For a single-wave FP32 tile split across two DMA stages, test the unsplit
tile and both serial and prefetched stages at the same grid. When each
stage owns a contiguous range of whole rows, the GM descriptor can start
at its first row, with shape `[count, 2*H]` and input offsets `[0,0]` and
`[0,H]`. Derive the output descriptor from the same first row and its true
count, including the partial last block. Inspect the emitted full-block
path. On 950PR the fixed GM window removed conditional missing-row fills,
and `vf_fusion=True` emitted one register math scope per stage. Even with
both changes, two prefetched stages took 2.554/2.679 µs against
2.592/2.625 µs for the unsplit control at H=509. They also lost both
rounds at H=536. Retain the unsplit schedule for these qualified shapes.
[Measured stage orders, compiled code and exact-binary replay](https://github.com/aloschilov/pyasc-skill-stack/pull/60#issuecomment-6861262761).

For a packed two-stage prefetch, inspect both `copy_in` operations in the
emitted full-block branch. Independent input readiness is visible when
the two MTE2 transfers use different buffer IDs and the first vector scope
acquires only its own input and output. Check separately whether the
second result aliases the first input, and whether either input still has
a conditional missing-row fill. On the 950PR build, exact stage-local GM
descriptors removed both fill sites and `reuse_alloc=0` gave all four tiles
distinct buffers. Case 11 still took 2.950 and 2.989 µs. Keep the simpler
single-wave dispatch when the staged public kernel loses its paired
qualification. A native event schedule alone cannot qualify a generated
AscTile launch.
[Generated buffer IDs, stage ordering and matched native measurements](https://github.com/aloschilov/pyasc-skill-stack/pull/60#issuecomment-6864335996).

- Start with the smallest power-of-two height T giving at least 8 KiB of
  useful half-tile payload. For positive R and H, compute
  `needed = (8192 + H*s - 1) // (H*s)`, then
  `T = 1 << (needed - 1).bit_length()`. Rounding down at either step
  gives too little payload for unaligned widths. A whole row is the minimum. This gives less
  than 16 KiB when a row is smaller than 8 KiB. Halve T until it fits the
  row count and the single-iteration UB budget. If a single row exceeds
  the budget, report that the row kernel cannot fit. A column-chunk
  fallback needs its own UB-qualified chunk size.
- Widen the two FP16 inputs to FP32. Compute the denominator from the
  gate and divide the gate directly by it, then multiply by the value and
  cast once to the output dtype. This saves the extra live FP32 tile in
  the reciprocal-then-multiply expression. Check the full output with the
  task's tolerance since the rounding order changes.
- Size UB from the emitted expression, dtype and compiler configuration.
  On the archive-bundled 950PR public build with `reuse_alloc=2`, the
  direct-division FP16 row kernel used **20 bytes per physical lane** at
  unroll 2 (163,840 bytes for 8,192 lanes). The reciprocal form used 24.
  Use the qualified direct-division coefficient for FP16 and retain the
  conservative 16 bytes/lane at unroll 1. Distinguish FP16 from BF16 in
  the host selector. Pass the caller's actual dtype or dtype tag to this
  selector. The element byte count alone maps both FP16 and BF16 to 2
  and cannot choose the correct two-stage capacity. Keep the existing upper bounds of 28 bytes/lane
  for FP32 and 32 for BF16 at unroll 2 until their emitted forms have
  been qualified. Subtract fixed workspace from the available UB once.
  Verify the compiled allocation on each new kernel/build before
  reusing a smaller coefficient for dispatch.
- At that height select unroll 2 if its footprint fits and ceil(rows/T)
  supplies at least two tiles per active AIV on average. Otherwise keep
  the height at unroll 1. Launch min(available AIVs, ceil(rows/T)) blocks.
  For FP32 when `ceil(rows/T) <= available AIVs`, specialize a single-wave
  kernel with a single bounded row interval per block and unroll 1.
  Bound the real height on both loads and the store. With fewer available
  blocks than tiles, the grid-stride loop must cover all remaining tiles.
  For example on 72 AIVs, 143 rows of 536 FP32 values **per half**
  select T=4 and 36 blocks. For 273 rows of 509 use eight rows and 35 blocks. Keep the balanced-grid helper below for controlled comparisons.

Choose the ordinary bounded row kernel's unroll independently of whether
a copy needs `real_shape`. A ragged final tile and a padded local row both
support a two-stage grid-stride loop. After fitting the one-stage tile,
use the two-stage live bytes per physical lane from the qualified emitter.

```python
def bounded_row_unroll(rows, height, width, blocks, two_stage_bytes_per_lane, usable_ub):
    tiles = (rows + height - 1) // height
    fits = height * width * two_stage_bytes_per_lane <= usable_ub
    return 2 if tiles >= 2 * blocks and fits else 1
```

Use positive dimensions and the actual number of launched blocks. For
FP16 with 2049 rows and H512, T8 on 72 blocks gives 257 tiles and one
valid row in the last tile. With a 253952-byte UB, 20 bytes per lane
select U2. For FP32 R52224/H50, use T64/W56 and 28 bytes per lane for
the conservative U2 fit, then launch the bounded body at U2. The
corresponding W30 emitter allocated 81920 and 86016 bytes. Trace both
public entries including the final kernel argument.

Use the padded width and the one-stage coefficient of the actual row
emitter for the final fit test, including height one. For example:

```python
def fit_row_height(height, rows, width, bytes_per_lane, usable_ub):
    if min(height, rows, width, bytes_per_lane) <= 0:
        return 0
    while height > 0:
        if height <= rows and height * width * bytes_per_lane <= usable_ub:
            return height
        height //= 2
    return 0
```

Keep the logical and padded widths as separate arguments throughout
the public dispatch. With H=509 in FP32, exactly four rows carry
8144 useful bytes even though their padded allocation is 8192 bytes.
Compute the requested height using H before calling the UB guard with W:

```python
def ordinary_dma_height(rows, half, itemsize, bytes_per_lane, usable_ub):
    if min(rows, half, itemsize) <= 0:
        return 0
    row_bytes = half * itemsize
    required = (8192 + row_bytes - 1) // row_bytes
    height = 1 << (required - 1).bit_length()
    width = ((row_bytes + 31) // 32) * (32 // itemsize)
    return fit_row_height(height, rows, width, bytes_per_lane, usable_ub)
```

A trace of the public FP32 selector on 273 rows must see half=509,
then width=512 only in the UB check. The ordinary eight-row fit gives 35 blocks on 72 AIVs. Passing 512 to both steps cut the tile to four
rows, launched 69 blocks and took 3.114 µs in a full hardware suite.

Run this guard before *both* ordinary row launch sites, including the
single-wave body. Zero falls through to streamed column chunks.
With 253952 bytes of UB, an FP32 half-row of width 32760 already needs
524160 bytes at 16 bytes per lane. A BF16 half-row of padded width 1046032
also exceeds UB at height one. Trace the actual public launches for
these widths and check that each uses a bounded column tile. For
short FP32 rows retain the ordinary geometry outside the single-wave specializations. In two
rounds with full arithmetic, increasing height from 4 to 8 at half-width
536 reduced the grid from 36 to 18 and increased latency from
2.411/2.442 to 2.655/2.461 µs. At half-width 509, increasing height from
8 to 12 reduced the grid from 35 to 23 and increased latency from
2.833/2.846 to 3.108/2.976 µs.

For long FP32 grids with DMA-aligned half-rows, size one 2D copy towards
16 KiB and try three pipeline stages before the ordinary row fallback.
Use the greatest power-of-two height at or below `16384 // (4*H)`.
For the measured regime require two to four rows per tile, at least eight
waves of row tiles, and `36 * T * H <= usable_ub` with the qualified
direct-division emitter and allocation reuse. Pass `T, H, 3` to the
existing row kernel. A partial final tile still uses its real row count.
For 2048 rows with aligned FP32 half-width 2048 the actual three-stage
allocation was 147456 bytes at T=2. At fixed T=2 and 72 blocks the
two-stage medians were 23.670/24.096 µs and the three-stage medians were
23.277/23.115 µs. In two additional runs three stages took 23.416/23.251 µs
on 72 blocks and 23.553/23.216 µs on 64. The published 950PR baseline
is 23.650 µs. Keep the natural grid unless its change qualifies separately.

In the three-stage FP32 branch, whole physical tiles permit a separate
full-copy specialization when `rows % height == 0` and `W == H`. Use
`@asctile.jit(reuse_alloc=2, opt_level=3)`, the same strided tile-index
loop, two `copy_in(gx, [r, col], [T, H])` calls for col 0 and H, and
`copy_out(out, gy, [r, 0])`. The final wave may have fewer active
blocks. The loop bound is the number of tiles, so every issued DMA has
a full extent even with a partial wave. Route a partial **tile** to the
bounded body. Apply this legality check to every caller of any fixed-copy
row JIT, that uses one uniform height, including the FP32 three-stage fallback:

```python
lanes_per_dma = 32 // x.element_size()
width = ((half + lanes_per_dma - 1) // lanes_per_dma) * lanes_per_dma
if rows % height == 0 and width == half:
    grid = blocks
    if x.dtype == torch.float32 and unroll == 3 and x.device.type == "npu":
        import ctypes
        grid = (blocks, ctypes.c_void_p(torch.npu.current_stream(x.device).npu_stream))
    full_rows[grid](x, out, rows, half, height, width, unroll)
else:
    bounded_rows[blocks](x, out, rows, half, height, width, unroll)
```

On the resident 950PR FP32 three-stage path, bind the fixed-copy launch
to the input device's current Torch stream at each call. With R2046,
H2048 and the same 72 blocks, two release-default rounds took 22.96 and
23.15 µs on the caller stream. The integer-grid control took 23.91 and
23.47 µs. A CPU-staged input still uses the integer grid.

Compute the alignment in a single unit. At H2048 in FP32 this gives
256 DMA units and W2048. Trace both R2048 and R2046 with T2 to the
fixed-copy JIT; R2047 has one row left for the bounded JIT. At H509
in FP32 the physical width is 512 and all DMA extents use H509.

The aligned FP16 two-wave specialization below uses two explicit extents.
At R1009/T7 its final read and write start at row 1001 and span eight rows. For generated code, replay the effective sizes of both
reads and the store. A copy that omits `real_shape` uses its full physical
shape. Golden checks of the logical output can pass after an out-of-range
write. For R2047/T2 verify that the last three-stage tile has one row too.
Keep ordinary L2 caching for both three-stage descriptors.

At 72 blocks and T2/U3, the fixed-copy level-3 specialization on the
950PR took 23.173/23.367 µs at 2048 rows and 22.436/22.506 µs at 2046
rows. Bounded default-level controls took 23.578/23.071 and
22.804/22.564 µs. The measurements support trying this specialization
and checking the complete suite. In a separate two-round control,
bypassing L2 on the output alone increased the original kernel times
to 34.555/34.703 and 34.582/33.738 µs respectively.

Check this branch before applying the default 8 KiB row-height selector.
The returned positive height is the JIT height in rows. The physical width
equals the logical half-width because this branch requires aligned FP32.

```python
def three_stage_row_height(rows, half, dtype, aivs, usable_ub):
    if dtype != torch.float32 or rows <= 0 or half <= 0 or aivs <= 0 or half % 8:
        return 0
    rows_per_dma = 16384 // (4 * half)
    if rows_per_dma < 2:
        return 0
    height = 1 << (rows_per_dma.bit_length() - 1)
    tiles = (rows + height - 1) // height
    if height > 4 or tiles < 8 * aivs or 36 * height * half > usable_ub:
        return 0
    return height
```

For the one-wave FP32 path, keep each logical half-row together. On
950PR, splitting each half into two column strips increased the grid
from 36 to 72 or from 35 to 70. At 143 rows by 536 FP32 lanes per half,
the full row took 2.390/2.477 µs in two fresh processes and the split
took 2.884/3.169 µs. At 273 rows by 509 the corresponding times were
2.981/2.808 and 3.403/3.004 µs. The height, arithmetic and valid
extents were fixed. Record the profiled grid and compiled loop body
when validating a new generation.

Assign strided 2D work in **tile indices**. For R rows and height T,
there are `ceil(R/T)` iterations across the entire grid. The starting row
is `tile_index*T`. Use the same row for both input copies and the output.
An emitter that issues a T-row write at every row computes overlapping
values and introduces cross-block writes. A matching numerical golden
on a deterministic input can hide this error. Check ownership by tracing
the emitted body across all blocks: the sum of valid extents must be R
and the sorted output intervals must partition [0,R). For R=52224,
T=64 and N=72 this is exactly 816 disjoint writes. For R=1009, T=7
and N=72 the bounded fallback issues 144 seven-row writes and one one-row
write. The merged-tail specialization issues 143 seven-row writes and one
eight-row write.

Define both ranked descriptors in the row JIT. A two-component offset
requires a rank-two descriptor even when its backing storage is contiguous.
Use the logical row width in GM and reserve the padded width for UB.

```python
gx = asctile.global_tensor(x, [R, 2 * H])
gy = asctile.global_tensor(y, [R, H])
count = (R + T - 1) // T
for tile_index in asctile.range(asctile.block_idx(), count,
                                asctile.block_num(), unroll_factor=U):
    r = tile_index * T
    valid = min(T, R - r)
    gate = asctile.copy_in(gx, [r, 0], [T, W], real_shape=[valid, H], pad_value=0.0)
    value = asctile.copy_in(gx, [r, H], [T, W], real_shape=[valid, H], pad_value=0.0)
    gate_f = gate.to(asctile.float32)
    out = (gate_f / (1.0 + asctile.exp(-gate_f))) * value.to(asctile.float32)
    asctile.copy_out(out.to(gate.dtype), gy, [r, 0], real_shape=[valid, H])
```

For a one-wave FP32 row kernel whose gate and value are read once,
use `asctile.global_tensor(x, [R, 2*H], l2_cache=False)` for the input
and the default L2 descriptor for the output. Scope this choice to the
short, single-wave kernel. Check the final partial tile with finite inputs.
On the 950PR, changing the input descriptor alone gave 2.215/2.451 µs
for 143 rows of half-width 536 and 2.753/2.725 µs for 273 rows of
half-width 509. Default descriptors gave 2.497/2.468 and 2.805/2.878 µs.
The release-default restored-input evaluator reproduced the direction with a
fresh OpenCode module. At H536 (36 blocks), bypass took 1.87/1.78 µs and
default caching took 2.18/2.14 µs. At H509 (35 blocks) the corresponding
pairs were 2.35/2.38 and 2.54/2.54 µs. The two rounds reversed the order of
the variants. Both reads used the same descriptor and the output kept its
default cache setting.
Use the same bypass descriptor for both disjoint, one-pass reads in this path.
In a matched test crossing both read orders with the gate's cache hint, giving
only the gate the default hint increased seven of eight paired process
medians and tied the eighth. The grid, output hint and arithmetic stayed fixed.
For an operand that is read again, measure its reuse separately before applying
this observation. [Paired read-order and cache measurements](https://github.com/aloschilov/pyasc-skill-stack/pull/60#issuecomment-6873662947).

Bypassing only the output took 2.641/2.700 and 2.960/2.950 µs.

When one wave covers the rows and a complete half-row fits UB, start with
the full half-width for each 2D DMA. Splitting the width at the same grid
shortens the contiguous burst and increases the rows per tile. Consider
column stripes when a full-width tile exceeds UB or there are too few
independent row tiles to use the available AIVs. When measuring that change,
hold the grid fixed and check both input halves and the bounded output of
every stripe, including the final partial row and column.

Keep the remaining columns in the same full-width 2D copy when a single
FP32 tile fits UB and spans several 64-lane vectors. On 950PR, splitting a
536-lane half into 512+24 and giving the last 24 lanes to extra blocks
took 2.371–2.405 µs across two fringe heights and two rounds. Whole rows
on 36 blocks took 2.164–2.199 µs. Splitting a 509-lane half into 448+61
on the original blocks also increased both round medians. If testing
independent column owners on another layout, check the physical output
addresses across every row. With FP32, an aligned column offset in row
zero stays 32-byte aligned in later rows only when H is a multiple of
eight. Verify that separate writers own disjoint DMA bursts.

To test eliminating the last partial row tile, compare the DMA-derived
height with a divisor of the row count. Changing height also changes the
number of active blocks. At the exact height, test column bands separately:
round the regular band up to a 32-byte burst and give the final band its
actual width. Check the last band's starting address and extent on every
row. Measure the full-width control in each round before changing the
dispatcher. On the 950PR a 143×536 FP32 output took 2.216 and 2.304 µs with
four rows per tile and 36 blocks. Eleven exact rows with two column bands
took 2.304 and 2.334 µs on 26 blocks. One and four bands took longer in
both rounds. Keep the DMA-derived height for this layout.

For the ordinary single-wave launch with exactly `ceil(R/T)` blocks and R > 0,
each starting row is valid. Compute `rows = min(T, R - r)`
directly and use `real_shape=[rows,H]` for all three DMA operations.
This one-wave body accepts any W >= H that fits UB, including padded
FP32 rows. Route on dtype and tile count after the fit check. The
`W == H` guard belongs to the separate fixed-size three-stage body.
Trace R273/H509 through the ordinary fit to T8/W512/N35 in the launched JIT.
The arithmetic is lane-wise and the output store excludes all padding.
Keep the two half-row reads batched across T rows through one source
descriptor. At R273/H509, per-row reads with a final concat took
6.191/6.183 µs, against 2.621/2.636 µs for batched 2D reads. Concat
in pairs took 4.100/3.754 µs. At R143/H536 the paired variant gave
2.524/2.203 µs against 2.206/2.247 µs. Both comparisons kept the
original tile, grid and exact output extent.

To benchmark an alternative padded single-wave FP32 grid on 950PR, distribute rows across up to
two-thirds of the available AIVs when the extra blocks shorten the
ordinary tile and each half still moves at least 8 KiB per full block.
Apply this after the ordinary positive UB fit and only to a single wave.
Use the balanced grid only when every block receives a row.

```python
def padded_row_grid(rows, half, width, height, aivs):
    natural = (rows + height - 1) // height
    target = min(rows, 2 * aivs // 3)
    if width <= half or target <= natural:
        return natural, height
    shorter = (rows + target - 1) // target
    if shorter < height and (rows // target) * half * 4 >= 8192:
        return target, shorter
    return natural, height
```

Pass `balanced = blocks > natural` and the actual block count as compile-time arguments.
For the balanced path use `q, rem = R // N, R % N`,
`r = b*q + min(b, rem)` and `valid = q + min(1, max(0, rem-b))`,
where b is the physical block index. Both reads use `[T,W]` in UB and
`[valid,H]` in GM, as does the store. Keep the input descriptor rank two
with dimensions `[R,2*H]` and column offsets 0 and H. On the tested
wheel, changing it to `[R,2,H]` and reading a singleton middle
dimension for each half failed the finite numerical check for 57,342 of
76,648 outputs at R143/H536. On an input labelled by physical
half-row, each source read advanced one half-row per local row inside
a tile. The intended stride was two half-rows. Only the first row per
tile had the intended labels at R143/H536 and R273/H509.

At R273/H509/W512 and 72 AIVs the ordinary height is eight. The balanced
path uses T6/N48. Blocks 0 through 32 own six rows each and the remaining
15 own five. In two shuffled rounds this geometry took 2.576 and 2.574 µs.
The eight-row N35 control took 2.621 µs in both. A fresh full-suite
generation at N48 took 3.646 µs. In a subsequent shuffled control, N48
took 3.301/2.800 µs and N35 took 2.713/3.337 µs. Use the ordinary
padded grid in generated modules until a paired comparison reproduces
the balanced advantage. For aligned FP32 retain the
ordinary T and grid, rotating the tile ordinal by half the grid:

```python
b = asctile.block_idx()
n = asctile.block_num()
tile_index = (b + n // 2) % n if W == H else b
r = tile_index * T
valid = min(T, R - r)
```

On the 950PR with T4/H536/W536/N36, this rotation took 2.152 and 2.185
µs in paired runs versus 2.217 and 2.314 µs for direct ownership. The
three-row tail moved from physical block 35 to block 17. At
T8/H509/W512/N35, rotating took 2.605/2.640 µs versus 2.597/2.639 µs.
Verify ownership on the actual JIT and compare medians in the full suite.

When testing the cost of this rotation, keep the mapping of block to tile
identical. If one wave owns every tile and the launch has exactly
`ceil(R/T)` blocks, that count is available from the compile-time shape.
For `0 <= b < n`, `shifted = b + n//2` wraps at most once. Subtracting
`n` when `shifted >= n` preserves the mapping for both odd and even grids.
Check the lowered kernel for a runtime divide or remainder, then time the
same layout on a finite input. On 950PR the subtraction improved all three
public R143/H536 rounds, but only one of three finite rounds at the same
R/H/T and grid. At R144 it improved all three finite rounds, while R139
and R140 were mixed. Keep the ordinary rotation above as the default.

For a single-wave FP32 row launch on a resident Torch NPU input, submit
the kernel to the current Torch stream for that input device. On the public
wheel whose `LaunchOptions` takes `core_num, stream`, pass a pointer-width
stream handle in the JIT subscript. The CPU staging route uses its regular
grid-only call. Query the stream at invocation time so a caller using a
different Torch stream gets the correct launch.

```python
import ctypes

if input.device.type == "npu":
    stream = ctypes.c_void_p(torch.npu.current_stream(input.device).npu_stream)
    row_single_wave_fp32[blocks, stream](input, out, R, H, height, W)
else:
    row_single_wave_fp32[blocks](input, out, R, H, height, W)
```

Verify that the preceding framework task and the kernel have the same stream
ID in the trace. Device IDs and correct output alone do not show this. On a
950PR with CANNBench cache conditioning, using that stream saved 0.595 and
0.613 µs at 36 blocks, and 0.545 and 0.437 µs at 35 blocks. The compiled
device binary was identical within each shape. Repeat the comparison when
changing the wheel or targeting a different launch family.

For the ordinary padded FP32 single-wave body, issue the bounded value
read first, then the bounded gate read, then compute
`(gate / (1.0 + exp(-gate))) * value`. Keep both DMA bounds identical
and keep gate as the argument of exp. At N35/T8/H509/W512 this order
took 2.544/2.594 µs versus 2.607/2.654 µs with gate read first.
Another shuffle gave 2.649/2.589 versus 3.158/3.333 µs. The earlier
denominator-before-value schedule gave 2.559/2.642 µs in its own run.
For aligned FP32 keep the gate read first and both reads ahead of math.
Verify the order at the actual JIT call. For a change to the aligned-row
default, repeat the paired test on finite neighboring widths in both launch
orders. With input bypass and four rows per tile on 950PR, value-first saved
0.54/0.06 µs at H536. At H528 and H544 the direction changed between rounds.
[Release and finite-input trials](https://github.com/aloschilov/pyasc-skill-stack/pull/60#issuecomment-6873970983).

Exercise the selector with a smaller UB that fits only one iteration, a
smaller UB that forces a shorter height, and short grids. Compare warmed
task times and the full golden on the target.

When benchmarking narrower tiles to improve pipeline occupancy, retain
the previous choice for short grids until both variants pass a paired
timing check. The padded FP16 half-widths 506 and 1,008 have different
numbers of rows per block. Halving an already pipelined tile helped the
first geometry. Halving a one-stage tile to force two stages regressed
the second. For a small FP32 grid with at most one tile per active AIV,
specialize the body to use the block index directly. Keep the row count
on the final tile bounded. Compare the fixed-grid body with the normal
loop at the same launch size, tile extent and unroll.

### Padded FP16 rows across multiple waves

On the qualified direct-division kernel, use the wave-balanced height below
for padded FP16 rows when the bounds allow at least two waves. Select this
geometry before testing the unroll of the resulting tile. Keep the ordinary
row selector for BF16, aligned halves and short grids. On the public
950PR direct-division build, take a 12 KiB per-half DMA target and cap
the height by the available UB. With physical width W, active AIVs N and
usable UB U, the positive height bound is
`B = min(12288 // (2*W), U // (20*W))`. If B is positive, start with
`waves = ceil(R / (N*B))` and `T = ceil(R / (N*waves))`. Compare at the
same grid and unroll against the previous height. Keep the incoming tile
height when B < 2 or R < 2*N*B. Test that fallback explicitly on a shape
where the total row count differs from the incoming tile height. Verify the emitted
allocation and golden output, including the final partial tile.

For direct-division FP16, the 20-byte coefficient already includes
both stages. Applying another factor of two in the unroll test selected
one stage in a clean generation. Use this selector before the ordinary
row fallback. Its result is a height in rows for the two-stage row JIT.
A zero delegates both height and unroll to the ordinary selector.

```python
def padded_fp16_two_stage_height(rows, half, dtype, aivs, usable_ub):
    if dtype != torch.float16 or min(rows, half, aivs, usable_ub) <= 0 or half % 16 == 0:
        return 0
    width = ((half + 15) // 16) * 16
    limit = min(12288 // (2 * width), usable_ub // (20 * width))
    if limit < 2 or rows < 2 * aivs * limit:
        return 0
    waves = (rows + aivs * limit - 1) // (aivs * limit)
    return (rows + aivs * waves - 1) // (aivs * waves)
```

The bounded row kernel takes both `half` (the GM half-row stride) and
`width` (the padded UB extent). Pass the selected height between them.
For 3,003 rows and a logical half-width of 506, a launch trace should
record the five scalar arguments `(3003, 506, 11, 512, 2)` after the
two tensor arguments. Bind the recorded arguments to the actual JIT
signature when checking host dispatch. This catches a missing stride
before device compilation.

```python
height = padded_fp16_two_stage_height(rows, half, dtype, aivs, usable_ub)
if height > 0:
    width = ((half + 15) // 16) * 16
    blocks = min(aivs, (rows + height - 1) // height)
    bounded_rows[blocks](x, out, rows, half, height, width, 2)
    return out
```

For aligned FP16 rows, check for two full waves plus a small row tail
before the ordinary DMA-height and unroll fallback. Let
`T, tail = divmod(R, 2*A)` for the detected AIV count A. Select this
two-stage schedule only for a 32-byte aligned FP16 half-row
(`(2*H) % 32 == 0`), T >= 2, a per-half payload
`8192 <= 2*T*H <= 16384`, a fitting two-stage UB footprint
`20*T*H <= U`, and `tail <= T//4`. Use A blocks, height T,
width H and two stages. When the tail is positive and the enlarged tile fits
(`20*(T+tail)*H <= U`), use the strided assignment in
`templates/paired-fp16-rows.py`. Block b starts its first tile at b*T
and its second at (A+b)*T. Only the last block enlarges its second copy
to T+tail. Call `paired_fp16_height` in the host dispatcher. It returns zero
for an FP16 half-width of 1032 because its byte length is 2064. Both reads and the store have that same static extent. Use
distinct JIT tensor variables in branches with different static shapes.
Otherwise use the bounded tile-index loop, retaining its ordinary height.

At R=1009, H=1008, A=72 this gives T=7 and a final eight-row tile.
Three interleaved rounds on 950PR took 5.911/5.267/5.682 µs versus
5.940/5.489/6.083 µs for the bounded loop. Two additional rounds at
R=577, H=1536 and R=1154, H=768 favored the same assignment. Keep the
height and the last block's extent derived from the actual shape and UB.

At R=3003, H=506, W=512 and N=72, B=12 gives four waves and T=11.
Independent warmed medians were 6.530 and 6.359 µs, versus 6.781 and
7.080 µs with the original height of eight. For the aligned 512-wide
control, T=10 took 5.016 and 5.043 µs while T=8 took 5.011 and 4.569 µs.
Retain the aligned selector for that regime.

For aligned FP16 half-rows that fill the entire physical width, test a
fixed full-wave loop. If the tile height divides R and N divides R / T,
use full-size 2D loads and stores for every iteration. Compare the whole
reported AIV count and a nearby divisor of the tile count that supports
several complete pipeline waves. Derive both candidates from the runtime
device properties. On one 72-AIV 950PR, 1,024 rows with a half-width of
1,024 had 256 four-row tiles. Using 64 blocks, four complete waves and
unroll two took 4.479 and 4.422 µs in two control groups. Guard the exact
width and height conditions; regenerate the remainder path for ragged
shapes.

### Exact waves for aligned FP32 rows

For an aligned FP32 2D tile, compare the ordinary grid with a nearby even grid that divides the number of tiles. Keep the tile, unroll, compiler flags and loop body identical in the comparison. Let `A = min(number_of_tiles, reported_AIVs)`. After at least four waves of useful work, test even divisors from `A - A % 2` down to `ceil(7*A/8)`, inclusive. Take the largest qualifying divisor. Keep `A` if the interval contains none. Handle zero work before dividing. The seven-eighths bound limits the reduction in parallelism during calibration.

On the 72-AIV 950PR, 2,048 aligned FP32 rows with a height of two produce 1,024 tiles. The same two-row, three-stage strided kernel took 23.333 and 23.115 µs at 64 blocks in two randomized control rounds, compared with 23.694 and 23.538 µs at 72. With 2,046 rows there are 1,023 tiles and no near-capacity even divisor. For that shape the 72-block medians were 22.553 and 22.609 µs. At 64 they were 22.885 and 22.519 µs. Apply the divisor choice only on a qualified layout. A dense FP32 gather with 816 tiles had mixed results at 68 blocks despite exact divisibility.
