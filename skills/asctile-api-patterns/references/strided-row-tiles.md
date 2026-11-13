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
  kernel: one tile directly at `block_idx*T`, no grid-stride loop, unroll 1.
  Bound the real height on both loads and the store. With fewer available
  blocks than tiles, the grid-stride loop must cover all remaining tiles.
  For example on 72 AIVs, 143 rows of 536 FP32 values **per half**
  select T=4 and 36 blocks; 273 rows of 509 select T=8 and 35 blocks.

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
then width=512 only in the UB check. The resulting eight-row tile uses
35 blocks on 72 AIVs. Passing 512 to both steps cut the tile to four
rows, launched 69 blocks and took 3.114 µs in a full hardware suite.

Run this guard before *both* ordinary row launch sites, including the
single-wave body. Zero falls through to streamed column chunks.
With 253952 bytes of UB, an FP32 half-row of width 32760 already needs
524160 bytes at 16 bytes per lane. A BF16 half-row of padded width 1046032
also exceeds UB at height one. Trace the actual public launches for
these widths and check that each uses a bounded column tile. For
short FP32 rows keep the qualified height and natural grid. In two
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

For a one-wave FP32 row kernel whose gate and value are read once,
use `asctile.global_tensor(x, [R, 2*H], l2_cache=False)` for the input
and the default L2 descriptor for the output. Scope this choice to the
short, single-wave kernel. Check the final partial tile with finite inputs.
On the 950PR, changing the input descriptor alone gave 2.215/2.451 µs
for 143 rows of half-width 536 and 2.753/2.725 µs for 273 rows of
half-width 509. Default descriptors gave 2.497/2.468 and 2.805/2.878 µs.
Bypassing only the output took 2.641/2.700 and 2.960/2.950 µs.

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

For aligned FP16 rows, check for two full waves plus a small row tail
before the ordinary DMA-height and unroll fallback. Let
`T, tail = divmod(R, 2*A)` for the detected AIV count A. Select this
two-stage schedule only for aligned H, T >= 2, a per-half payload
`8192 <= 2*T*H <= 16384`, a fitting two-stage UB footprint
`20*T*H <= U`, and `tail <= T//4`. Use A blocks, height T,
width H and unroll two. The block-stride loop and its bounded
final copy must still process all `ceil(R/T)` tiles. At R=1009,
H=1008 and A=72 this gives 144 full seven-row tiles and one row
in the final tile. At fixed grid, seven rows with two stages took
5.590/5.610 µs, eight rows with two stages 5.998/6.021 µs, and
eight rows with one stage 6.484/6.465 µs. Keep other layouts
on their independently qualified paths.

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
