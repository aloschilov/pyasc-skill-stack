# Height and unroll for strided SwiGLU row copies

Use this geometry for the existing public 950PR SwiGLU 2D kernel. It reads
two half-rows with strided copies and writes one half-row. The short-half
gather and long flat-chunk routes have different DMA and UB costs.

For a logical half of H contiguous elements of s bytes, use H*s for the
useful payload of each row in one input descriptor. Set the physical
width W to H rounded up to a multiple of 32/s elements. Size the tile
from the useful payload and check local allocation using W including
padding. The copy's real shape still uses H.

- Start with the smallest power-of-two height T giving at least 8 KiB of
  useful half-tile payload. A whole row is the minimum. This gives less
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
  the host selector. Keep the existing upper bounds of 28 bytes/lane
  for FP32 and 32 for BF16 at unroll 2 until their emitted forms have
  been qualified. Subtract fixed workspace from the available UB once.
  Verify the compiled allocation on each new kernel/build before
  reusing a smaller coefficient for dispatch.
- At that height select unroll 2 if its footprint fits and ceil(rows/T)
  supplies at least two tiles per active AIV on average. Otherwise keep
  the height at unroll 1. Launch min(available AIVs, ceil(rows/T)) blocks.
  The grid-stride loop owns the final partial tile.

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

For a pipelined FP16 row kernel with a padded half-width and multiple
waves, also test a height between adjacent powers of two. On the public
950PR direct-division build, take a 12 KiB per-half DMA target and cap
the height by the available UB. With physical width W, active AIVs N and
usable UB U, the positive height bound is
`B = min(12288 // (2*W), U // (20*W))`. If B is positive, start with
`waves = ceil(R / (N*B))` and `T = ceil(R / (N*waves))`. Compare at the
same grid and unroll against the previous height. Keep the incoming tile
height when B < 2 or R < 2*N*B. Test that fallback explicitly on a shape
where the total row count differs from the incoming tile height. Verify the emitted
allocation and golden output, including the final partial tile.

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

### Full tiles on long FP32 grids

On 950PR, consider a three-stage copy/compute loop for aligned FP32
half-rows. Size each half-tile near 16 KiB. The measured range used a
power-of-two height of two to four rows, at least eight waves and
`36 * height * width` bytes of usable UB. Inspect the emitted allocation.
Use fixed-size copies when the physical and logical widths match and the
height divides the row count. A partial final wave is legal because the
strided loop stops at the tile count. A partial tile needs bounded copies.

For this single-pass specialization construct the input descriptor with
`l2_cache=False` and use the default output descriptor. On two neighboring
FP32 shapes, input bypass improved all four comparisons in reversed run
orders. Output bypass added over 10 µs in every comparison. Keep the
ordinary cache policy on the bounded fallback. Bind the launch to the
current stream. These controls used restored inputs on an Ascend 950PR.
[Paired cache controls and compile evidence](https://github.com/aloschilov/pyasc-skill-stack/pull/60#issuecomment-6880895812).

For an ahead-of-time width band, budget the widest emitted physical tile.
If the table declares a height of four and a padded half-width of 4,096,
the three-stage FP32 body consumed 589,824 bytes on the CI compiler. Two
rows at width 2,048 consumed 147,456 bytes. The 950PR budget was 253,952
bytes. Express the smaller tile and width step in the table itself; the
service launcher uses the declared dimensions without a runtime UB clamp.

For the separate single-wave FP32 body, compare
`@asctile.jit(reuse_alloc=2, vf_fusion=True)` with fusion off at the same
tile, launch grid and cache settings. The fused bounded loads, pointwise
math and store emitted one register loop and used two physical UB tiles.
Fusion reduced both process medians on each of two shapes; case 11 went
from 2.82/2.86 µs to 2.26/2.29 µs on local 950PR. Check finite outputs
and padded tails. Apply the decorator to a dedicated single-wave JIT so
other shapes keep their qualified compiler options.
[Matched fusion controls and emitted code](https://github.com/aloschilov/pyasc-skill-stack/pull/60#issuecomment-6881145027).

### Exact waves for aligned FP32 rows

For an aligned FP32 2D tile, compare the ordinary grid with a nearby even grid that divides the number of tiles. Keep the tile, unroll, compiler flags and loop body identical in the comparison. Let `A = min(number_of_tiles, reported_AIVs)`. After at least four waves of useful work, test even divisors from `A - A % 2` down to `ceil(7*A/8)`, inclusive. Take the largest qualifying divisor. Keep `A` if the interval contains none. Handle zero work before dividing. The seven-eighths bound limits the reduction in parallelism during calibration.

On the 72-AIV 950PR, 2,048 aligned FP32 rows with a height of two produce 1,024 tiles. The same two-row, three-stage strided kernel took 23.333 and 23.115 µs at 64 blocks in two randomized control rounds, compared with 23.694 and 23.538 µs at 72. With 2,046 rows there are 1,023 tiles and no near-capacity even divisor. For that shape the 72-block medians were 22.553 and 22.609 µs. At 64 they were 22.885 and 22.519 µs. Apply the divisor choice only on a qualified layout. A dense FP32 gather with 816 tiles had mixed results at 68 blocks despite exact divisibility.
