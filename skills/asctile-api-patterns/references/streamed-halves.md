# Flat SwiGLU streams

Start from the [qualified flat kernel](../templates/streamed-swiglu-flat.py) on the public 950PR build. Preserve its loop bounds and per-row addresses when generating the host selector.

Derive the number of chunks from the logical half-row width and the UB-qualified tile. Specialize the address calculation when a half-row fits in one chunk or when the whole input is one row. For full aligned chunks use full-width copies. Use the true extent on a final partial chunk. Distribute only live chunks across the available vector blocks.

Compare consecutive chunks per block with a grid-stride loop. Consecutive ranges can use the quotient and remainder of total chunks divided by the active blocks to balance work without dummy iterations. For long grids of FP32 rows, also try one strided descriptor for several complete rows. For widened FP16 or BF16 compare a third pipeline stage against the larger tile allowed by two stages. Check the compiled allocation, full output and warmed device times before adopting a regime.

Use one supported `for t in asctile.range(begin, end, unroll_factor=U)` loop. Its logical step is one. For a balanced range with quotient q and remainder r, use the two prefix counts `q*b + min(b,r)` at block boundaries b and b+1. Define both outside runtime conditionals. A single chunk per row still has R distinct work items with source offsets `2*t*H` and `(2*t+1)*H`. A single row with many chunks instead uses `col=t*T`. Test both cases, multiple chunks on multiple rows, and a partial final chunk against the full output.

For wide halves, dispatch full chunks and partial chunks inside one launch.
The full-chunk loop covers `R * (H // T)` chunks and uses plain fixed-size
loads and stores. A second block-stride loop covers one `H % T` remainder
per row when it exists. Each real element must have one writer. A tile
larger than the logical half leaves all work to the remainder loop. Test
this case and a half exactly divisible by the tile. On Ascend950PR_9599
the large FP16 stream with one row took 240.944 µs for 12,672 lanes and
two stages in this implementation, versus 243.401 µs with the same
geometry and a valid-extent calculation inside every iteration. These
were separate diagnostic medians. Repeat the comparison inside a fixed
candidate before promoting a default. A pair of local slices from a
single UB input was rejected by the public compiler on this build.

When many wide rows each have a remainder, compare that second loop with
handling each row's last chunk in its ordinary traversal. A second pass
revisits every row and can lose locality. For 2,039 FP32 rows of half-width
32,760, the separate-pass candidate took 542.489 µs in the second suite.
An inline-tail control at 8,192 lanes and unroll 2 took 519.137 µs.
Those configurations also differ in tile width. To isolate traversal order,
hold tile width, unroll and block count fixed. If the half-width admits a
DMA-aligned exact divisor within UB, also test full copies at that width.

For smaller streams, inspect the final wave before choosing between
nearby aligned chunk sizes. With R rows and A active blocks, multiples
of `A / gcd(R,A)` chunks per row distribute the work evenly. Given a
capacity in lanes, round `ceil(H/capacity)` chunks up to that multiple,
then round `ceil(H/chunks)` lanes up to the DMA alignment. Recheck UB
and the actual chunk count after both roundings. Compare the resulting
tile with the capacity-based tile using the same kernel and input.

Before dispatch, enumerate the schedule on small shapes with one row and multiple chunks, multiple rows with remainders, and half-widths below the tile. Verify every logical output element has exactly one writer and that every source offset stays in its own half-row. For a full chunk use its ordinal within its own row, `(t % full_chunks_per_row) * tile`, with a guarded divisor when there are no full chunks.

When a moderate single-row BF16 stream spans at least four waves, include
consecutive chunks per block in the timing comparison. For two pipeline
stages, round the tentative wave count up to an even integer, then align
the resulting lanes per chunk to the dtype's DMA quantum and recheck UB.
For a half-width of 2,080,641, 72 AIVs and a target of 8,192 lanes, four
waves give 7,232 aligned lanes per chunk. Consecutive chunks took 7.680
and 7.525 µs in separate golden-checked control groups. Requalify the
ordering for multi-row inputs or a different device.

For a single FP32 row, also test rounding the tile **down** to the DMA
quantum. With N active blocks and W complete waves, choose
`T = (H // (N*W) // quantum) * quantum`. Require T to be positive
and to fit the compiled UB budget. Compute the residual after all W
waves. If it is at most N quanta, each remaining block can read and
write one quantum using the true extent. Guard its offset before
subtracting it from the residual. One launch then covers both the
unmasked full waves and the short tail. For H = 1,181,953 and N = 72,
four waves of 4,104 FP32 lanes cover 1,181,952 outputs. Block zero
handles the final element in an eight-lane tile. The full-case median
was 7.909 µs on Ascend950PR_9599. Benchmark other wave counts or the
ordinary bounded loop when the residual needs more than N quanta.


For moderate single-row BF16 streams spanning one to two grids of the base tile, qualify both aligned roundings before switching away from grid-stride scheduling. With H = 1,046,017 and 72 AIVs, the 8,192-lane grid-stride kernel took 5.428 and 5.097 µs. Rounding down to 7,264 lanes with the one-element tail in the main launch took 5.356 and 5.923 µs. Rounding up to 7,280 lanes and assigning consecutive chunks took 5.162 and 5.780 µs. Retain the base tile on this target for that range.

For aligned FP32 half-rows of 8–16 KiB, include `opt_level=3` at the same tile, unroll and allocation-reuse level when qualifying the compiler build. Test both the strided-row path for a well-filled grid and the flat path when a whole half-row fits one tile. On 950PR, 2,048 rows of 2,048 lanes took 23.474 and 23.345 µs with the explicit level, versus 23.660 and 23.594 µs with the default. At 512 rows the flat path took 7.676 and 7.110 µs versus 7.702 and 8.091 µs. On the four-row wide BF16 stream the direction reversed between trials. Scope this compiler choice to qualifying FP32 paths and verify it against an independent full-suite run.

### Several wide BF16 rows

For a moderate BF16 stream with more than one row but fewer rows than AIVs, compare traversing the same column chunk across rows before advancing to the next column. For a logical tile ordinal `i`, use row `i % rows` and column offset `(i // rows) * width`. The final column of each row needs its true extent on both input copies and the output. Benchmark this schedule together with chunk width.

For half-widths between one and two full grids of the base tile, try two complete grids per row. With `A` active AIVs, round `ceil(half_width / (2*A))` up to the BF16 DMA quantum of 16 lanes. Verify the resulting chunk count is exactly `2*A` and the compiled UB allocation fits. At 4 rows of 1,044,735 lanes, 72 AIVs and an 8,192-lane base, this gives 144 chunks of 7,264 lanes per row, with 5,983 valid lanes in the final chunk. In two randomized controls the column-first schedule took 12.940 and 12.963 µs. Row-first at the same width took 12.879 and 13.211 µs. The starting 8,192-lane row-first kernel took 13.808 and 13.233 µs. Keep the one-row and long multi-row streaming paths separately qualified.

### Aligned FP32 whole rows

When one flat tile covers an aligned FP32 half-row of 8–16 KiB, compare assigning each block consecutive rows with assigning block `b` rows `b + step * grid`. For moderate input traffic (up to 64 MiB) and four to eight natural waves, look for an even divisor of the row count within one eighth of the available grid. Use the block-stride specialization only if each block has a whole number of four-step groups. Pass the selected grid as a compile-time parameter, use full-extent DMA for both halves and the output, and retain the existing bounded kernel for partial rows or waves.

On 950PR, 512 rows with 2,048 FP32 values per half gave eight rows on each of 64 blocks. Unroll four used 98,304 bytes of UB (48 bytes per output lane). It took 6.839 and 7.159 µs with block-stride ownership in two fresh processes. Consecutive ownership took 7.232 and 7.457 µs at the same grid and unroll. The original 72-block dispatch took 7.845 and 7.741 µs. With automatic dispatch the same shape took 6.906, 7.512 and 7.026 µs across three complete suites. Check the target allocation and full-suite repeat before extending this rule.
