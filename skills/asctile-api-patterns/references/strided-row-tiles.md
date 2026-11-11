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
  the budget, select a column-chunk implementation or report that the
  row kernel cannot fit.
- For the public swiglu_rows kernel with reuse_alloc=2, budget 16 bytes
  per physical lane for unroll 1 and 28 for FP32 unroll 2. Use a conservative
  32 bytes per lane for widened 16-bit inputs at unroll 2, and verify the
  compiler allocation for each emitted dtype and build. Subtract any fixed
  workspace before applying the bound.
- At that height select unroll 2 if its footprint fits and ceil(rows/T)
  supplies at least two tiles per active AIV on average. Otherwise keep
  the height at unroll 1. Launch min(available AIVs, ceil(rows/T)) blocks.
  The grid-stride loop owns the final partial tile.

Exercise the selector with a smaller UB that fits only one iteration, a
smaller UB that forces a shorter height, and short grids. Compare warmed
task times and the full golden on the target.
