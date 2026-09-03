# Exact-v2 local compile feedback

The stable GELU formulas are accepted syntactically, and all three tanh
specializations compile. Exact-mode specializations fail only because the
2048-element tile overflows Unified Buffer:

- FP16/BF16 input: 819968 bytes used; 253952 available.
- FP32 input: 795392 bytes used; 253952 available.
- Measured safe baseline from the pinned-v2 integration: exact tile 512 with
  `unroll_factor=2`.
- The tanh tile 1024 is already safe (172032--184320 bytes) and should remain
  unchanged.

Preserve the measured-correct Numerical Recipes formula and reduce only the
exact-mode tile size. Ensure scalar constants remain on the right side of tile
arithmetic.
