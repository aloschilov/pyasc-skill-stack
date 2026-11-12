# Independent OpenCode generation on David, 2026-11-12

The independently generated SwiGLU passed all 20 full-output checks on David. Its geometric mean relative to CANNBench was **0.6792×**, with 3/20 cases above 1×. It met **1/20** saved archive times (case 10). The earlier integrated candidate recorded 1.0482× and met 20/20 in its last full run.

OpenCode generated a single kernel, `_swiglu_tiles`, for every shape and selected unroll 1 in all 20 cases. The measured grid matched the integrated candidate in 12/20 cases. All new grids used 72 blocks except case 11, which used 69. The integrated candidate used 64 in cases 1–4, 12 and 15, 35 in case 11 and 36 in case 13.

Case 12 shows the largest loss. For 1,000,003 BF16 pairs, OpenCode ran two strided copies per tile, one for each half. Each copy padded one useful lane per row to 16 lanes. It selected a 256 × 16 tile, unroll 1 and 72 blocks. The integrated `swiglu_permuted_pairs` read consecutive pairs, with a 4,096-row tile, unroll 2 and 64 blocks. OpenCode measured 393.194 µs, the integrated candidate 6.303 µs and CANNBench 7.273 µs.

One OpenCode 1.2.19 invocation of `cannbot/GLM-5.3` received the public case/specification and the skill files, including their reusable flat-kernel template, from commit `c0d20dd`. The input excluded the integrated candidate, its dispatch table and prior measurements. The model had no tool access. Only the Markdown fence was removed from its answer. The supervisor recorded exit 124, but the event log contains the complete text followed by `step_finish(stop)`. The extracted source and the file checked before each hardware case have the same SHA-256. [Request](request.txt) · [Input manifest](generation-inputs.json) · [Original response events](opencode.jsonl) · [Generated operator](swi_glu.py)

The run used the public archive-bundled PyASC runtime and CANN 9.2.0 on Ascend950PR_9599 in Docker. Each case ran in a fresh process. Profiling recorded the device time for three warmed calls after checking the full result against the golden. All four calls used the same grid. The table reports the median of the three samples and the host tile and unroll observed for that case. Speed is published 950PR CANNBench time divided by the new median. The geometric mean uses the 20 unrounded ratios. A case meets the archive threshold when its new median is at most the saved David replay median for `job_f9f0facb7d6b`. [New measurements](measurements.json) · [Archive medians](../native/archive-measurements.json) · [Previous integrated run](../parity/round11-repeats.json) · [CSV comparison](../comparison.csv)

| Case | Previous kernel / grid | OpenCode grid | Tile (rows × cols) | Unroll | OpenCode µs | CANNBench µs | CANNBench / OpenCode | Archive µs | Within archive |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 12 | swiglu_permuted_pairs / 64 | 72 | 256 × 16 | 1 | 393.194 | 7.273 | 0.0185× | 6.345 | no |
| 9 | swiglu_local_waves / 72 | 72 | 256 × 16 | 1 | 21.634 | 6.782 | 0.3135× | 6.385 | no |
| 13 | swiglu_short_rows / 36 | 72 | 2 × 536 | 1 | 3.186 | 1.500 | 0.4708× | 2.550 | no |
| 11 | swiglu_short_rows / 35 | 69 | 4 × 512 | 1 | 3.282 | 2.021 | 0.6158× | 3.010 | no |
| 19 | swiglu_column_stripe / 72 | 72 | 4 × 1024 | 1 | 19.808 | 14.280 | 0.7209× | 13.411 | no |
| 3 | swiglu_flat / 64 | 72 | 4 × 1024 | 1 | 62.425 | 48.590 | 0.7784× | 45.645 | no |
| 16 | swiglu_inline_tail / 72 | 72 | 1 × 4096 | 1 | 10.687 | 8.790 | 0.8225× | 8.310 | no |
| 17 | swiglu_inline_tail / 72 | 72 | 1 × 8192 | 1 | 10.156 | 8.760 | 0.8625× | 8.398 | no |
| 6 | swiglu_striped / 72 | 72 | 1 × 4096 | 1 | 7.104 | 6.200 | 0.8727× | 6.134 | no |
| 4 | swiglu_flat / 64 | 72 | 1 × 8192 | 1 | 269.533 | 241.490 | 0.8960× | 243.929 | no |
| 14 | swiglu_rows / 72 | 72 | 16 × 512 | 1 | 8.325 | 7.540 | 0.9057× | 7.385 | no |
| 8 | swiglu_flat_edge / 72 | 72 | 1 × 8192 | 1 | 9.391 | 8.540 | 0.9094× | 8.488 | no |
| 7 | swiglu_rows / 72 | 72 | 8 × 1008 | 1 | 6.848 | 6.500 | 0.9492× | 6.687 | no |
| 2 | swiglu_rows / 64 | 72 | 4 × 2048 | 1 | 24.678 | 23.650 | 0.9583× | 23.782 | no |
| 15 | swiglu_whole_row / 64 | 72 | 4 × 2048 | 1 | 8.097 | 7.780 | 0.9608× | 7.520 | no |
| 5 | swiglu_inline_tail / 72 | 72 | 4 × 2048 | 1 | 542.537 | 522.100 | 0.9623× | 523.609 | no |
| 18 | swiglu_rows / 72 | 72 | 4 × 2048 | 1 | 24.337 | 23.630 | 0.9709× | 23.214 | no |
| 1 | swiglu_aligned_rows / 64 | 72 | 8 × 1024 | 1 | 6.162 | 6.380 | 1.0354× | 5.016 | no |
| 20 | swiglu_full_slab / 72 | 72 | 128 × 56 | 1 | 18.541 | 19.560 | 1.0550× | 15.734 | no |
| 10 | swiglu_rows / 72 | 72 | 16 × 512 | 1 | 5.440 | 6.180 | 1.1360× | 5.756 | yes |
| **GM** | | | | | | | **0.6792×** | | |
