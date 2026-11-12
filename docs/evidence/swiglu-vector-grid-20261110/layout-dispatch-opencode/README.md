# SwiGLU shape dispatch generated from scratch on David

OpenCode generated three SwiGLU kernels from the public task and the updated skills. On Ascend950PR_9599, the candidate passed all 20 full-output checks, beat CANNBench in 14 cases, and reached **0.9906×** geometric mean relative to CANNBench. Seven full-series medians met the saved archive thresholds. The previous independent generation reached **0.6792×** and met one archive threshold. The generator model and the skill context both changed between the two runs. [Previous independent generation](../clean-opencode/README.md)

## Which kernels ran

| Path in the generated dispatcher | Profiler task | Cases |
|---|---|---|
| Short BF16 halves of at most 32 bytes, with an aligned slab that fits UB. Read consecutive source rows once and gather gate and value with two index maps. | `_swiglu_slab` | 9, 12 |
| A complete row tile fits UB at a height selected from the DMA target and data type. Copy the two strided half-rows in batches. | `_swiglu_rows` | 1, 2, 3, 7, 10, 11, 13, 14, 15, 18, 20 |
| Stream each half in contiguous chunks when the row-tile selector finds no fit. | `_swiglu_flat` | 4, 5, 6, 8, 16, 17, 19 |

The operator derives the row count and half-width from the shape and split axis. The active context reported 72 vector blocks and 253,952 bytes of UB. All 20 grids matched the previous independent run: 69 blocks for case 11 and 72 elsewhere. Each launch used a single grid, including the two pilots with a partial final slab. [Generated dispatcher and kernels](swi_glu.py) · [Recorded launches](measurements.json)

| BF16 pilot | Input shape | Rows per tile | Unroll | Blocks | Pilot µs | Full series µs | Previous full series µs | CANNBench µs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 9 | 363 × 367 × 14 | 512 | 2 | 72 | 6.385 | 6.414 | 21.634 | 6.782 |
| 12 | 1,000,003 × 2 | 4,096 | 2 | 72 | 6.851 | 6.716 | 393.194 | 7.273 |

## Reproduction and timing

OpenCode 1.2.19 generated the timed candidate with `cannbot/DeepSeek-V4.1-Flash`. The previous independent run used `cannbot/GLM-5.3`. To reproduce the final prompt from the repository root, run `python3 docs/evidence/swiglu-vector-grid-20261110/layout-dispatch-opencode/context.py . 58536eb41eea955c5d2b1f5b9d898cbb5f04c872`. It assembles the public task files, the scoped skill sections (including their reusable flat-kernel template) and the same neutral request. The input excluded earlier candidates, measured route assignments and timings. OpenCode had no tool access. Removing the surrounding Markdown fence yielded SHA-256 `c9de95b2c877435ac160da27f6069602a87e857768f2570fe09f9769e4a4a720`, which matches the operator checked before each pilot and each full-series case. [Request](../clean-opencode/request.txt) · [Prompt manifest](generation-inputs.json) · [OpenCode response](opencode.jsonl)

Two initial GLM calls exhausted their output budgets before emitting source. A subsequent GLM candidate returned through its CPU branch on both pilots. An earlier Flash candidate called `input.to("npu")` on the evaluator's CPU-backed tensors and failed before launch. The updated skill documents the public runtime's tensor and launch contract. A fresh Flash call then generated the measured source. [Attempt records and candidate hashes](generation-inputs.json)

The pilots ran first, followed by the 20-case suite in Docker with the archive-bundled PyASC and CANN 9.2.0. Each fresh process checked the complete output and profiled three warmed NPU calls. The table uses the median device time, confirmed by AI vector-core tasks. Speed is the published CANNBench time for 950PR divided by the full-suite median. GM uses all 20 unrounded ratios. An archive pass requires that median to be at most the saved replay median of `job_f9f0facb7d6b` on David. Both pilots are excluded from the GM and the threshold count. [Full samples and grid traces](measurements.json) · [Archive replay](../native/archive-measurements.json) · [Comparison CSV with per-case tile and unroll](../comparison.csv)

## All 20 cases, slowest to fastest relative to CANNBench

| Case | Measured family | Grid | OpenCode µs | Previous OpenCode µs | CANNBench µs | CANNBench / OpenCode | CANNBench / previous | Archive threshold µs | Met |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 13 | 2D | 72 | 3.008 | 3.186 | 1.500 | 0.4987× | 0.4708× | 2.550 | no |
| 11 | 2D | 69 | 3.225 | 3.282 | 2.021 | 0.6267× | 0.6158× | 3.010 | no |
| 8 | flat | 72 | 9.044 | 9.391 | 8.540 | 0.9443× | 0.9094× | 8.488 | no |
| 4 | flat | 72 | 252.406 | 269.533 | 241.490 | 0.9568× | 0.8960× | 243.929 | no |
| 19 | flat | 72 | 14.611 | 19.808 | 14.280 | 0.9773× | 0.7209× | 13.411 | no |
| 5 | flat | 72 | 530.810 | 542.537 | 522.100 | 0.9836× | 0.9623× | 523.609 | no |
| 2 | 2D | 72 | 23.546 | 24.678 | 23.650 | 1.0044× | 0.9583× | 23.782 | yes |
| 16 | flat | 72 | 8.751 | 10.687 | 8.790 | 1.0045× | 0.8225× | 8.310 | no |
| 17 | flat | 72 | 8.716 | 10.156 | 8.760 | 1.0050× | 0.8625× | 8.398 | no |
| 3 | 2D | 72 | 48.309 | 62.425 | 48.590 | 1.0058× | 0.7784× | 45.645 | no |
| 15 | 2D | 72 | 7.550 | 8.097 | 7.780 | 1.0305× | 0.9608× | 7.520 | no |
| 18 | 2D | 72 | 22.839 | 24.337 | 23.630 | 1.0346× | 0.9709× | 23.214 | yes |
| 9 | reindex | 72 | 6.414 | 21.634 | 6.782 | 1.0574× | 0.3135× | 6.385 | no |
| 12 | reindex | 72 | 6.716 | 393.194 | 7.273 | 1.0829× | 0.0185× | 6.345 | no |
| 14 | 2D | 72 | 6.887 | 8.325 | 7.540 | 1.0948× | 0.9057× | 7.385 | yes |
| 20 | 2D | 72 | 17.857 | 18.541 | 19.560 | 1.0954× | 1.0550× | 15.734 | no |
| 7 | 2D | 72 | 5.651 | 6.848 | 6.500 | 1.1502× | 0.9492× | 6.687 | yes |
| 6 | flat | 72 | 5.374 | 7.104 | 6.200 | 1.1537× | 0.8727× | 6.134 | yes |
| 10 | 2D | 72 | 5.175 | 5.440 | 6.180 | 1.1942× | 1.1360× | 5.756 | yes |
| 1 | 2D | 72 | 4.951 | 6.162 | 6.380 | 1.2886× | 1.0354× | 5.016 | yes |
| **GM** | | | | | | **0.9906×** | **0.6792×** | | |
