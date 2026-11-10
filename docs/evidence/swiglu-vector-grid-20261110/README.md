# SwiGLU launch grids on Ascend950PR_9599

All 20 published Season 2 SwiGLU cases passed the full golden check for cap64, auto and the regenerated UB-height variant. Cap64 and auto share tile shapes and math. The UB variant starts from auto and selects the height of padded short rows from the active UB capacity. Cap64 limits each launch to 64 independent work items. Auto uses the vector resources reported for the active context, bounded by the work available to that launch. A separately owned row tail launches one block.

## Measured grids and device time

The auto and UB grids report the `block_num` argument recorded on return from each successful PyASC `runtime.launch_kernel` call. All four invocations per case had the same grid. The cap64 grid comes from the retained dispatch code, input geometry and profiler task names. A grid of `72+1` means two successive launches.

Speed relative to CANNBench is `baseline_perf_us / candidate_us`. The auto vs cap64 column uses `cap64_us / auto_us`. The UB vs auto column uses `auto_us / ub_us`. Tile rows gives the physical height of the main UB-variant launch. The last row uses the geometric mean of the 20 unrounded ratios. CANNBench publishes a scoring-capped baseline for the 950pr family with status `pending_npu_revalidation`. [Pinned CANNBench metadata](https://gitcode.com/cann/cann-bench/blob/72fb4064d80bcbd6195a100cd4bbabac7a4f8817/tasks/metadata/950pr.json)

| Case | Input shape | Cap64 grid | Auto grid | CANNBench µs | Cap64 µs | Auto µs | Cap64 vs CANNBench (×) | Auto vs CANNBench (×) | Auto vs cap64 (×) | UB grid | Tile rows | UB µs | UB vs CANNBench (×) | UB vs auto (×) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 × 2048 | 64 | 64 | 6.380 | 9.965 | 10.763 | 0.64024 | 0.59277 | 0.9259 | 64 | 16 | 10.411 | 0.61281 | 1.0338 |
| 2 | 2048 × 4096 | 64 | 72 | 23.650 | 38.081 | 38.892 | 0.62104 | 0.60809 | 0.9791 | 72 | 16 | 38.143 | 0.62004 | 1.0196 |
| 3 | 4096 × 8192 | 64 | 72 | 48.590 | 150.123 | 147.905 | 0.32367 | 0.32852 | 1.0150 | 72 | 16 | 148.123 | 0.32804 | 0.9985 |
| 4 | 8192 × 16384 | 64 | 72 | 241.490 | 591.541 | 539.494 | 0.40824 | 0.44762 | 1.0965 | 72 | 1 | 538.953 | 0.44807 | 1.0010 |
| 5 | 2039 × 65520 | 64+1 | 72+1 | 522.100 | 1144.921 | 1150.281 | 0.45601 | 0.45389 | 0.9953 | 72+1 | 16 | 1161.036 | 0.44968 | 0.9907 |
| 6 | 1022 × 2047 | 64 | 72 | 6.200 | 12.065 | 11.190 | 0.51388 | 0.55407 | 1.0782 | 72 | 1 | 10.552 | 0.58757 | 1.0605 |
| 7 | 1009 × 2016 | 63+1 | 63+1 | 6.500 | 23.766 | 23.435 | 0.27350 | 0.27736 | 1.0141 | 63+1 | 16 | 23.607 | 0.27534 | 0.9927 |
| 8 | 1538 × 1537 | 64 | 72 | 8.540 | 13.416 | 11.756 | 0.63655 | 0.72644 | 1.1412 | 72 | 1 | 11.805 | 0.72342 | 0.9958 |
| 9 | 363 × 367 × 14 | 64+1 | 72+1 | 6.782 | 110.477 | 95.993 | 0.06139 | 0.07065 | 1.1509 | 72+1 | 128 | 31.520 | 0.21516 | 3.0455 |
| 10 | 2049 × 1024 | 64+1 | 72+1 | 6.180 | 14.946 | 15.160 | 0.41349 | 0.40765 | 0.9859 | 72+1 | 16 | 15.023 | 0.41137 | 1.0091 |
| 11 | 3 × 7 × 13 × 1018 | 17+1 | 17+1 | 2.021 | 12.184 | 11.186 | 0.16587 | 0.18067 | 1.0892 | 17+1 | 16 | 11.939 | 0.16928 | 0.9369 |
| 12 | 1000003 × 2 | 64+1 | 72+1 | 7.273 | 1272.485 | 1107.532 | 0.00572 | 0.00657 | 1.1489 | 72+1 | 128 | 445.974 | 0.01631 | 2.4834 |
| 13 | 11 × 13 × 16 × 67 | 8+1 | 8+1 | 1.500 | 13.627 | 13.610 | 0.11008 | 0.11021 | 1.0012 | 8+1 | 16 | 13.694 | 0.10954 | 0.9939 |
| 14 | 3 × 7 × 11 × 13 × 1012 | 64+1 | 72+1 | 7.540 | 23.042 | 22.275 | 0.32723 | 0.33850 | 1.0344 | 72+1 | 16 | 22.657 | 0.33279 | 0.9831 |
| 15 | 512 × 4096 | 32 | 32 | 7.780 | 18.810 | 18.529 | 0.41361 | 0.41988 | 1.0152 | 32 | 16 | 18.151 | 0.42863 | 1.0208 |
| 16 | 254 × 16383 | 64 | 72 | 8.790 | 21.173 | 19.477 | 0.41515 | 0.45130 | 1.0871 | 72 | 1 | 18.883 | 0.46550 | 1.0315 |
| 17 | 4096 × 1023 | 64 | 72 | 8.760 | 19.352 | 17.641 | 0.45267 | 0.49657 | 1.0970 | 72 | 1 | 17.237 | 0.50821 | 1.0234 |
| 18 | 2 × 1023 × 4096 | 64+1 | 72+1 | 23.630 | 56.562 | 57.033 | 0.41777 | 0.41432 | 0.9917 | 72+1 | 16 | 56.648 | 0.41714 | 1.0068 |
| 19 | 4 × 510 × 4097 | 64 | 72 | 14.280 | 46.400 | 41.973 | 0.30776 | 0.34022 | 1.1055 | 72 | 4 | 41.736 | 0.34215 | 1.0057 |
| 20 | 2 × 3 × 17 × 512 × 100 | 64 | 72 | 19.560 | 33.595 | 31.090 | 0.58223 | 0.62914 | 1.0806 | 72 | 16 | 30.661 | 0.63794 | 1.0140 |
| **Geometric mean** | | | | | | | **0.28294** | **0.29703** | **1.04980** | | | | **0.33050** | **1.11267** |

[Full comparison CSV](comparison.csv) includes the earlier reference, parent, archived and generated candidates, three device-time samples for cap64, auto and UB, and the observed auto and UB launch counts in call order.

## Short-row result

With 1,000,003 BF16 rows and one useful output element per row, case 12 went from 1,107.532 to 445.974 µs (2.483× the auto speed, 0.01631× CANNBench). Both versions launch `72+1`. The taller tile reduces the full row groups from 62,500 to 7,812. A separate launch handles the last 67 rows. Each local row still has 16 physical lanes for one useful output element. The standalone pilot passed at 446.172 µs. The table uses the subsequent full-series run.

Case 9 has seven useful BF16 outputs per row. Its time changed from 95.993 to 31.520 µs (3.045×). Those two cases enter the short-row route. Across the other 18, the geometric mean of auto time divided by UB time was 1.0062× in separate measurement series.

OpenCode 1.2.19 with GLM-5.3 regenerated the host dispatch from the measured auto source using the updated short-row skill. Its first executable estimate selected 512 rows and the compiler reported 655,360 UB bytes, above the available 253,952 bytes. An isolated calibration confirmed 163,840 bytes for 128 rows. The repair prompt supplied this measured 80-byte cost per padded lane and requested a scoped edit to the generated selector. The measured snapshot incorporates that OpenCode edit and omits the superseded capacity clamp. The module docstring describes the pre-repair estimate.

The executed selector queries the active SoC after platform setup, subtracts 32 KiB, divides by the measured cost per physical row, bounds the height by useful rows per AIV and rounds down to a power of two. On case 12, modeled capacities of 131,072, 253,952 and 524,288 bytes select heights of 64, 128 and 256 at 72 AIVs. Only 253,952 bytes was the hardware capacity. The warmup captured the actual height of 128 and 163,840 compiled UB bytes for each of the two launches. [Measured source](ub/swi_glu.py) · [Pilot and 20-case measurements](ub/measurements.json) · [Profiler harness](measure.py)

## Measurement and provenance

- Ascend950PR_9599, device 0, CANN 9.2.0, PyASC `6b3447b45f1f7bd71fa8a23d98b1e76572b3f475`, Python 3.10.14 and PyTorch 2.5.1+cpu. The runtime reported 36 AI cores and 72 vector cores. Device and current-thread vector limits were both 72.
- OpenCode 1.2.19 with GLM-5.3 generated the tiling candidate. The cap64 and auto variants change its host launch policy. Auto reuses the skill template's `vector_blocks` helper. The constant eight in the wide-path tile-width calculation and dispatch threshold is common to both variants.
- Each case uses a separate process and the CANNBench data generator seeded with the case ID. The first call warmed the kernel and produced the output checked against the full golden tensor with `torch.testing.assert_close(equal_nan=True)`, including shape and dtype. The next three calls ran under the PyASC device profiler. Each sample sums the AI_VECTOR_CORE durations within one operator call. The per-case time is their median. Transfers, host dispatch, resource queries and compilation are outside this device metric. Cap64 was measured in the preceding series.
- For 0, 3 and 1,000 work items the template returned 0, 3 and 72 on the NPU. In the pinned CaModel image the simulator driver reported Ascend950PR_9589 and the helper returned 0, 3 and 64 with Ascend950PR_9599 requested. Both backends passed the template's numerical checks at 8,192, 8,191 and 131,072 elements.

The [measured auto candidate](auto/swi_glu.py) keeps the generator's original docstring, which describes the initial eight-block policy. To use the snapshot with this PyASC pin, place the stack's [runtime helper](../../../integrations/cannbench/smoke-kernel/_pyasc_runtime.py) beside it and put the [kernel template](../../../skills/asctile-codegen-workflow/templates/kernel-template.py) on the Python path as `grid_template.py`, replacing `from asc.experimental import asctile` with `import asctile` for this runtime.

| Measured source | SHA-256 |
|---|---|
| OpenCode output | `f63e111e2f158bff86835517f56ef19b4f0bb43ba92a26a9dda117130bbb9a1f` |
| G64, wide branch only | `83e2707d96f2105c5d2195ee1841e043165e26e3e066ae5d3414394f29bd4070` |
| Cap64, all branches | `87069b0d42693f85239af16233e26dfe80a967ded1fc79038ae342f57b97ce11` |
| Auto, all branches | `9155e3a6979cb35de23fa2be010b8c18bba7afd52cf201669dab84a0b2f4f8c1` |
| Runtime-specific grid template | `cd51051e19cc813a5881636076aa4e1759e60486a6767a46a20006778a329ce9` |
| OpenCode UB-height candidate | `1a511da8b9e22195f82ec309d3ab0a7fdd6c456575db2ad4327772a2d298ae7d` |
