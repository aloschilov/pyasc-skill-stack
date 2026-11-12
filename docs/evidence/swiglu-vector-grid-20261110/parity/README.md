# SwiGLU on David: fixed-candidate comparisons

## Qualified fixed candidate

The third complete run of round eleven met the saved archive time for all 20 cases on Ascend950PR_9599. The same source met 18/20 in each of its first two runs. Each case’s median across the three run medians also met its saved time. All 60 full-output golden checks passed. Each run used a fresh process per case and three warmed device samples per median. The final run's CANNBench / auto geometric mean was 1.0482×, with 18 cases above 1×.

The integrated implementation uses the OpenCode-generated even-grid helper and manually integrated, hardware-qualified kernels and dispatch rules. Case 15 uses eight whole FP32 rows per block on a 64-block grid, unroll four and 98,304 bytes of UB. Case 9 uses full signed gate indices at 928 rows and 72 blocks. Case 12 uses 4,096-row tiles and 64 blocks in the qualified operator. The 62-block case-12 result below is a separate control.

[Qualified operator](qualified/swi_glu.py) · [Matching runtime](qualified/_pyasc_runtime.py) · [Final run table](#two-repeats-of-round-eleven) · [First run profiles](round11.json) · [Repeat profiles](round11-repeats.json)

## First integrated candidate

The reference for the per-case threshold is the saved median from `job_f9f0facb7d6b`. The paired archive column is a fresh measurement of the same source. The single candidate was held fixed across the suite. Each median is the middle of three warmed device times after a full-output golden check. Candidate and archive order alternated by case. CANNBench times use the published 950PR `baseline_perf_us`.

| Case | Candidate (µs) | CANNBench (µs) | CANNBench / candidate | Saved archive (µs) | Paired archive (µs) |
|---:|---:|---:|---:|---:|---:|
| 13 | 2.584 | 1.500 | 0.5805× | 2.550 | 2.663 |
| 11 | 2.893 | 2.021 | 0.6986× | 3.010 | 3.009 |
| 7 | 7.348 | 6.500 | 0.8846× | 6.687 | 5.622 |
| 16 | 9.420 | 8.790 | 0.9331× | 8.310 | 8.890 |
| 9 | 6.935 | 6.782 | 0.9779× | 6.385 | 6.407 |
| 4 | 245.677 | 241.490 | 0.9830× | 243.929 | 244.040 |
| 8 | 8.653 | 8.540 | 0.9869× | 8.488 | 7.916 |
| 5 | 522.352 | 522.100 | 0.9995× | 523.609 | 523.036 |
| 2 | 23.643 | 23.650 | 1.0003× | 23.782 | 23.668 |
| 18 | 22.724 | 23.630 | 1.0399× | 23.214 | 23.039 |
| 3 | 46.170 | 48.590 | 1.0524× | 45.645 | 44.815 |
| 19 | 13.498 | 14.280 | 1.0579× | 13.411 | 13.255 |
| 12 | 6.872 | 7.273 | 1.0584× | 6.345 | 6.696 |
| 15 | 6.985 | 7.780 | 1.1138× | 7.520 | 7.479 |
| 14 | 6.756 | 7.540 | 1.1160× | 7.385 | 7.613 |
| 17 | 7.695 | 8.760 | 1.1384× | 8.398 | 7.899 |
| 20 | 16.420 | 19.560 | 1.1912× | 15.734 | 17.146 |
| 6 | 5.044 | 6.200 | 1.2292× | 6.134 | 5.986 |
| 10 | 4.833 | 6.180 | 1.2787× | 5.756 | 5.500 |
| 1 | 4.783 | 6.380 | 1.3339× | 5.016 | 4.701 |
| **GM** | | | **1.0161×** | | |

All 20 golden checks passed. Ten candidate medians met the saved threshold and eleven met the paired archive median. The earlier FP16 direct candidate met eight saved thresholds.

[Complete profiler records and source hashes](round1.json) · [Patch against fp16-direct/swi_glu.py](round1.patch)

## Second integrated candidate

One source was held fixed across all 20 golden checks. The published CANNBench reference is identical to round one. The archived submission was measured during round one. Each candidate time below is the median of three warmed launches in round two.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.335 | 1.500 | 0.6424× |
| 11 | 2.820 | 2.021 | 0.7167× |
| 9 | 7.292 | 6.782 | 0.9301× |
| 5 | 542.489 | 522.100 | 0.9624× |
| 12 | 7.434 | 7.273 | 0.9783× |
| 8 | 8.651 | 8.540 | 0.9872× |
| 4 | 244.445 | 241.490 | 0.9879× |
| 2 | 23.323 | 23.650 | 1.0140× |
| 15 | 7.580 | 7.780 | 1.0264× |
| 14 | 7.290 | 7.540 | 1.0343× |
| 18 | 22.773 | 23.630 | 1.0376× |
| 17 | 8.409 | 8.760 | 1.0417× |
| 3 | 45.404 | 48.590 | 1.0702× |
| 19 | 13.115 | 14.280 | 1.0888× |
| 7 | 5.756 | 6.500 | 1.1293× |
| 16 | 7.754 | 8.790 | 1.1336× |
| 6 | 5.461 | 6.200 | 1.1353× |
| 20 | 15.824 | 19.560 | 1.2361× |
| 1 | 4.919 | 6.380 | 1.2970× |
| 10 | 4.641 | 6.180 | 1.3316× |
| **GM** | | | **1.0255×** |

Twelve medians met the saved archive thresholds. The eight remaining cases were 4, 5, 8, 9, 12, 15, 17 and 20. In particular, removing bounds from the full-chunk loop helped the BF16 whole-row case. Processing the remainder of each FP32 row in a second pass increased latency on the many-row ragged input in case 5.

[Raw measurements and source hash](round2.json) · [Patch against round one](round2.patch)

## Third integrated candidate

All 20 full outputs passed the golden check on David. The table uses one
source hash across all cases and the median of three warmed device tasks
per case. Twelve medians met the saved archive threshold. Cases 3, 8, 9,
12, 14, 15, 17 and 20 were slower than the saved archive. Cases 4 and 5
now met the threshold. Trial times from other kernels are excluded.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.482 | 1.500 | 0.6044× |
| 11 | 2.892 | 2.021 | 0.6988× |
| 4 | 242.399 | 241.490 | 0.9962× |
| 8 | 8.514 | 8.540 | 1.0031× |
| 5 | 519.782 | 522.100 | 1.0045× |
| 14 | 7.471 | 7.540 | 1.0092× |
| 2 | 23.364 | 23.650 | 1.0122× |
| 15 | 7.657 | 7.780 | 1.0161× |
| 17 | 8.600 | 8.760 | 1.0186× |
| 18 | 22.695 | 23.630 | 1.0412× |
| 9 | 6.409 | 6.782 | 1.0582× |
| 3 | 45.793 | 48.590 | 1.0611× |
| 16 | 8.273 | 8.790 | 1.0625× |
| 12 | 6.731 | 7.273 | 1.0805× |
| 19 | 13.176 | 14.280 | 1.0838× |
| 10 | 5.504 | 6.180 | 1.1228× |
| 7 | 5.766 | 6.500 | 1.1273× |
| 20 | 17.066 | 19.560 | 1.1461× |
| 6 | 5.222 | 6.200 | 1.1873× |
| 1 | 4.715 | 6.380 | 1.3531× |
| **GM** | | | **1.0211×** |

[Raw measurements and source hash](round3.json) · [Patch against round two](round3.patch)

## Fourth integrated candidate

All 20 full-output checks passed with one source. Sixteen medians met the saved
archive threshold. The remaining cases exceeded it by 0.519 µs (case 1),
0.372 µs (9), 0.420 µs (16) and 0.056 µs (20). Case 12 used signed int16
indices for both BF16 gathers and took 6.109 µs. The 72-AIV device launched
64 blocks for the aligned BF16 full-row stream in case 3. Case 17 balanced
the FP16 chunk count across six waves and used 4,864 lanes. The candidate
includes manually integrated changes following the generation and controls.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.465 | 1.500 | 0.6085× |
| 11 | 2.817 | 2.021 | 0.7174× |
| 4 | 242.403 | 241.490 | 0.9962× |
| 5 | 522.355 | 522.100 | 0.9995× |
| 2 | 23.618 | 23.650 | 1.0014× |
| 9 | 6.757 | 6.782 | 1.0037× |
| 16 | 8.730 | 8.790 | 1.0069× |
| 8 | 8.364 | 8.540 | 1.0210× |
| 18 | 22.709 | 23.630 | 1.0406× |
| 17 | 8.292 | 8.760 | 1.0564× |
| 19 | 13.233 | 14.280 | 1.0791× |
| 3 | 44.971 | 48.590 | 1.0805× |
| 10 | 5.688 | 6.180 | 1.0865× |
| 15 | 7.031 | 7.780 | 1.1065× |
| 7 | 5.856 | 6.500 | 1.1100× |
| 14 | 6.779 | 7.540 | 1.1123× |
| 6 | 5.541 | 6.200 | 1.1189× |
| 1 | 5.535 | 6.380 | 1.1527× |
| 12 | 6.109 | 7.273 | 1.1905× |
| 20 | 15.790 | 19.560 | 1.2388× |
| **GM** | | | **1.0246×** |

[Single-build profiler records](round4.json) · [Patch against round three](round4.patch) · [Randomized independent control groups](round4-controls.json)

## Fifth integrated candidate

All 20 full golden checks passed. Seventeen medians met the archived
per-case thresholds. Cases 1, 9 and 16 now measured 4.558, 6.157 and
7.845 µs. Case 8 exceeded the saved archive by 0.137 µs, case 12 by
0.655 µs and case 20 by 0.015 µs. The tables use the original three
warmed launches in a fresh process for each case. All timing controls
have separate records.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.392 | 1.500 | 0.6271× |
| 11 | 2.717 | 2.021 | 0.7438× |
| 8 | 8.625 | 8.540 | 0.9901× |
| 4 | 241.889 | 241.490 | 0.9984× |
| 5 | 522.338 | 522.100 | 0.9995× |
| 2 | 23.374 | 23.650 | 1.0118× |
| 12 | 7.000 | 7.273 | 1.0390× |
| 18 | 22.487 | 23.630 | 1.0508× |
| 17 | 8.159 | 8.760 | 1.0737× |
| 3 | 45.035 | 48.590 | 1.0789× |
| 19 | 13.027 | 14.280 | 1.0962× |
| 15 | 7.096 | 7.780 | 1.0964× |
| 9 | 6.157 | 6.782 | 1.1015× |
| 7 | 5.813 | 6.500 | 1.1182× |
| 16 | 7.845 | 8.790 | 1.1205× |
| 14 | 6.621 | 7.540 | 1.1388× |
| 6 | 5.173 | 6.200 | 1.1985× |
| 20 | 15.749 | 19.560 | 1.2420× |
| 10 | 4.854 | 6.180 | 1.2732× |
| 1 | 4.558 | 6.380 | 1.3997× |
| **GM** | | | **1.0557×** |

[Single-build profiles](round5.json) · [Source patch against round four](round5.patch) · [Control groups](round5-controls.json)

## Sixth integrated candidate

All 20 cases passed the full-output check. The case 8 FP32 stream used four full waves and a one-element tail in the same launch. Case 20 interleaved the block order on full 64-row FP32 slabs. Sixteen medians met the saved archive times. Cases 10, 12, 15 and 19 exceeded them by 0.337, 0.425, 0.198 and 0.041 µs.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.468 | 1.500 | 0.6078× |
| 11 | 2.971 | 2.021 | 0.6802× |
| 4 | 241.792 | 241.490 | 0.9988× |
| 5 | 521.996 | 522.100 | 1.0002× |
| 15 | 7.718 | 7.780 | 1.0080× |
| 10 | 6.093 | 6.180 | 1.0143× |
| 2 | 23.268 | 23.650 | 1.0164× |
| 14 | 7.371 | 7.540 | 1.0229× |
| 18 | 22.747 | 23.630 | 1.0388× |
| 19 | 13.452 | 14.280 | 1.0616× |
| 3 | 45.396 | 48.590 | 1.0704× |
| 12 | 6.770 | 7.273 | 1.0743× |
| 8 | 7.909 | 8.540 | 1.0798× |
| 9 | 6.203 | 6.782 | 1.0933× |
| 16 | 8.031 | 8.790 | 1.0945× |
| 7 | 5.840 | 6.500 | 1.1130× |
| 17 | 7.732 | 8.760 | 1.1330× |
| 6 | 5.129 | 6.200 | 1.2088× |
| 20 | 15.500 | 19.560 | 1.2619× |
| 1 | 4.528 | 6.380 | 1.4090× |
| **GM** | | | **1.0339×** |

[Single-build profiles](round6.json) · [Patch against round five](round6.patch) · [Control timings](round6-controls.json)

### Independent repeats of round six

Two complete repeats ran each source and case in a fresh process. Case order was shuffled with a fixed seed and the order of the two sources was reversed on the second repeat. All 80 full-output checks passed. The first repeat met 17 of 20 saved archive medians with a CANNBench-normalized GM of 1.0442×. The second met 18 of 20 with a GM of 1.0408×. Case 12 missed the saved threshold in the original run and both repeats (6.770, 6.520, 6.882 µs, archive 6.345 µs). The following table contains only the second repeat.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.402 | 1.500 | 0.6245× |
| 11 | 2.787 | 2.021 | 0.7252× |
| 9 | 7.074 | 6.782 | 0.9587× |
| 4 | 242.009 | 241.490 | 0.9979× |
| 5 | 522.610 | 522.100 | 0.9990× |
| 2 | 23.352 | 23.650 | 1.0128× |
| 18 | 22.826 | 23.630 | 1.0352× |
| 8 | 8.081 | 8.540 | 1.0568× |
| 12 | 6.882 | 7.273 | 1.0568× |
| 3 | 45.300 | 48.590 | 1.0726× |
| 14 | 7.013 | 7.540 | 1.0751× |
| 15 | 7.230 | 7.780 | 1.0761× |
| 19 | 13.188 | 14.280 | 1.0828× |
| 16 | 8.116 | 8.790 | 1.0830× |
| 10 | 5.567 | 6.180 | 1.1101× |
| 7 | 5.829 | 6.500 | 1.1151× |
| 17 | 7.773 | 8.760 | 1.1270× |
| 6 | 5.078 | 6.200 | 1.2210× |
| 20 | 15.646 | 19.560 | 1.2502× |
| 1 | 4.502 | 6.380 | 1.4171× |
| **GM** | | | **1.0408×** |

[All repeat samples, grids and source hashes](round6-repeats.json)

## Seventh integrated candidate

The dense BF16 pair route selected 4,096 rows per tile, two stages and an even/odd block ordering on 64 AIVs. Case 12 passed the full golden in 6.018 µs. All 20 cases passed correctness. The original run met 17 of 20 saved archive times. Cases above threshold (case, excess in µs): 9 (+0.520), 10 (+0.052), 14 (+0.080).

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.443 | 1.500 | 0.6140× |
| 11 | 2.879 | 2.021 | 0.7020× |
| 9 | 6.905 | 6.782 | 0.9822× |
| 4 | 241.895 | 241.490 | 0.9983× |
| 5 | 521.995 | 522.100 | 1.0002× |
| 2 | 23.426 | 23.650 | 1.0096× |
| 14 | 7.465 | 7.540 | 1.0100× |
| 18 | 22.703 | 23.630 | 1.0408× |
| 10 | 5.808 | 6.180 | 1.0640× |
| 3 | 45.636 | 48.590 | 1.0647× |
| 16 | 8.160 | 8.790 | 1.0772× |
| 19 | 13.186 | 14.280 | 1.0830× |
| 17 | 7.918 | 8.760 | 1.1063× |
| 15 | 7.011 | 7.780 | 1.1097× |
| 8 | 7.689 | 8.540 | 1.1107× |
| 7 | 5.852 | 6.500 | 1.1107× |
| 6 | 5.449 | 6.200 | 1.1378× |
| 12 | 6.018 | 7.273 | 1.2085× |
| 20 | 15.385 | 19.560 | 1.2714× |
| 1 | 4.578 | 6.380 | 1.3936× |
| **GM** | | | **1.0399×** |

[Fixed-build profiles](round7.json) · [Source patch against round six](round7.patch) · [Independent control trials](round7-controls.json)

### Two repeats of round seven

The first repeat met 18 saved thresholds, and the second met 15. All 40 repeated full-output checks passed. In the second repeat, the excess times for cases 1, 9, 12, 14 and 20 were 0.361, 0.700, 0.216, 0.167, 0.090 µs, respectively. Case 9 exceeded the saved threshold in each run. Across three run-level medians per case, the median exceeded the saved threshold for cases 9, 14.

Latest complete run, ordered by CANNBench / auto.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.547 | 1.500 | 0.5889× |
| 11 | 2.809 | 2.021 | 0.7195× |
| 9 | 7.085 | 6.782 | 0.9572× |
| 14 | 7.552 | 7.540 | 0.9984× |
| 5 | 522.318 | 522.100 | 0.9996× |
| 2 | 23.482 | 23.650 | 1.0072× |
| 8 | 8.361 | 8.540 | 1.0214× |
| 4 | 232.567 | 241.490 | 1.0384× |
| 18 | 22.605 | 23.630 | 1.0453× |
| 17 | 8.291 | 8.760 | 1.0566× |
| 3 | 45.354 | 48.590 | 1.0713× |
| 15 | 7.234 | 7.780 | 1.0755× |
| 19 | 13.126 | 14.280 | 1.0879× |
| 12 | 6.561 | 7.273 | 1.1085× |
| 7 | 5.828 | 6.500 | 1.1153× |
| 16 | 7.668 | 8.790 | 1.1463× |
| 1 | 5.377 | 6.380 | 1.1865× |
| 6 | 5.029 | 6.200 | 1.2328× |
| 20 | 15.824 | 19.560 | 1.2361× |
| 10 | 4.755 | 6.180 | 1.2997× |
| **GM** | | | **1.0351×** |

[All repeated timings and grids](round7-repeats.json)

## Eighth fixed candidate

For a two-wave BF16 slab, the new kernel assigns consecutive intervals of tiles to each AIV. For strided FP16 with a padded half-width, the selector balances rows across waves within the DMA and UB bounds. OpenCode generated the host helper on David. Integration corrected its fallback returns to the incoming tile height and restricted the exact-grid guard to aligned half-rows. The function passed 12,600 host geometry comparisons.

The first hardware run passed all 20 golden checks and met 18 saved archive thresholds. Its two excess times were 0.346 µs on case 2 and 0.222 µs on case 12. Case 9 selected 928 rows, two stages and 72 blocks, taking 6.034 µs. Case 14 selected 11 rows, two stages and 72 blocks, taking 6.263 µs.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.404 | 1.500 | 0.6240× |
| 11 | 2.862 | 2.021 | 0.7061× |
| 2 | 24.128 | 23.650 | 0.9802× |
| 5 | 523.001 | 522.100 | 0.9983× |
| 4 | 236.799 | 241.490 | 1.0198× |
| 18 | 22.687 | 23.630 | 1.0416× |
| 3 | 45.495 | 48.590 | 1.0680× |
| 6 | 5.803 | 6.200 | 1.0684× |
| 16 | 8.122 | 8.790 | 1.0822× |
| 19 | 13.193 | 14.280 | 1.0824× |
| 7 | 5.934 | 6.500 | 1.0954× |
| 8 | 7.794 | 8.540 | 1.0957× |
| 15 | 7.082 | 7.780 | 1.0986× |
| 12 | 6.567 | 7.273 | 1.1075× |
| 17 | 7.869 | 8.760 | 1.1132× |
| 9 | 6.034 | 6.782 | 1.1240× |
| 14 | 6.263 | 7.540 | 1.2039× |
| 20 | 15.634 | 19.560 | 1.2511× |
| 10 | 4.852 | 6.180 | 1.2737× |
| 1 | 4.689 | 6.380 | 1.3606× |
| **GM** | | | **1.0550×** |

[Full hardware profiles](round8.json) · [Source patch](round8.patch) · [Tile controls](round8-tile-controls.json) · [Schedule controls](round8-schedule-controls.json)

### Two repeats of round eight

All 60 full-output checks passed across three independent runs with identical operator source. The first repeat met 16 saved thresholds and the second met 19. The first repeat missed cases 12, 15, 19 and 20. In the second repeat only case 6 exceeded its saved threshold, at 6.314 µs against 6.134 µs. Across runs, case 9 took 6.034, 6.017 and 6.107 µs and case 14 took 6.263, 6.325 and 6.346 µs. Case 12 exceeded its saved threshold in the first two runs.

Latest complete run, ordered by CANNBench / auto.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.536 | 1.500 | 0.5915× |
| 11 | 2.774 | 2.021 | 0.7286× |
| 6 | 6.314 | 6.200 | 0.9819× |
| 4 | 241.396 | 241.490 | 1.0004× |
| 5 | 520.430 | 522.100 | 1.0032× |
| 2 | 23.418 | 23.650 | 1.0099× |
| 18 | 22.864 | 23.630 | 1.0335× |
| 8 | 8.196 | 8.540 | 1.0420× |
| 17 | 8.322 | 8.760 | 1.0526× |
| 3 | 45.079 | 48.590 | 1.0779× |
| 16 | 8.148 | 8.790 | 1.0788× |
| 19 | 13.233 | 14.280 | 1.0791× |
| 15 | 7.011 | 7.780 | 1.1097× |
| 9 | 6.107 | 6.782 | 1.1105× |
| 7 | 5.767 | 6.500 | 1.1271× |
| 10 | 5.389 | 6.180 | 1.1468× |
| 14 | 6.346 | 7.540 | 1.1882× |
| 12 | 5.915 | 7.273 | 1.2296× |
| 20 | 15.689 | 19.560 | 1.2467× |
| 1 | 4.521 | 6.380 | 1.4112× |
| **GM** | | | **1.0468×** |

[Both repeat profiles](round8-repeats.json)

## Ninth fixed candidate

The operator uses four groups of starting tiles for dense BF16 pairs, reconstructs the second half of the signed-int16 FP32 full-slab index in UB, and selects compiler optimization level 3 on the qualified wide aligned FP32 row paths. The 74 randomized full-output controls were run as fresh processes on the public evaluator runtime. The single-row BF16 tile stayed at 8,192 lanes after the two-wave variants showed mixed timings. Two OpenCode attempts to generate an alternative host selector timed out. The integrated ninth candidate uses the measured changes from the diagnostic kernels.

All 20 goldens passed on the first full run of this source. Thirteen medians met their saved archive thresholds. Cases 2, 9, 12, 16, 18, 19 and 20 exceeded them. The table uses the three warmed samples from each case in that run.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.540 | 1.500 | 0.5906× |
| 11 | 2.832 | 2.021 | 0.7136× |
| 2 | 23.796 | 23.650 | 0.9939× |
| 18 | 23.755 | 23.630 | 0.9947× |
| 7 | 6.525 | 6.500 | 0.9962× |
| 5 | 522.605 | 522.100 | 0.9990× |
| 4 | 241.402 | 241.490 | 1.0004× |
| 19 | 13.851 | 14.280 | 1.0310× |
| 6 | 5.964 | 6.200 | 1.0396× |
| 16 | 8.408 | 8.790 | 1.0454× |
| 9 | 6.482 | 6.782 | 1.0463× |
| 3 | 45.489 | 48.590 | 1.0682× |
| 12 | 6.735 | 7.273 | 1.0799× |
| 15 | 7.118 | 7.780 | 1.0930× |
| 8 | 7.791 | 8.540 | 1.0961× |
| 10 | 5.477 | 6.180 | 1.1284× |
| 17 | 7.625 | 8.760 | 1.1489× |
| 14 | 6.173 | 7.540 | 1.2214× |
| 20 | 15.821 | 19.560 | 1.2363× |
| 1 | 4.650 | 6.380 | 1.3720× |
| **GM** | | | **1.0300×** |

[Full hardware profiles](round9.json) · [Source patch](round9.patch) · [Control profiles](round9-controls.json)

### Two repeats of round nine

All 60 full-output checks passed. The two repeats met 18 and 15 saved thresholds. Their CANNBench / auto geometric means were 1.0451× and 1.0358×.

Case 20 exceeded its saved threshold in all three runs. Cases 9, 12 and 19 exceeded theirs in two of three runs. The full-suite measurements did not repeat the consistent paired-control improvements for the dense BF16 pair order and FP32 half-seed index.

Latest complete run in ascending order of CANNBench / auto.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.456 | 1.500 | 0.6107× |
| 11 | 2.865 | 2.021 | 0.7054× |
| 5 | 519.828 | 522.100 | 1.0044× |
| 2 | 23.501 | 23.650 | 1.0063× |
| 9 | 6.681 | 6.782 | 1.0151× |
| 4 | 234.880 | 241.490 | 1.0281× |
| 12 | 7.026 | 7.273 | 1.0352× |
| 19 | 13.766 | 14.280 | 1.0373× |
| 18 | 22.620 | 23.630 | 1.0447× |
| 3 | 45.722 | 48.590 | 1.0627× |
| 17 | 8.167 | 8.760 | 1.0726× |
| 10 | 5.735 | 6.180 | 1.0776× |
| 8 | 7.899 | 8.540 | 1.0811× |
| 16 | 8.111 | 8.790 | 1.0837× |
| 7 | 5.895 | 6.500 | 1.1026× |
| 6 | 5.609 | 6.200 | 1.1054× |
| 15 | 6.966 | 7.780 | 1.1169× |
| 14 | 6.383 | 7.540 | 1.1813× |
| 20 | 15.885 | 19.560 | 1.2314× |
| 1 | 4.587 | 6.380 | 1.3909× |
| **GM** | | | **1.0358×** |

[Both repeat profiles](round9-repeats.json)

## Tenth fixed candidate

The 68 grid controls, 22 BF16 wave controls and 24 FP32 slab controls all passed full-output checks. OpenCode generated a host grid selector. Its first answer skipped eligible even divisors. The corrected function matched an exhaustive oracle for 65,584 group-count and capacity pairs on both hosts. Host replay of the dispatch for all 20 shapes predicts five changed launches, on cases 2, 9, 12, 19 and 20.

The fixed operator selects the near-capacity even divisor for the qualified wide FP32 rows, reconstructs half of the cached gather index for the two-wave multi-lane BF16 slab, and uses the two-group order with optimization level three for dense BF16 pairs. It traverses columns across short batches of wide BF16 rows, retaining the true extent of each final chunk. Full FP32 slabs use the full index table. The complete operator SHA-256 is `0989ce641aa7797eb61c80eac96ea0e178fa7841fc684a0706e8be05fadda938`.

[Source patch](round10.patch) · [Grid controls](round10-grid-controls.json) · [BF16 wave controls](round10-wave-controls.json) · [FP32 slab controls](round10-slab-controls.json)

### First full run of round ten

All 20 full outputs passed. 19 medians met their saved archive times.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.402 | 1.500 | 0.6245× |
| 11 | 2.881 | 2.021 | 0.7015× |
| 15 | 8.108 | 7.780 | 0.9595× |
| 5 | 522.221 | 522.100 | 0.9998× |
| 2 | 23.628 | 23.650 | 1.0009× |
| 8 | 8.435 | 8.540 | 1.0124× |
| 4 | 238.056 | 241.490 | 1.0144× |
| 18 | 22.927 | 23.630 | 1.0307× |
| 3 | 45.407 | 48.590 | 1.0701× |
| 10 | 5.491 | 6.180 | 1.1255× |
| 19 | 12.670 | 14.280 | 1.1271× |
| 7 | 5.746 | 6.500 | 1.1312× |
| 9 | 5.976 | 6.782 | 1.1349× |
| 16 | 7.639 | 8.790 | 1.1507× |
| 12 | 6.162 | 7.273 | 1.1803× |
| 17 | 7.307 | 8.760 | 1.1989× |
| 14 | 6.288 | 7.540 | 1.1991× |
| 6 | 5.115 | 6.200 | 1.2121× |
| 20 | 15.633 | 19.560 | 1.2512× |
| 1 | 4.593 | 6.380 | 1.3891× |
| **GM** | | | **1.0597×** |

[Full hardware profiles](round10.json)

### Two repeats of round ten

All 60 full golden checks passed. Across the three complete suites, 19, 14, 18 of 20 medians met their saved archive times. CANNBench / auto geometric means were 1.0597×, 1.0528×, 1.0364×.

Latest complete run in ascending order of CANNBench / auto.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.532 | 1.500 | 0.5924× |
| 11 | 2.812 | 2.021 | 0.7187× |
| 9 | 6.928 | 6.782 | 0.9789× |
| 12 | 7.367 | 7.273 | 0.9872× |
| 4 | 241.804 | 241.490 | 0.9987× |
| 5 | 520.343 | 522.100 | 1.0034× |
| 2 | 23.374 | 23.650 | 1.0118× |
| 18 | 22.467 | 23.630 | 1.0518× |
| 16 | 8.301 | 8.790 | 1.0589× |
| 3 | 45.469 | 48.590 | 1.0686× |
| 17 | 8.065 | 8.760 | 1.0862× |
| 8 | 7.816 | 8.540 | 1.0926× |
| 15 | 7.085 | 7.780 | 1.0981× |
| 19 | 12.851 | 14.280 | 1.1112× |
| 14 | 6.772 | 7.540 | 1.1134× |
| 7 | 5.798 | 6.500 | 1.1211× |
| 10 | 5.505 | 6.180 | 1.1226× |
| 6 | 5.480 | 6.200 | 1.1314× |
| 20 | 15.595 | 19.560 | 1.2542× |
| 1 | 4.463 | 6.380 | 1.4295× |
| **GM** | | | **1.0364×** |

[Both repeat profiles](round10-repeats.json)

### Whole-row controls for round eleven

At the median of the three round-ten case medians, cases 9 and 15 exceeded their saved archive times. The other 18 met their thresholds. On case 15, changing only the previous flat grid to 64 took 7.585 and 7.581 µs. Keeping that grid and processing full rows cyclically with unroll four took 6.839 and 7.159 µs. All 16 control outputs passed. The next integrated source uses cyclic complete rows where the geometry and UB fit, and starts two-wave multi-lane BF16 slabs with full gate indices again. Its source SHA-256 is `2808b0c721bdc447de1678e13ad269d7bc44f1a2e815fe809cd8d2d06e555e4b`. [Control samples, grid and compiled allocations](round11-row-controls.json) · [Source patch against round ten](round11.patch)

Additional randomized controls compared case 9 using a complete gate index, a half-index seed, independent gate/value tables and an explicit compiler optimization level. On case 12 the 62-block grid gave 5.930 and 5.922 µs with the same kernel, tile, unroll and UB allocation as the 64-block control at 6.890 and 6.135 µs. All 24 controls passed. Round eleven keeps the 64-block choice so that the full-row and multi-lane changes can be assessed independently. [Samples and observed kernel geometry](round11-index-controls.json)

### First full run of round eleven

All 20 full outputs passed. 18 medians met their saved archive times.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.549 | 1.500 | 0.5885× |
| 11 | 2.803 | 2.021 | 0.7210× |
| 4 | 241.634 | 241.490 | 0.9994× |
| 5 | 521.269 | 522.100 | 1.0016× |
| 2 | 23.508 | 23.650 | 1.0060× |
| 12 | 7.072 | 7.273 | 1.0284× |
| 18 | 22.660 | 23.630 | 1.0428× |
| 19 | 13.692 | 14.280 | 1.0429× |
| 6 | 5.921 | 6.200 | 1.0471× |
| 8 | 7.912 | 8.540 | 1.0794× |
| 3 | 45.011 | 48.590 | 1.0795× |
| 7 | 6.013 | 6.500 | 1.0810× |
| 16 | 8.067 | 8.790 | 1.0896× |
| 9 | 6.091 | 6.782 | 1.1134× |
| 10 | 5.544 | 6.180 | 1.1147× |
| 15 | 6.906 | 7.780 | 1.1266× |
| 17 | 7.684 | 8.760 | 1.1400× |
| 14 | 6.512 | 7.540 | 1.1579× |
| 20 | 15.684 | 19.560 | 1.2471× |
| 1 | 4.479 | 6.380 | 1.4244× |
| **GM** | | | **1.0414×** |

[Full hardware profiles](round11.json)

### Two repeats of round eleven

All 60 full golden checks passed. Across the three complete suites, 18, 18, 20 of 20 medians met their saved archive times. CANNBench / auto geometric means were 1.0414×, 1.0454×, 1.0482×.

Latest complete run in ascending order of CANNBench / auto.

| Case | Auto (µs) | CANNBench (µs) | CANNBench / auto |
|---:|---:|---:|---:|
| 13 | 2.488 | 1.500 | 0.6029× |
| 11 | 2.782 | 2.021 | 0.7265× |
| 5 | 519.707 | 522.100 | 1.0046× |
| 7 | 6.445 | 6.500 | 1.0085× |
| 2 | 23.394 | 23.650 | 1.0109× |
| 8 | 8.321 | 8.540 | 1.0263× |
| 4 | 233.776 | 241.490 | 1.0330× |
| 18 | 22.442 | 23.630 | 1.0529× |
| 6 | 5.885 | 6.200 | 1.0535× |
| 9 | 6.373 | 6.782 | 1.0642× |
| 3 | 45.226 | 48.590 | 1.0744× |
| 19 | 13.148 | 14.280 | 1.0861× |
| 17 | 7.987 | 8.760 | 1.0968× |
| 15 | 7.026 | 7.780 | 1.1073× |
| 10 | 5.551 | 6.180 | 1.1133× |
| 12 | 6.303 | 7.273 | 1.1539× |
| 16 | 7.594 | 8.790 | 1.1575× |
| 14 | 6.273 | 7.540 | 1.2020× |
| 20 | 15.661 | 19.560 | 1.2490× |
| 1 | 4.416 | 6.380 | 1.4447× |
| **GM** | | | **1.0482×** |

[Both repeat profiles](round11-repeats.json)
