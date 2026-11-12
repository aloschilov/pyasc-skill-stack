# SwiGLU launch grids on Ascend950PR_9599

Independent OpenCode regeneration on David produced one 2D kernel, matched 12/20 previous grids and passed all 20 golden checks. Its CANNBench geometric mean was 0.6792×, and 1/20 medians met the saved archive thresholds. [Generated source, measured grids and per-case table](clean-opencode/README.md)

The final fixed SwiGLU candidate met all 20 saved archive times in its third full run on David. Its geometric mean against CANNBench was 1.0482×. [Qualified source, per-case table and repeat results](parity/README.md#qualified-fixed-candidate)

All 20 published Season 2 SwiGLU cases passed the full golden check for cap64, auto, UB-height auto, packed pairs, the gate-only variant, the native register variant, the archived job replay and the public shape-dispatch and reindex candidates. Cap64 and auto share tile shapes and math. The UB variant starts from auto and selects the height of padded short rows from the active UB capacity. Cap64 limits each launch to 64 independent work items. Auto uses the vector resources reported for the active context, bounded by the work available to that launch. A separately owned row tail launches one block.

## Measured grids and device time

The auto, UB, packed, gate-only, register and job grids report the `block_num` argument recorded on return from each successful PyASC `runtime.launch_kernel` call. All four invocations per case had the same grid. The cap64 grid comes from the retained dispatch code, input geometry and profiler task names. A grid of `72+1` means two successive launches.

Speed relative to CANNBench is `baseline_perf_us / candidate_us`. The auto vs cap64 column uses `cap64_us / auto_us`. The UB vs auto column uses `auto_us / ub_us`. Tile rows gives the physical height of the main UB-variant launch. Packed vs UB-auto is UB time divided by packed time; packed vs auto compares against the original grid-only auto. All packed times come from the full series, not the pilot. The last row uses the geometric mean of the 20 unrounded ratios. CANNBench publishes a scoring-capped baseline for the 950pr family with status `pending_npu_revalidation`. [Pinned CANNBench metadata](https://gitcode.com/cann/cann-bench/blob/72fb4064d80bcbd6195a100cd4bbabac7a4f8817/tasks/metadata/950pr.json)

| Case | Input shape | Cap64 grid | Auto grid | CANNBench µs | Cap64 µs | Auto µs | Cap64 vs CANNBench (×) | Auto vs CANNBench (×) | Auto vs cap64 (×) | UB grid | Tile rows | UB µs | UB vs CANNBench (×) | UB vs auto (×) | Packed grid | Packed µs | Packed vs CANNBench (×) | Packed vs auto (×) | Packed vs UB-auto (×) | Gate grid | Gate µs | Gate vs CANNBench (×) | Gate vs packed (×) | Gate vs auto (×) | Register grid | Register µs | Register vs CANNBench (×) | Register vs gate (×) | Register vs auto (×) | Job grid | Job µs | Job vs CANNBench (×) | Job vs register (×) | Fresh route | Fresh grid | Fresh µs | Fresh vs CANNBench (×) | Fresh vs register auto (×) | Fresh vs job (×) | Fresh vs grid auto (×) | DMA route | DMA grid | Tile | Unroll | DMA µs | DMA vs CANNBench (×) | DMA vs Fresh (×) | DMA vs job (×) | DMA vs grid auto (×) | Pair reindex route | Grid | Tile | Unroll | Pair reindex µs | vs CANNBench (×) | vs DMA auto (×) | vs job (×) | 2D auto route | Grid | Tile | Unroll | 2D auto µs | vs CANNBench (×) | vs pair auto (×) | vs job (×) | Direct FP16 route | Grid | Tile | Unroll | Direct µs | vs 2D auto (×) | vs job (×) | vs CANNBench (×) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 × 2048 | 64 | 64 | 6.380 | 9.965 | 10.763 | 0.64024 | 0.59277 | 0.9259 | 64 | 16 | 10.411 | 0.61281 | 1.0338 | 64 | 9.765 | 0.65335 | 1.1022 | 1.0662 | 64 | 10.284 | 0.62038 | 0.94953 | 1.04658 | 64 | 9.975 | 0.63960 | 1.03098 | 1.07900 | 72 | 5.016 | 1.27193 | 1.98864 | 2D copy | 72 | 7.059 | 0.90381 | 1.41309 | 0.71058 | 1.52472 | 2D copy | 72 | 6 | 1 | 6.802 | 0.93796 | 1.03778 | 0.73743 | 1.58233 | 2D copy | 72 | 6 | 1 | 6.286 | 1.01495 | 1.08209 | 0.79796 | 2D copy | 72 | 4 | 2 | 4.661 | 1.36880 | 1.34864 | 1.07616 | 2D copy | 72 | 4 | 2 | 4.719 | 0.98771 | 1.06294 | 1.35198 |
| 2 | 2048 × 4096 | 64 | 72 | 23.650 | 38.081 | 38.892 | 0.62104 | 0.60809 | 0.9791 | 72 | 16 | 38.143 | 0.62004 | 1.0196 | 72 | 38.994 | 0.60650 | 0.9974 | 0.9782 | 72 | 39.192 | 0.60344 | 0.99495 | 0.99235 | 72 | 39.477 | 0.59908 | 0.99278 | 0.98518 | 72 | 23.782 | 0.99445 | 1.65995 | flat chunks | 72 | 24.423 | 0.96835 | 1.61639 | 0.97375 | 1.59243 | flat chunks | 72 | 2048 | 2 | 24.145 | 0.97950 | 1.01151 | 0.98497 | 1.61077 | flat chunks | 72 | 2048 | 2 | 24.198 | 0.97735 | 0.99781 | 0.98281 | flat chunks | 72 | 2048 | 2 | 24.383 | 0.96994 | 0.99241 | 0.97535 | flat chunks | 72 | 2048 | 2 | 24.113 | 1.01120 | 0.98627 | 0.98080 |
| 3 | 4096 × 8192 | 64 | 72 | 48.590 | 150.123 | 147.905 | 0.32367 | 0.32852 | 1.0150 | 72 | 16 | 148.123 | 0.32804 | 0.9985 | 72 | 147.193 | 0.33011 | 1.0048 | 1.0063 | 72 | 149.534 | 0.32494 | 0.98434 | 0.98911 | 72 | 146.886 | 0.33080 | 1.01803 | 1.00694 | 72 | 45.645 | 1.06452 | 3.21801 | flat chunks | 72 | 49.693 | 0.97780 | 2.95587 | 0.91854 | 2.97637 | flat chunks | 72 | 4096 | 2 | 48.147 | 1.00920 | 1.03211 | 0.94803 | 3.07195 | flat chunks | 72 | 4096 | 2 | 49.068 | 0.99026 | 0.98123 | 0.93024 | flat chunks | 72 | 4096 | 2 | 49.551 | 0.98061 | 0.99025 | 0.92117 | flat chunks | 72 | 4096 | 2 | 49.284 | 1.00542 | 0.92616 | 0.98592 |
| 4 | 8192 × 16384 | 64 | 72 | 241.490 | 591.541 | 539.494 | 0.40824 | 0.44762 | 1.0965 | 72 | 1 | 538.953 | 0.44807 | 1.0010 | 72 | 539.944 | 0.44725 | 0.9992 | 0.9982 | 72 | 539.569 | 0.44756 | 1.00069 | 0.99986 | 72 | 539.676 | 0.44747 | 0.99980 | 0.99966 | 72 | 243.929 | 0.99000 | 2.21243 | flat chunks | 72 | 247.218 | 0.97683 | 2.18300 | 0.98670 | 2.18226 | flat chunks | 72 | 10560 | 2 | 248.658 | 0.97117 | 0.99421 | 0.98098 | 2.16962 | flat chunks | 72 | 10560 | 2 | 246.983 | 0.97776 | 1.00678 | 0.98763 | flat chunks | 72 | 10560 | 2 | 248.237 | 0.97282 | 0.99495 | 0.98265 | flat chunks | 72 | 10560 | 2 | 248.520 | 0.99886 | 0.98153 | 0.97171 |
| 5 | 2039 × 65520 | 64+1 | 72+1 | 522.100 | 1144.921 | 1150.281 | 0.45601 | 0.45389 | 0.9953 | 72+1 | 16 | 1161.036 | 0.44968 | 0.9907 | 72+1 | 1158.588 | 0.45063 | 0.9928 | 1.0021 | 72+1 | 1149.157 | 0.45433 | 1.00821 | 1.00098 | 72+1 | 1148.542 | 0.45458 | 1.00054 | 1.00151 | 72 | 523.609 | 0.99712 | 2.19351 | flat chunks | 72 | 516.925 | 1.01001 | 2.22187 | 1.01293 | 2.22524 | flat chunks | 72 | 10560 | 2 | 516.282 | 1.01127 | 1.00125 | 1.01419 | 2.22801 | flat chunks | 72 | 10560 | 2 | 517.985 | 1.00794 | 0.99671 | 1.01086 | flat chunks | 72 | 10560 | 2 | 518.014 | 1.00789 | 0.99994 | 1.01080 | flat chunks | 72 | 10560 | 2 | 517.303 | 1.00137 | 1.01219 | 1.00927 |
| 6 | 1022 × 2047 | 64 | 72 | 6.200 | 12.065 | 11.190 | 0.51388 | 0.55407 | 1.0782 | 72 | 1 | 10.552 | 0.58757 | 1.0605 | 72 | 11.540 | 0.53726 | 0.9697 | 0.9144 | 72 | 11.605 | 0.53425 | 0.99440 | 0.96424 | 72 | 11.010 | 0.56312 | 1.05404 | 1.01635 | 72 | 6.134 | 1.01076 | 1.79491 | flat chunks | 72 | 5.231 | 1.18524 | 2.10476 | 1.17262 | 2.13917 | flat chunks | 72 | 10560 | 2 | 5.406 | 1.14687 | 0.96763 | 1.13467 | 2.06992 | flat chunks | 72 | 10560 | 2 | 5.790 | 1.07081 | 0.93368 | 1.05941 | flat chunks | 72 | 10560 | 2 | 6.024 | 1.02922 | 0.96116 | 1.01826 | flat chunks | 72 | 10560 | 2 | 5.582 | 1.07918 | 1.09889 | 1.11071 |
| 7 | 1009 × 2016 | 63+1 | 63+1 | 6.500 | 23.766 | 23.435 | 0.27350 | 0.27736 | 1.0141 | 63+1 | 16 | 23.607 | 0.27534 | 0.9927 | 63+1 | 23.011 | 0.28247 | 1.0184 | 1.0259 | 63+1 | 23.008 | 0.28251 | 1.00013 | 1.01856 | 63+1 | 23.598 | 0.27545 | 0.97500 | 0.99309 | 72 | 6.687 | 0.97204 | 3.52894 | 2D copy | 72 | 7.398 | 0.87862 | 3.18978 | 0.90389 | 3.16775 | 2D copy | 72 | 6 | 1 | 6.978 | 0.93150 | 1.06019 | 0.95830 | 3.35841 | 2D copy | 72 | 6 | 1 | 7.381 | 0.88064 | 0.94540 | 0.90597 | 2D copy | 72 | 8 | 1 | 6.531 | 0.99525 | 1.13015 | 1.02389 | 2D copy | 72 | 8 | 1 | 5.963 | 1.09525 | 1.12142 | 1.09006 |
| 8 | 1538 × 1537 | 64 | 72 | 8.540 | 13.416 | 11.756 | 0.63655 | 0.72644 | 1.1412 | 72 | 1 | 11.805 | 0.72342 | 0.9958 | 72 | 11.631 | 0.73424 | 1.0107 | 1.0150 | 72 | 12.131 | 0.70398 | 0.95878 | 0.96909 | 72 | 11.566 | 0.73837 | 1.04885 | 1.01643 | 72 | 8.488 | 1.00613 | 1.36263 | flat chunks | 72 | 9.105 | 0.93795 | 1.27029 | 0.93224 | 1.29116 | flat chunks | 72 | 10560 | 2 | 8.575 | 0.99592 | 1.06181 | 0.98985 | 1.37096 | flat chunks | 72 | 10560 | 2 | 8.191 | 1.04261 | 1.04688 | 1.03626 | flat chunks | 72 | 10560 | 2 | 8.519 | 1.00247 | 0.96150 | 0.99636 | flat chunks | 72 | 10560 | 2 | 8.801 | 0.96796 | 0.96444 | 0.97034 |
| 9 | 363 × 367 × 14 | 64+1 | 72+1 | 6.782 | 110.477 | 95.993 | 0.06139 | 0.07065 | 1.1509 | 72+1 | 128 | 31.520 | 0.21516 | 3.0455 | 72+1 | 29.573 | 0.22933 | 3.2460 | 1.0658 | 72+1 | 30.445 | 0.22276 | 0.97136 | 3.15300 | 72+1 | 29.961 | 0.22636 | 1.01615 | 3.20393 | 72 | 6.385 | 1.06218 | 4.69240 | reindex | 72 | 15.229 | 0.44533 | 1.96736 | 0.41927 | 6.30330 | reindex | 72 | 512 | 2 | 6.367 | 1.06518 | 2.39186 | 1.00283 | 15.07665 | reindex | 72 | 512 | 2 | 7.189 | 0.94339 | 0.88566 | 0.88816 | reindex | 72 | 512 | 2 | 6.875 | 0.98647 | 1.04567 | 0.92873 | reindex | 72 | 512 | 2 | 6.396 | 1.07489 | 0.99828 | 1.06035 |
| 10 | 2049 × 1024 | 64+1 | 72+1 | 6.180 | 14.946 | 15.160 | 0.41349 | 0.40765 | 0.9859 | 72+1 | 16 | 15.023 | 0.41137 | 1.0091 | 72+1 | 15.039 | 0.41093 | 1.0080 | 0.9989 | 72+1 | 14.878 | 0.41538 | 1.01082 | 1.01895 | 72+1 | 15.106 | 0.40911 | 0.98491 | 1.00357 | 72 | 5.756 | 1.07366 | 2.62439 | 2D copy | 72 | 6.885 | 0.89760 | 2.19405 | 0.83602 | 2.20189 | 2D copy | 72 | 12 | 1 | 6.964 | 0.88742 | 0.98866 | 0.82654 | 2.17691 | 2D copy | 72 | 12 | 1 | 5.996 | 1.03069 | 1.16144 | 0.95997 | 2D copy | 72 | 8 | 2 | 5.339 | 1.15752 | 1.12306 | 1.07810 | 2D copy | 72 | 8 | 2 | 4.873 | 1.09563 | 1.18120 | 1.26821 |
| 11 | 3 × 7 × 13 × 1018 | 17+1 | 17+1 | 2.021 | 12.184 | 11.186 | 0.16587 | 0.18067 | 1.0892 | 17+1 | 16 | 11.939 | 0.16928 | 0.9369 | 17+1 | 12.068 | 0.16747 | 0.9269 | 0.9893 | 17+1 | 10.885 | 0.18567 | 1.10868 | 1.02765 | 17+1 | 10.836 | 0.18651 | 1.00452 | 1.03230 | 35 | 3.010 | 0.67143 | 3.60000 | 2D copy | 46 | 2.974 | 0.67956 | 3.64358 | 1.01210 | 3.76126 | 2D copy | 46 | 6 | 1 | 2.975 | 0.67933 | 0.99966 | 1.01176 | 3.76000 | 2D copy | 46 | 6 | 1 | 3.044 | 0.66393 | 0.97733 | 0.98883 | 2D copy | 35 | 8 | 1 | 2.959 | 0.68300 | 1.02873 | 1.01724 | 2D copy | 35 | 8 | 1 | 2.903 | 1.01929 | 1.03686 | 0.69618 |
| 12 | 1000003 × 2 | 64+1 | 72+1 | 7.273 | 1272.485 | 1107.532 | 0.00572 | 0.00657 | 1.1489 | 72+1 | 128 | 445.974 | 0.01631 | 2.4834 | 72+1 | 31.424 | 0.23145 | 35.2448 | 14.1921 | 72 | 10.773 | 0.67511 | 2.91692 | 102.80627 | 72 | 7.492 | 0.97077 | 1.43793 | 147.82862 | 72 | 6.345 | 1.14626 | 1.18077 | packed BF16 | 72 | 10.625 | 0.68452 | 0.70513 | 0.59718 | 104.23831 | packed BF16 | 72 | 2048 | 1 | 10.997 | 0.66136 | 0.96617 | 0.57698 | 100.71219 | reindex | 72 | 4096 | 2 | 6.166 | 1.17953 | 1.78349 | 1.02903 | reindex | 72 | 4096 | 2 | 7.482 | 0.97207 | 0.82411 | 0.84804 | reindex | 72 | 4096 | 2 | 7.268 | 1.02944 | 0.87300 | 1.00069 |
| 13 | 11 × 13 × 16 × 67 | 8+1 | 8+1 | 1.500 | 13.627 | 13.610 | 0.11008 | 0.11021 | 1.0012 | 8+1 | 16 | 13.694 | 0.10954 | 0.9939 | 8+1 | 13.524 | 0.11091 | 1.0064 | 1.0126 | 8+1 | 13.633 | 0.11003 | 0.99200 | 0.99831 | 8+1 | 13.325 | 0.11257 | 1.02311 | 1.02139 | 36 | 2.550 | 0.58824 | 5.22549 | 2D copy | 29 | 2.628 | 0.57078 | 5.07040 | 0.97032 | 5.17884 | 2D copy | 29 | 5 | 1 | 2.598 | 0.57737 | 1.01155 | 0.98152 | 5.23865 | 2D copy | 29 | 5 | 1 | 2.673 | 0.56117 | 0.97194 | 0.95398 | 2D copy | 36 | 4 | 1 | 2.628 | 0.57078 | 1.01712 | 0.97032 | 2D copy | 36 | 4 | 1 | 2.623 | 1.00191 | 0.97217 | 0.57186 |
| 14 | 3 × 7 × 11 × 13 × 1012 | 64+1 | 72+1 | 7.540 | 23.042 | 22.275 | 0.32723 | 0.33850 | 1.0344 | 72+1 | 16 | 22.657 | 0.33279 | 0.9831 | 72+1 | 21.870 | 0.34476 | 1.0185 | 1.0360 | 72+1 | 22.160 | 0.34025 | 0.98691 | 1.00519 | 72+1 | 21.798 | 0.34590 | 1.01661 | 1.02188 | 72 | 7.385 | 1.02099 | 2.95166 | 2D copy | 72 | 9.345 | 0.80685 | 2.33258 | 0.79026 | 2.38363 | 2D copy | 72 | 12 | 1 | 9.594 | 0.78591 | 0.97405 | 0.76975 | 2.32176 | 2D copy | 72 | 12 | 1 | 9.817 | 0.76806 | 0.97728 | 0.75227 | 2D copy | 72 | 16 | 1 | 8.745 | 0.86221 | 1.12258 | 0.84448 | 2D copy | 72 | 16 | 2 | 7.449 | 1.17398 | 0.99141 | 1.01222 |
| 15 | 512 × 4096 | 32 | 32 | 7.780 | 18.810 | 18.529 | 0.41361 | 0.41988 | 1.0152 | 32 | 16 | 18.151 | 0.42863 | 1.0208 | 32 | 18.452 | 0.42163 | 1.0042 | 0.9837 | 32 | 18.612 | 0.41801 | 0.99140 | 0.99554 | 32 | 18.340 | 0.42421 | 1.01483 | 1.01031 | 72 | 7.520 | 1.03457 | 2.43883 | flat chunks | 72 | 7.949 | 0.97874 | 2.30721 | 0.94603 | 2.33099 | flat chunks | 72 | 2048 | 2 | 7.224 | 1.07697 | 1.10036 | 1.04097 | 2.56492 | flat chunks | 72 | 2048 | 2 | 7.837 | 0.99273 | 0.92178 | 0.95955 | flat chunks | 72 | 2048 | 2 | 7.259 | 1.07177 | 1.07963 | 1.03596 | flat chunks | 72 | 2048 | 2 | 7.670 | 0.94641 | 0.98044 | 1.01434 |
| 16 | 254 × 16383 | 64 | 72 | 8.790 | 21.173 | 19.477 | 0.41515 | 0.45130 | 1.0871 | 72 | 1 | 18.883 | 0.46550 | 1.0315 | 72 | 19.095 | 0.46033 | 1.0200 | 0.9889 | 72 | 18.552 | 0.47380 | 1.02927 | 1.04986 | 72 | 18.764 | 0.46845 | 0.98870 | 1.03800 | 72 | 8.310 | 1.05776 | 2.25800 | flat chunks | 72 | 8.732 | 1.00664 | 2.14888 | 0.95167 | 2.23053 | flat chunks | 72 | 10560 | 2 | 9.051 | 0.97116 | 0.96476 | 0.91813 | 2.15192 | flat chunks | 72 | 10560 | 2 | 8.254 | 1.06494 | 1.09656 | 1.00678 | flat chunks | 72 | 10560 | 2 | 8.677 | 1.01302 | 0.95125 | 0.95770 | flat chunks | 72 | 10560 | 2 | 8.920 | 0.97276 | 0.93161 | 0.98543 |
| 17 | 4096 × 1023 | 64 | 72 | 8.760 | 19.352 | 17.641 | 0.45267 | 0.49657 | 1.0970 | 72 | 1 | 17.237 | 0.50821 | 1.0234 | 72 | 17.588 | 0.49807 | 1.0030 | 0.9800 | 72 | 17.211 | 0.50898 | 1.02190 | 1.02498 | 72 | 17.730 | 0.49408 | 0.97073 | 0.99498 | 72 | 8.398 | 1.04311 | 2.11122 | flat chunks | 72 | 8.223 | 1.06530 | 2.15615 | 1.02128 | 2.14532 | flat chunks | 72 | 10560 | 2 | 7.971 | 1.09898 | 1.03161 | 1.05357 | 2.21315 | flat chunks | 72 | 10560 | 2 | 8.073 | 1.08510 | 0.98737 | 1.04026 | flat chunks | 72 | 10560 | 2 | 8.142 | 1.07590 | 0.99153 | 1.03144 | flat chunks | 72 | 10560 | 2 | 8.115 | 1.00333 | 1.03487 | 1.07948 |
| 18 | 2 × 1023 × 4096 | 64+1 | 72+1 | 23.630 | 56.562 | 57.033 | 0.41777 | 0.41432 | 0.9917 | 72+1 | 16 | 56.648 | 0.41714 | 1.0068 | 72+1 | 57.264 | 0.41265 | 0.9960 | 0.9892 | 72+1 | 56.456 | 0.41856 | 1.01431 | 1.01022 | 72+1 | 56.285 | 0.41983 | 1.00304 | 1.01329 | 72 | 23.214 | 1.01792 | 2.42461 | flat chunks | 72 | 22.698 | 1.04106 | 2.47973 | 1.02273 | 2.51269 | flat chunks | 72 | 2048 | 2 | 22.787 | 1.03699 | 0.99609 | 1.01874 | 2.50287 | flat chunks | 72 | 2048 | 2 | 22.800 | 1.03640 | 0.99943 | 1.01816 | flat chunks | 72 | 2048 | 2 | 23.127 | 1.02175 | 0.98586 | 1.00376 | flat chunks | 72 | 2048 | 2 | 23.377 | 0.98931 | 0.99303 | 1.01082 |
| 19 | 4 × 510 × 4097 | 64 | 72 | 14.280 | 46.400 | 41.973 | 0.30776 | 0.34022 | 1.1055 | 72 | 4 | 41.736 | 0.34215 | 1.0057 | 72 | 42.183 | 0.33852 | 0.9950 | 0.9894 | 72 | 41.643 | 0.34291 | 1.01297 | 1.00792 | 72 | 41.147 | 0.34705 | 1.01205 | 1.02007 | 72 | 13.411 | 1.06480 | 3.06815 | flat chunks | 72 | 13.202 | 1.08165 | 3.11672 | 1.01583 | 3.17929 | flat chunks | 72 | 10560 | 2 | 12.929 | 1.10449 | 1.02112 | 1.03728 | 3.24642 | flat chunks | 72 | 10560 | 2 | 13.833 | 1.03231 | 0.93465 | 0.96949 | flat chunks | 72 | 10560 | 2 | 13.705 | 1.04196 | 1.00934 | 0.97855 | flat chunks | 72 | 10560 | 2 | 13.115 | 1.04499 | 1.02257 | 1.08883 |
| 20 | 2 × 3 × 17 × 512 × 100 | 64 | 72 | 19.560 | 33.595 | 31.090 | 0.58223 | 0.62914 | 1.0806 | 72 | 16 | 30.661 | 0.63794 | 1.0140 | 72 | 30.020 | 0.65157 | 1.0356 | 1.0214 | 72 | 30.248 | 0.64665 | 0.99246 | 1.02784 | 72 | 31.076 | 0.62942 | 0.97336 | 1.00045 | 72 | 15.734 | 1.24317 | 1.97509 | 2D copy | 72 | 20.611 | 0.94901 | 1.50774 | 0.76338 | 1.50842 | 2D copy | 72 | 54 | 1 | 20.621 | 0.94855 | 0.99952 | 0.76301 | 1.50769 | 2D copy | 72 | 54 | 1 | 23.144 | 0.84514 | 0.89099 | 0.67983 | 2D copy | 72 | 64 | 2 | 16.612 | 1.17746 | 1.39321 | 0.94715 | 2D copy | 72 | 64 | 2 | 16.587 | 1.00151 | 0.94857 | 1.17924 |
| **Geometric mean** | | | | | | | **0.28294** | **0.29703** | **1.04980** | | | | **0.33050** | **1.11267** |  |  | **0.37834** | **1.27374** | **1.14476** |  |  | **0.39941** | **1.05571** | **1.34470** |  |  | **0.40924** | **1.02460** | **1.37778** |  |  | **1.00317** | **2.45130** |  |  |  | **0.88093** | **2.15260** | **0.87814** | **2.96580** |  |  |  |  |  | **0.92978** | **1.05546** | **0.92685** | **3.13028** |  |  |  |  |  | **0.94521** | **1.01659** | **0.94223** |  |  |  |  |  | **0.98344** | **1.04045** | **0.98034** |  |  |  |  |  | **1.02370** | **1.00357** | **1.00675** |

[Full comparison CSV](comparison.csv) includes the earlier reference, parent, archived and generated candidates, three device-time samples for cap64, auto, UB and packed, and the observed auto, UB and packed launch counts in call order. The gate-only columns include its three samples and complete launch history. Register and job grids are observed launch counts. Register vs gate and register vs auto use the prior time divided by the register time. Job vs register uses register time divided by job time. The JSON profiles below include all samples and grids.

## FP16 direct division and measured UB

OpenCode 1.2.19 with cannbot/GLM-5.3 regenerated the 2D row kernel and its host geometry from the updated skill and their previous implementations. The request supplied their interfaces and no shape, case ID, target unroll or target timing. The resulting 2D kernel widens its inputs, divides the gate by `1 + exp(-gate)`, and multiplies by the value. The generated geometry docstring abbreviates the denominator incorrectly as `sigmoid(gate)`. The executed arithmetic and full-output comparisons use the expression above.

The FP16 selector uses 20 bytes per padded lane at unroll 2 on this build. On case 14 the selector chose 16 rows, a physical width of 512 and unroll 2. The compiler reported exactly 163,840 UB bytes with 253,952 available. The previous geometry chose unroll 1. The 188 tiles share one 72-block grid including the final 11 rows. Host probes also exercised the exact FP16 UB boundary, BF16 with the same element size, and the short FP32 grid.

The independent case-14 pilot passed at 7.860 µs. Case 14 in the full series ran in 7.449 µs versus 8.745 µs for the previous adaptive 2D auto, 7.385 µs for the archive and 7.540 µs for the published CANNBench baseline. All 20 full outputs passed. Geometric means of reference time divided by the candidate median over the 20 cases were 1.02370× versus previous auto, 1.00357× versus the archive, and 1.00675× versus CANNBench. The first two baselines come from earlier runs. Each candidate time uses the median of three warmed device tasks. The pilot is excluded from the ratios. The seven cases using the regenerated 2D kernel measured 1.05170× versus the previous auto. The other thirteen retained their kernels and geometry and measured 1.00893× across the separate series.

[Generation and boundary checks](fp16-direct/regenerate.py) · [Executed operator](fp16-direct/swi_glu.py) · [Pilot and all 20 profiles](fp16-direct/measurements.json)

## Adaptive 2D row copies

OpenCode generated the host function for ordinary strided 2D copies from the updated skill. The qualification script installed the returned function in the previous pair-reindex operator and retained the separate short-gather fallback. An earlier whole-selector generation also changed that fallback. The table reports the scoped regeneration. The model request supplied no case ID, target height or target unroll. The device kernels and runtime adapter match the previous candidate. The selector chooses a power-of-two row height from the logical DMA payload and checks the aligned physical width against live UB. It selects software pipelining when the second iteration fits and there is enough work per vector core.

Case 20 has 52,224 FP32 rows with 50 useful elements per half. A 64-row descriptor moves 12,800 useful bytes and has 56 physical columns. Unroll 2 compiled to 100,352 bytes of the available 253,952. One 72-block launch processes 816 tiles. The previous auto used height 54 and unroll 1.

The independent case-20 pilot passed at 16.355 µs. In the subsequent 20-case series case 20 took 16.612 µs, versus 23.144 µs for the previous auto, 15.734 µs for the archived job and 19.560 µs in the published 950PR scoring metadata. All 20 full goldens passed. The geometric means of reference time divided by the warmed candidate median are 0.98344× versus CANNBench, 1.04045× versus previous auto and 0.98034× versus the archive. Baselines come from their earlier series. The seven 2D cases measured 1.15853× versus previous auto. The 13 gather and flat cases kept their kernel geometry and measured 0.98193× across the separate runs.

[Regeneration and selector probes](rows-2d/regenerate.py) · [Executed operator](rows-2d/swi_glu.py) · [Runtime adapter](dma/_pyasc_runtime.py) · [Pilot and all 20 profiles](rows-2d/measurements.json)

## Dense BF16 pairs on the public reindex build

OpenCode regenerated the host selector from the previous DMA auto, the operator contract and the updated pair and slab references. The generation prompt supplied no case, requested tile or requested unroll. Its selector reused the existing public gather kernel with cached even and odd int32 indices. The regeneration script runs the shape and reduced-capacity checks on the returned code before packaging it. Device kernels and the runtime adapter are identical to the preceding DMA candidate. The hardware profiler measures warmed vector tasks with prebuilt indices.

On the 1,000,003 × 2 BF16 input, a 16 KiB input transfer fits 4,096 pairs. At this height unroll 2 uses 114,688 of 253,952 UB bytes and 245 tiles provide over two tiles per active AIV. A single 72-block launch covers the final 579 pairs. The independent pilot measured 7.502 µs (samples 7.656, 7.502, 6.634). The full series recorded 6.885, 6.166 and 6.054 µs. The full-series median was 6.166 µs versus 10.997 µs for DMA auto, 6.345 µs for the archived job and 7.273 µs in the published scoring metadata.

All 20 full goldens passed. Speed is reference time divided by the candidate median. The geometric means across 20 cases were 0.94521× vs CANNBench, 1.01659× vs DMA auto and 0.94223× vs the archive. The other 19 cases retained their kernels, grids, tiles and unrolls and measured 0.98696× relative to the separate DMA series.

[Regeneration and selector checks](pair-reindex/regenerate.py) · [Executed operator](pair-reindex/swi_glu.py) · [Shared runtime adapter](dma/_pyasc_runtime.py) · [Pilot and all 20 profiles](pair-reindex/measurements.json)

## Reindex tile height from DMA and UB

OpenCode regenerated the selector and host entry point on David. The device kernels are byte-identical to the previous public shape-dispatch candidate (“Fresh”). The selector received its previous source, the [reindex skill](../../../skills/asctile-api-patterns/references/reindex-slabs.md) and a request to preserve the existing routes and read live capacities. The request contained no case ID, target height or target unroll. The skill gives the measured BF16 per-lane UB costs and a 16 KiB DMA target for this build. The runtime adapter binds the generated capacity queries to this PyASC build. An earlier draft misread the per-lane UB cost. A subsequent draft extended short-row routing to wide BF16 inputs and overflowed UB on case 6. The final skill gives the units and confines the calculation to the existing short-row gather branch. A further small-work check found that an intermediate selector accepted an unaligned one-row gather. The final rule includes the byte-alignment predicate and a fallback example. A fresh whole-selector attempt changed the established 2D/flat threshold; the measured selector is the preceding route-preserving OpenCode output with a scoped OpenCode alignment repair. The repair request supplied no tile height or unroll. The final selector preserved all 20 previous routes and the tile geometry for the other 19 shapes in a host-side comparison.

For case 9, the input row is 28 bytes. The three row bounds are 585 from a 16 KiB DMA, 1,511 from 253,952 bytes of UB at 168 bytes per row, and 1,851 from 133,221 rows over 72 AIVs. Rounding the smallest bound down gives 512 rows and 14,336 input bytes per full transfer. There are 261 tiles. Unroll 2 fits at 100,352 compiled UB bytes and 72 blocks cover all tiles, including the tail. The last tile handles 101 valid rows. In host-only probes at the same shape, 90,000 UB bytes yield 512 rows at unroll 1, and 40,000 bytes yield 128 rows at unroll 2. These are simulated capacities. The pilot passed at 6.449 µs. The independent matrix run measured 6.367 µs versus 15.229 µs for Fresh (144 rows, unroll 1), 6.385 µs for the archived job and 6.782 µs for the published baseline.

All 20 cases passed their full golden check. The new columns give observed kernel route, launch grid, tile, unroll and device-time median. Tile is measured in rows for gather and 2D copies, pairs for packed BF16 and elements for flat chunks. Each speed is reference time divided by the new time. The 20-case geometric means are 0.92978× against CANNBench, 1.05546× against Fresh, 0.92685× against the archived job and 3.13028× against grid-only auto. For the other 19 cases, which kept the same layout, tile, unroll and grid, the geometric mean versus Fresh is 1.01098× across separate series. The reference columns retain their earlier series. [Executed source](dma/swi_glu.py) · [Pilot and 20 hardware profiles](dma/measurements.json)

## OpenCode shape dispatch on the public reindex build

The two pilot goldens passed before the full matrix. Case 9 took 15.092 µs and case 12 took 11.123 µs in the pilots. All 20 outputs in the subsequent series passed the same full-tensor check. Each Fresh route and Fresh grid in the table comes from the task name and launch recorded on David. Cases 1, 7, 10, 11, 13, 14 and 20 used 2D copies. Cases 2–6, 8 and 15–19 used flat chunks. Case 9 used reindex and case 12 used the packed BF16 path.

OpenCode 1.2.19 with GLM-5.3 received the SwiGLU contract and skills at `9d890d01`. Its first whole-module attempt returned a plan. Both whole-module code requests exhausted their 32,000-token output budgets before emitting source. A separate selector request supplied the SwiGLU skill excerpt and the observed 253952-byte UB and 72 AIVs. The request left the route names and thresholds to the model. The generation directory excluded the job operator and its dispatcher. The [first executable selector](shape/first-selector.txt) picked packed BF16 neighbour pairs first, slab gathers for half-rows up to 32 bytes, 2D copies through 3072 bytes, and flat chunks above that. A second OpenCode response preserved the route conditions and converted its tile to one integer using the supplied packed and flat limits of 2048 pairs and 10560 elements. The selector embeds this device’s capacity and count in its tile heuristics. Actual launch counts still come from the active vector resources and available work.

OpenCode generated the 2D, gather and flat kernels in separate requests after choosing the layouts. The requests included real API syntax and additional addressing feedback for gather and flat. The measured module also includes hand-written launch glue and fixes to PyASC type annotations, GM tensor bindings and offset lists. The flat draft contained identity gathers using an unavailable `arange`. Removing them let the kernel consume the copied tiles directly. The packed implementation comes from the earlier public two-transpose template. [Executed source](shape/swi_glu.py) · [Prompts, three device samples per case and pilots](shape/measurements.json)

On case 9, the selector batches 144 rows of two seven-element halves, caches the two index tensors and performs two local gathers from one contiguous slab. The generated gather used 926 tiles, unroll 1 and 24,192 bytes of UB. The archive used 261 tiles of 512 rows, unroll 2 and 100,352 bytes of UB. Both launched 72 blocks. The full-series median was 15.229 µs, against 29.961 µs for register auto, 6.385 µs for the replayed job and 6.782 µs in CANNBench. On case 12, the BF16 last-axis guard wins before the short-half rule. Its 72-block public kernel works on 2048 pairs per tile with the tail in the main launch. It took 10.625 µs against 7.492 µs for register auto, 6.345 µs for the job and 7.273 µs in CANNBench.

Across all 20 cases the geometric mean of reference time divided by Fresh time is **0.88093×** for CANNBench, **2.15260×** for register auto, **0.87814×** for the archived job and **2.96580×** for the original grid auto. Fresh and job use the same archive-bundled PyASC extension on CPython 3.12.13. Register auto used PyASC `6b3447b4` on CPython 3.10.14. All hardware comparisons use separate warmed runs on Ascend950PR_9599 with CANN 9.2.0 and the same profiler protocol. CANNBench is the published 950PR scoring baseline.

## Register SwiGLU and archived submission replay

OpenCode 1.2.19 with GLM-5.3 generated the [layout guard and launch](native/dispatch.txt)
from skill revision `3775d51` for a native-enabled 950PR build. The executed
[candidate](native/swi_glu.py) combines that output, the
[register kernel and selector](../../../skills/asctile-api-patterns/templates/register-bf16-swiglu.py)
and the prior gate-only operator. Only case 12 selected the register path.
Its tile contains 2048 pairs. A single 72-block grid handles all 489 tiles,
including the final 579 pairs. The compiled specialization uses 12288 bytes
of UB. It loads adjacent BF16 pairs, applies `DeInterleave` in registers,
computes FP32 SwiGLU there and rounds the result to BF16 for the store.
Public-only candidates can use the separate public tile template.

The independent pilot passed in 7.127 µs. In the all-case series the
median was 7.492 µs against 10.773 µs for the previous gate-only
auto and 7.273 µs in the published scoring metadata. The other 19
cases retained the prior kernels and grids. Their geometric mean ratio to
the preceding gate-only series was 1.00648×. The full-series ratios
compare separate runs. [Pilot and all 20 native profiles](native/measurements.json)

The job columns replay `job_f9f0facb7d6b-submission_asctile_l1_public_v2 (1).zip`
(SHA-256 `59beca25e04ab8bb546c7bf8e6bf0c4a90612f6c13928a8a896cc5dd120edd3d`). Its unmodified `cann_bench.swi_glu`
and bundled `asc` ran under CPython 3.12.13. The generated candidate used
CPython 3.10.14 and PyASC `6b3447b4`. Both ran on the same 950PR device
with CANN 9.2.0, PyTorch 2.5.1+cpu, the same case seeds, full golden checks
and the same three-call warmed vector-task metric. The replay omits an explicit
L2 flush. Timings from the archive's cold-tuning notes and an online service
score are separate measurements. Version 1.0.0 in the archive is the submission
package version. The recorded extension hash identifies the bundled PyASC.
[Archive replay and hashes](native/archive-measurements.json)

The job dispatches from the physical half-row size. For short halves it reads
one contiguous slab and uses two indexed gathers in UB. Its case-12
specialization processes 4096 pairs per tile with loop unroll 2. It measured
6.345 µs. The compute kernels, runtime versions, tile sizes,
loop unroll and division sequence differ between the archive and the native
candidate, so this comparison measures complete implementations. On case 9
(two halves of seven) the gather route took 6.385 µs versus 29.961 µs for
the generated row kernel. For intermediate
rows the archive aggregates rows into 2D copies. For wide halves it uses
contiguous flat chunks with unrolling. All 20 archived outputs passed the
golden check. The archive’s own self-check with its default seed of 2026 also
passed MERE and MARE for all 20 cases. The job's measured speed relative to CANNBench has geometric
mean 1.00317×.

## Gate-only packed path

OpenCode 1.2.19 with GLM-5.3 generated the host dispatch from the updated skill.
The [assembly script](gate/assemble.py) combines its [generated dispatch](gate/dispatch.txt),
the [skill template](../../../skills/asctile-api-patterns/templates/packed-bf16-swiglu.py)
and the UB-height parent into the [executed source](gate/swi_glu.py).
The initial dispatcher referenced an undefined runner cap and the obsolete tail-launch signature.
After repair, a fresh request given the final skill’s host example reproduced the measured file byte for byte.
The hardware profile records a 253952-byte active UB capacity. Case 12 selected
2048 pairs (184832 compiled UB bytes). The 488 full tiles and final 579 pairs
ran on a single 72-block grid. Each tile loads adjacent BF16 pairs, separates
the two planes and applies FP32 exp to its gate plane. The 19 other cases
used the previous kernels.

The independent case-12 pilot took 10.619 µs. In the fresh 20-case
series case 12 took 10.773 µs. Gate vs packed uses the previous
256-pair packed time divided by the new time. Gate vs auto uses the original
grid-only time divided by the new time. All 20 cases passed the full golden check.
The table uses full-series medians. The ratios compare separate hardware series.
[Pilot and all 20 profiles](gate/measurements.json)

## Packed BF16 pair result

The packed branch ran only for case 12: contiguous BF16, normalized split axis
last, input extent two. Other cases used the same kernels and launch grids as
UB-auto, including case 9 (two halves of seven, not seven interleaved pairs).

The candidate starts from [UB-auto](ub/swi_glu.py) and uses the
[256-pair device template at its measured revision](https://github.com/aloschilov/pyasc-skill-stack/blob/ab19c33d633ceb06afd150df3a659d73e4782ea4/skills/asctile-api-patterns/templates/packed-bf16-swiglu.py).
OpenCode 1.2.19 / GLM-5.3 generated the host dispatch and repaired its initial
axis check. The first guard compared the axis **index** to 2 and therefore
fell through to UB-auto on the rank-two pilot. A full-source generation attempt
did not produce the measured kernel. The packaged candidate mechanically inserts
the template and the repaired OpenCode dispatch into the UB candidate.
[Executed source](packed/swi_glu.py) · [Pilot and all 20 profiles](packed/measurements.json).

One full tile reads 512 contiguous BF16 values, deinterleaves 256 pairs with two
local aligned transposes and writes 256 BF16 outputs. The final tile reads
134 values and stores 67. The full-group and tail specializations each compiled to
23,104 UB bytes on David; padded UB-auto required
163,840 bytes for 128 rows. The hardware pilot passed the full golden at
31.116 µs. The separate full-series result was
31.424 µs, versus 445.974 µs for UB-auto and
7.273 µs for the scoring-capped CANNBench baseline.
The resulting speeds are 14.1921× vs UB-auto and
0.23145× vs CANNBench. All four invocations used
`72+1`. Across 20 cases the geometric means are
1.14476× vs UB-auto, 1.27374× vs
grid-only auto and 0.37834× vs CANNBench. Across the other
19 the measured geometric mean vs UB-auto was 1.00270× in separate
series using unchanged paths.

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
| Packed pairs (OpenCode dispatch and skill template) | `d664fe0484c30f9f49a38173b3dd473fd7206465b87d489aaa68cd96cab1c709` |
| Native-enabled OpenCode register candidate | `ff47b526faf14e76159720d3753731c3ffa795339659612994588c066ade71ed` |
| Archived job f9f0facb7d6b operator | `b1084b683f3e632525c85b9bb049aa36e62f734677016b6d43151b78e38ee4d6` |
| UB-sized gate-only pair tile with an in-grid tail | `eab4342b39ffa94a6fe5f81a325088c31c967ca7ee990d09c15cc67262b4b5ce` |
