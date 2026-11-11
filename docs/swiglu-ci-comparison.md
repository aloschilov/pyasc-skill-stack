# Why the first CI SwiGLU measured 0.739×

Both ratios are geometric means of baseline time divided by candidate time on the same 19 case IDs. The original local study used three-sample medians on Ascend950PR_9599. The CI submission ran on the CANNBench Ascend950PR_9589 with a separately generated kernel, different lowering and an updated baseline. Case 12 used the packed BF16 template in both runs and is shown separately.

| Case | Compute | Local tile / unroll | CI tile / unroll | Local µs | CI µs | Local study baseline µs | CI baseline µs | Local ratio | CI ratio |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | rows | 6×1024 / 1 | 3×1024 / 1 | 7.059 | 8.08 | 6.380 | 6.38 | 0.904× | 0.790× |
| 2 | flat | 2048 / 2 | 4096 / 1 | 24.423 | 35.42 | 23.650 | 23.65 | 0.968× | 0.668× |
| 3 | flat | 4096 / 2 | 4096 / 1 | 49.693 | 64.93 | 48.590 | 48.59 | 0.978× | 0.748× |
| 4 | flat | 10560 / 2 | 4096 / 1 | 247.218 | 273.81 | 241.490 | 241.49 | 0.977× | 0.882× |
| 5 | flat | 10560 / 2 | 4096 / 1 | 516.925 | 543.48 | 522.100 | 522.10 | 1.010× | 0.961× |
| 6 | flat | 10560 / 2 | 4096 / 1 | 5.231 | 8.17 | 6.200 | 6.20 | 1.185× | 0.759× |
| 7 | rows | 6×1008 / 1 | 3×1024 / 1 | 7.398 | 11.07 | 6.500 | 6.50 | 0.879× | 0.587× |
| 8 | flat | 10560 / 2 | 4096 / 1 | 9.105 | 10.41 | 8.540 | 8.54 | 0.938× | 0.820× |
| 9 | slab | 144 / 1 | 128 / 1 | 15.229 | 14.77 | 6.782 | 22.05 | 0.445× | 1.493× |
| 10 | rows | 12×512 / 1 | 6×512 / 1 | 6.885 | 8.47 | 6.180 | 6.18 | 0.898× | 0.730× |
| 11 | rows | 6×512 / 1 | 3×512 / 1 | 2.974 | 4.66 | 2.021 | 3.56 | 0.680× | 0.764× |
| 12 | pair_groups | 2048 / 1 | 2048 / 1 | 10.625 | 14.07 | 7.273 | 24.24 | 0.685× | 1.723× |
| 13 | rows | 5×536 / 1 | 2×768 / 1 | 2.628 | 5.23 | 1.500 | 3.25 | 0.571× | 0.621× |
| 14 | rows | 12×512 / 1 | 6×512 / 1 | 9.345 | 12.60 | 7.540 | 7.54 | 0.807× | 0.598× |
| 15 | flat | 2048 / 2 | 4096 / 1 | 7.949 | 12.93 | 7.780 | 7.78 | 0.979× | 0.602× |
| 16 | flat | 10560 / 2 | 4096 / 1 | 8.732 | 11.97 | 8.790 | 8.79 | 1.007× | 0.734× |
| 17 | flat | 10560 / 2 | 4096 / 1 | 8.223 | 11.16 | 8.760 | 8.76 | 1.065× | 0.785× |
| 18 | flat | 2048 / 2 | 4096 / 1 | 22.698 | 35.69 | 23.630 | 23.63 | 1.041× | 0.662× |
| 19 | flat | 10560 / 2 | 4096 / 1 | 13.202 | 19.93 | 14.280 | 14.28 | 1.082× | 0.717× |
| 20 | rows | 54×56 / 1 | 24×64 / 1 | 20.611 | 38.56 | 19.560 | 19.56 | 0.949× | 0.507× |

Across the 19 generated cases the candidate took 1.380× as long, while the baseline took 1.142× as long. Multiplying 0.893 by 1.142/1.380 recovers 0.739. Device/kernel changes and baseline changes both matter.

- **Flat (11 cases).** CI always used 4,096 lanes and unroll 1. The local selector used 2,048, 4,096 or 10,560 lanes and unroll 2. The baseline was unchanged on all 11; CI took 1.355× as long geometrically. Case 3 kept its tile width but changed unroll. Case 5 grew from 516.925 to 543.48 µs, a much smaller relative change than case 15 (7.949 to 12.93 µs).
- **Strided rows (7 cases).** CI took 1.493× as long. In case 20 its width bucket padded 50 values to 64, and its fixed 6 KiB budget cut the height to 24. The local kernel used a 54×56 tile. Case 13 padded 536 to 768 in CI and cut the height from five rows to two. Its reported ratio nevertheless improved because the service baseline moved from 1.50 to 3.25 µs. Case 11 also changed baseline.
- **Short gather (case 9).** The candidate time decreased slightly. Most of the apparent speedup against baseline came from the baseline changing from 6.782 to 22.05 µs.

Tiles are reconstructed from the submitted package and the local dispatch trace. The service receipt records compute names and elapsed time, without a measured launch grid. These comparisons identify concrete dispatch differences. Isolating a causal contribution from tile size, unroll or core count would require a controlled comparison on the same runner.

[Local source, grid and three-sample measurements](https://github.com/aloschilov/pyasc-skill-stack/tree/909f47ca0c58963f04031e7e7bb2a0f583c82137/docs/evidence/swiglu-vector-grid-20261110) · [CI-generated source and official per-case receipt](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-swiglu-performance-43061369765)
