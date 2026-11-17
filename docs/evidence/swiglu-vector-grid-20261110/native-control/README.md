# Season 2 native SwiGLU control

CANNBench's public manifest at `72fb4064d80bcbd6195a100cd4bbabac7a4f8817` identifies task version 2.0.0. L1 contains Exp and SwiGLU with 20 cases each. The scoring baseline here is `tasks/metadata/950pr.json`, field `baseline_perf_us`.

On David, direct `aclnnSwiGlu` in CANN 9.2 produced the following FP32 vector-core task durations on 2026-11-13. Each median uses the last 12 of 17 synchronized calls with resident buffers under msprof. The read-back is checked against every element of the matching public Torch golden. Case 13 also has a separate finite control on the same shape.

| Case | Input | Fixture | Native blocks | Native µs | CANNBench µs | CANNBench/native |
|---:|---|---|---:|---:|---:|---:|
| 11 | 3×7×13×1018, dim −1 | Public | 24 | 3.5055 | 2.021 | 0.5765× |
| 13 | 11×13×16×67, dim 2 | Public | 18 | 3.1935 | 1.500 | 0.4697× |
| 13 | 11×13×16×67, dim 2 | Finite control | 18 | 3.2305 | — | — |

These measured native durations exceed the published targets on this device/software combination. The finite control validates the nondegenerate arithmetic. Score generated kernels using the published baseline as the numerator. The captured bundle of CSV and JSON records had SHA-256 `40b7b66ea008ed54e0baf101e66b6c33689f6919c519db5d6807db4d715b1da7`.

To replay in the target CANN environment, build the driver and run from a new directory with the public evaluator on the Python path. Set `TASK` to the Season 2 SwiGLU task directory. Each fixture needs its own fresh output directory.

```sh
g++ -O2 "$SOURCE/swiglu.cpp" -I"$ASCEND_HOME_PATH/include" \
    -L"$ASCEND_HOME_PATH/lib64" -lopapi -lnnopbase -lascendcl -o bench
python "$SOURCE/replay.py" "$TASK" 11
python "$SOURCE/replay.py" "$TASK" 13
```

## Frequency boost and L2 clean

A separate run on David used `torch_npu.npu_swiglu` from torch_npu 2.10.0 with the public input buffers from the ACLNN replay. Each row used a fresh process. The first call checked the public golden. Three warm-up calls preceded the five timed calls. The conditioned run used the installed CANNBench provider to warm up once and clean L2 before each timed call, as in the Season 2 profiling schedule. The trace confirmed a cache-clean kernel immediately before each of the five measured SwiGLU tasks. The method isolates the combined effect of the two settings with msprof and resident inputs.

| Case | Blocks | Mode | Native median, µs | CANNBench, µs | CANNBench/native |
|---:|---:|---|---:|---:|---:|
| 11 | 24 | Plain | 3.188 | 2.021 | 0.6339× |
| 11 | 24 | Boost and L2 clean | 3.965 | 2.021 | 0.5097× |
| 13 | 18 | Plain | 2.971 | 1.500 | 0.5049× |
| 13 | 18 | Boost and L2 clean | 3.607 | 1.500 | 0.4159× |

Both modes passed the public golden. The published 950PR metadata caps submitted durations at `3 * t_hw_us`, rounded to three decimal places. Cases 11 and 13 are at that cap and the metadata marks NPU revalidation as pending. The original uncapped timings are unavailable in this release. Projected CSVs, correctness and process statuses had bundle SHA-256 `7bcadbf784fa964eeacc4cabbd0fb1bcdfaa7971cc62b6161f1a040a30a7069b`. The results above refer to native SwiGLU. The OpenCode measurements in the MR use the three-call PyAsc profiler.

## Fixed grid and resident allocation controls

The two experiments used the same W52 module on David. In the tile experiment, `base` is the unedited module. `s1` distributes whole rows evenly across the grid. `s2` and `s4` distribute two and four column stripes over that grid. In case 11 their maximum local shapes were 17×256 and 35×128. In case 13 they were 8×272 and 16×136. Replaying the actual emitted copies on the host verified both input halves and one output write per element for all eight case/variant combinations.

The allocation experiment used W52 without source changes. `staged` called the public wrapper nine times with a CPU tensor. `resident` replayed the same low-level launch nine times within the lifetime of one pair of device allocations, synchronizing between launches. Each round reports the median of the last five device tasks for the allocation experiment and of three tasks for the tile experiment. The resident measurement describes warmed, reused addresses. Each public output matched the golden, with an additional finite input check for case 13.

| Case | Variant | Blocks | Round 1, µs | Round 2, µs | CANNBench, µs | CANNBench/variant (R1 / R2) |
|---:|---|---:|---:|---:|---:|---:|
| 11 | base | 35 | 2.660 | 2.744 | 2.021 | 0.7598× / 0.7365× |
| 11 | s1 | 35 | 2.622 | 2.651 | 2.021 | 0.7708× / 0.7624× |
| 11 | s2 | 35 | 2.841 | 2.959 | 2.021 | 0.7114× / 0.6830× |
| 11 | s4 | 35 | 3.109 | 3.113 | 2.021 | 0.6500× / 0.6492× |
| 11 | staged | 35 | 2.608 | 2.610 | 2.021 | 0.7749× / 0.7743× |
| 11 | resident | 35 | 2.469 | 2.553 | 2.021 | 0.8186× / 0.7916× |
| 13 | base | 36 | 2.181 | 2.213 | 1.500 | 0.6878× / 0.6778× |
| 13 | s1 | 36 | 2.261 | 2.237 | 1.500 | 0.6634× / 0.6705× |
| 13 | s2 | 36 | 2.303 | 2.589 | 1.500 | 0.6513× / 0.5794× |
| 13 | s4 | 36 | 2.420 | 2.412 | 1.500 | 0.6198× / 0.6219× |
| 13 | staged | 36 | 2.173 | 2.143 | 1.500 | 0.6903× / 0.7000× |
| 13 | resident | 36 | 2.059 | 2.058 | 1.500 | 0.7285× / 0.7289× |

Both striped variants took longer in both rounds than the full-width controls. Reusing device allocations reduced the measured vector task duration in both rounds, while both cases remained below the published target.

To reproduce, place the W52 source and runtime helper in a fresh `base/` package, with the Season 2 SwiGLU contract at `generation/task`. Run `tile_aspect.py` to generate the three controls and use the repository's `profile_unary.py PACKAGE swi_glu CASE` on each. For the allocation experiment run `resident_replay.py base CASE MODE` in a new process per measurement. Returned bundle SHA-256 was `a83fb7ebd36f021c547c423a363e686e6939e23fd1b5ddf421ae2df44c457a8c`.
