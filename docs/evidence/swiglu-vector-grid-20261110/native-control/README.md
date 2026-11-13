# Season 2 native SwiGLU control

CANNBench's public manifest at `72fb4064d80bcbd6195a100cd4bbabac7a4f8817` identifies task version 2.0.0. L1 contains Exp and SwiGLU with 20 cases each. The scoring baseline here is `tasks/metadata/950pr.json`, field `baseline_perf_us`.

On David, direct `aclnnSwiGlu` in CANN 9.2 produced the following FP32 vector-core task durations on 2026-11-13. Each median uses the last 12 of 17 synchronized calls with resident buffers under msprof. The read-back is checked against every element of the matching public Torch golden. Case 13 also has a separate finite control on the same shape.

| Case | Input | Fixture | Native blocks | Native µs | CANNBench µs | CANNBench/native |
|---:|---|---|---:|---:|---:|---:|
| 11 | 3×7×13×1018, dim −1 | Public | 24 | 3.5055 | 2.021 | 0.5765× |
| 13 | 11×13×16×67, dim 2 | Public | 18 | 3.1935 | 1.500 | 0.4697× |
| 13 | 11×13×16×67, dim 2 | Finite control | 18 | 3.2305 | — | — |

These measured native durations exceed the published targets on this device/software combination. The finite control validates the nondegenerate arithmetic. Retain the published denominator for generated kernels. The captured bundle of CSV and JSON records had SHA-256 `40b7b66ea008ed54e0baf101e66b6c33689f6919c519db5d6807db4d715b1da7`.

To replay in the target CANN environment, build the driver and run from a new directory with the public evaluator on the Python path. Set `TASK` to the Season 2 SwiGLU task directory. Each fixture needs its own fresh output directory.

```sh
g++ -O2 "$SOURCE/swiglu.cpp" -I"$ASCEND_HOME_PATH/include" \
    -L"$ASCEND_HOME_PATH/lib64" -lopapi -lnnopbase -lascendcl -o bench
python "$SOURCE/replay.py" "$TASK" 11
python "$SOURCE/replay.py" "$TASK" 13
```
