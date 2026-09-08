# GeLU: matched tiling and reuse-allocation follow-up

Pinned pyasc v2: `adadd7d66ed0ee16d33d79487bf584899a26ef1e`. Mathematics is unchanged from the [20/20 corrected target run](../gelu-target-corrected-20260907/README.md); only high-level launch/tiling/JIT configuration is tuned. No compiler or low-level-kernel changes.

Hardware: [job_7389707a7850](https://cannbench.com/workspace/jobs/job_7389707a7850) — correctness; hardware results pending.

## Per-case navigation and actual launch configuration

`tile_shape=[N]` is a one-dimensional tile of the flattened input, not a row of the original shape. `AIV launched/useful` distinguishes the launch count from the number of partitions containing data. Useful count is analytically reconstructed, not hardware occupancy telemetry. This vector kernel launches AIV blocks, not AIC matrix blocks. All entries explicitly state reuse_alloc. Static allocation is requested as None and resolves to enabled; insert_sync=True, opt_level=3, debug=False. UB is compiler-reported per-specialization storage; the available budget is 248 KiB (253952 B), not a target to fill artificially.

| Case | Shape | dtype | Mode | tile_shape | Unroll | AIV launched/useful | reuse_alloc | VF fusion | UB KiB / budget % | Previous µs | New µs | Reference µs | Speedup | Accuracy | Kernel |
|---|---|---|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---|---|
| [1](https://cannbench.com/workspace/jobs/job_7389707a7850) | [1024, 1024] | float16 | none | [8192] | 2 | 72/64 | 1 | False | 96 / 38.7% | 11.8700 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |
| [2](https://cannbench.com/workspace/jobs/job_7389707a7850) | [2048, 2048] | float32 | none | [5120] | 1 | 72/69 | 1 | True | 240 / 96.8% | 132.8100 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |
| [3](https://cannbench.com/workspace/jobs/job_7389707a7850) | [4096, 4096] | bfloat16 | none | [8192] | 2 | 72/71 | 1 | False | 96 / 38.7% | 147.8100 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |
| [4](https://cannbench.com/workspace/jobs/job_7389707a7850) | [8192, 8192] | float16 | tanh | [8192] | 2 | 72/72 | 1 | False | 128 / 51.6% | 206.0800 | pending | pending | pending | pending | [source](candidate/gelu.py#L37) |
| [5](https://cannbench.com/workspace/jobs/job_7389707a7850) | [8192, 8192] | float32 | tanh | [15872] | 2 | 72/72 | 1 | False | 248 / 100.0% | 339.3900 | pending | pending | pending | pending | [source](candidate/gelu.py#L37) |
| [6](https://cannbench.com/workspace/jobs/job_7389707a7850) | [1023, 1023] | bfloat16 | tanh | [8192] | 2 | 72/64 | 1 | False | 128 / 51.6% | 5.8400 | pending | pending | pending | pending | [source](candidate/gelu.py#L37) |
| [7](https://cannbench.com/workspace/jobs/job_7389707a7850) | [1009, 1021] | float16 | none | [8192] | 2 | 72/63 | 1 | False | 96 / 38.7% | 11.9100 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |
| [8](https://cannbench.com/workspace/jobs/job_7389707a7850) | [1537, 769] | float32 | tanh | [15872] | 2 | 72/38 | 1 | False | 248 / 100.0% | 8.7500 | pending | pending | pending | pending | [source](candidate/gelu.py#L37) |
| [9](https://cannbench.com/workspace/jobs/job_7389707a7850) | [363, 367, 373] | bfloat16 | none | [8192] | 2 | 72/72 | 1 | False | 96 / 38.7% | 433.1000 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |
| [10](https://cannbench.com/workspace/jobs/job_7389707a7850) | [2049, 513] | float16 | tanh | [8192] | 2 | 72/65 | 1 | False | 128 / 51.6% | 5.8800 | pending | pending | pending | pending | [source](candidate/gelu.py#L37) |
| [11](https://cannbench.com/workspace/jobs/job_7389707a7850) | [3, 7, 13, 4001] | float32 | none | [5120] | 1 | 72/72 | 1 | True | 240 / 96.8% | 37.2700 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |
| [12](https://cannbench.com/workspace/jobs/job_7389707a7850) | [1000003] | bfloat16 | tanh | [8192] | 2 | 72/62 | 1 | False | 128 / 51.6% | 5.8800 | pending | pending | pending | pending | [source](candidate/gelu.py#L37) |
| [13](https://cannbench.com/workspace/jobs/job_7389707a7850) | [11, 13, 17, 67, 67] | float32 | none | [5120] | 1 | 72/72 | 1 | True | 240 / 96.8% | 342.5200 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |
| [14](https://cannbench.com/workspace/jobs/job_7389707a7850) | [3, 7, 11, 13, 1009] | float16 | tanh | [8192] | 2 | 72/62 | 1 | False | 128 / 51.6% | 11.9400 | pending | pending | pending | pending | [source](candidate/gelu.py#L37) |
| [15](https://cannbench.com/workspace/jobs/job_7389707a7850) | [512, 2049] | float32 | none | [5120] | 1 | 72/69 | 1 | True | 240 / 96.8% | 36.4700 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |
| [16](https://cannbench.com/workspace/jobs/job_7389707a7850) | [255, 8193] | bfloat16 | none | [8192] | 2 | 72/64 | 1 | False | 96 / 38.7% | 22.4800 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |
| [17](https://cannbench.com/workspace/jobs/job_7389707a7850) | [4097, 511] | float16 | tanh | [8192] | 2 | 72/64 | 1 | False | 128 / 51.6% | 9.2300 | pending | pending | pending | pending | [source](candidate/gelu.py#L37) |
| [18](https://cannbench.com/workspace/jobs/job_7389707a7850) | [2, 511, 2049] | float32 | none | [5120] | 1 | 72/69 | 1 | True | 240 / 96.8% | 68.7700 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |
| [19](https://cannbench.com/workspace/jobs/job_7389707a7850) | [4, 255, 2049] | bfloat16 | tanh | [8192] | 2 | 72/64 | 1 | False | 128 / 51.6% | 9.2100 | pending | pending | pending | pending | [source](candidate/gelu.py#L37) |
| [20](https://cannbench.com/workspace/jobs/job_7389707a7850) | [2, 3, 17, 1024, 101] | float32 | none | [5120] | 1 | 72/72 | 1 | True | 240 / 96.8% | 331.0900 | pending | pending | pending | pending | [source](candidate/gelu.py#L43) |

Previous timings: [job_6589259af036](https://cannbench.com/workspace/jobs/job_6589259af036). Cross-run comparisons are observational, not a repeated controlled A/B. [CSV](case-results.csv), [host partition audit](evidence/dispatch.json), [package verification](evidence/package-x86.json).

## Matched local screen

One AIV block per timing probe. FP32 exact uses 8192 logical elements; other routes use 32768. Compare only identical dtype/mode/size. Three warmed Model samples, with output checks, are ticks, not hardware microseconds. Tail/padding overhead is included. The initial stopped 65536-element probes are excluded.

| dtype | Mode | Elements | Tile | Unroll | reuse_alloc | VF | UB KiB | Median ticks | Valid | Evidence |
|---|---|---:|---:|---:|---:|---|---:|---:|---|---|
| float16 | none | 32768 | 15872 | 2 | 2 | 0 | 248 | 22524 | True | [result](evidence/perf-n32768-float16-none-t15872-u2-r2-vf0/result.json) |
| float16 | none | 32768 | 16384 | 2 | 1 | 0 | 192 | 19210 | True | [result](evidence/perf-n32768-float16-none-t16384-u2-r1-vf0/result.json) |
| float16 | none | 32768 | 20480 | 2 | 1 | 0 | 240 | 22454 | True | [result](evidence/perf-n32768-float16-none-t20480-u2-r1-vf0/result.json) |
| float16 | none | 32768 | 8192 | 2 | 1 | 0 | 96 | 19255 | True | [result](evidence/perf-n32768-float16-none-t8192-u2-r1-vf0/result.json) |
| float16 | tanh | 32768 | 15872 | 2 | 1 | 0 | 248 | 10436 | True | [result](evidence/perf-n32768-float16-tanh-t15872-u2-r1-vf0/result.json) |
| float16 | tanh | 32768 | 20480 | 2 | 2 | 0 | 240 | 10731 | True | [result](evidence/perf-n32768-float16-tanh-t20480-u2-r2-vf0/result.json) |
| float16 | tanh | 32768 | 8192 | 2 | 1 | 0 | 128 | 7546 | True | [result](evidence/perf-n32768-float16-tanh-t8192-u2-r1-vf0/result.json) |
| float32 | tanh | 32768 | 15872 | 2 | 1 | 0 | 248 | 10080 | True | [result](evidence/perf-n32768-float32-tanh-t15872-u2-r1-vf0/result.json) |
| float32 | tanh | 32768 | 15872 | 2 | 2 | 0 | 248 | 11688 | True | [result](evidence/perf-n32768-float32-tanh-t15872-u2-r2-vf0/result.json) |
| float32 | none | 8192 | 1024 | 1 | 0 | 1 | 136.125 | 16804 | True | [result](evidence/perf-n8192-float32-none-t1024-u1-r0-vf1/result.json) |
| float32 | none | 8192 | 1024 | 1 | 1 | 1 | 48 | 15619 | True | [result](evidence/perf-n8192-float32-none-t1024-u1-r1-vf1/result.json) |
| float32 | none | 8192 | 1024 | 1 | 2 | 1 | 44 | 15974 | True | [result](evidence/perf-n8192-float32-none-t1024-u1-r2-vf1/result.json) |
| float32 | none | 8192 | 5120 | 1 | 1 | 1 | 240 | 14162 | True | [result](evidence/perf-n8192-float32-none-t5120-u1-r1-vf1/result.json) |
| float32 | none | 8192 | 5120 | 1 | 2 | 1 | 220 | 14215 | True | [result](evidence/perf-n8192-float32-none-t5120-u1-r2-vf1/result.json) |

### Additional aligned geometry/unrolling control (not a second submission)

FP32 exact, 15360 identical logical elements, one AIV block, reuse_alloc=1 and VF fusion enabled. The input is divisible by all three tiles; no padding confound. Each row passed finite-input accuracy and three warmed repeated checks. The tile3840/unroll2 alternative has NOT passed the full special-value/multicore-tail qualification and is NOT in the submitted archive.

| tile_shape | Unroll | UB KiB | Warmed ticks | Median ticks | Role |
|---|---:|---:|---|---:|---|
| [1024] | 1 | 48 | [28105, 28218, 28032] | 28105 | previous geometry |
| [5120] | 1 | 240 | [20495, 20393, 20506] | 20495 | submitted geometry |
| [3840] | 2 | 240 | [18617, 18623, 18618] | 18618 | unsubmitted, qualification pending |

On this aligned workload, submitted tile5120/unroll1 is 1.3713× faster than the previous tile1024/unroll1. At the same 240 KiB UB, tile3840/unroll2 is another 1.1008× faster locally than tile5120/unroll1. This is further evidence that UB fill alone is not the objective; scheduling/unrolling matters. These ratios are NOT CANNBench reference speedups. [Raw evidence](evidence/interaction-summary.json).


## Interpretation and limits

Launch/partition finding: case8 ([1537,769], FP32 tanh, tile_shape=[15872]) launches 72 AIV blocks but only 38 receive data under the current contiguous assignment. Each partition reserves two tiles; maximum UB usage does not imply balanced work. A size-aware tile choice or balanced/cyclic tile assignment is a high-level follow-up, not necessarily a compiler-pass change. Fewer idle blocks alone also does not prove lower latency: the longest active partition and memory traffic matter. This submission deliberately isolates one geometry constant; it does not claim complete launch/partition optimization.

Selected configuration: FP32 exact changes tile_shape from [1024] to [5120], keeping reuse_alloc=1, unroll=1 and VF fusion enabled. On the matched 8192-element one-block screen this improves median ticks from 15619 to 14162 (1.1029×); static UB increases from 48 to 240 KiB. Mode2 at [5120] gives 14215 ticks, so it is not selected. Unroll2 at [5120] needs 320 KiB and was pruned before simulation. All other routes keep their previous geometry and reuse_alloc=1. [Selection evidence](evidence/selection-reasons.json).

For FP16 exact, [8192]/96 KiB takes 19255 ticks, [16384]/192 KiB takes 19210, and [20480]/240 KiB takes 22454 on 32768 logical elements. The aligned larger tile differs by only 0.23%; the maximal-UB tile pays padding cost. FP16 tanh [15872]/248 KiB takes 10436 ticks versus 7546 for [8192]/128 KiB. These small-input observations do not prove the ordering at every full hardware shape. A pragmatic 3% selection margin retains the proven configuration for near-ties; it is not a statistical confidence interval. BF16 inherits the same low-precision geometry and is independently numerically qualified, not independently perf-tuned in this screen.

UB is not a utilization target by itself. At identical FP32 exact tile1024, disabling reuse increases storage from 48 KiB to 136.125 KiB without increasing useful work. At tile5120, mode1 needs 240 KiB, mode2 needs 220 KiB. Conversely FP16 exact at tile8192 needs 96 KiB with mode1 but 128 KiB with mode2; FP16 tanh uses 128 versus 96 KiB. More occupied UB can represent redundant intermediates, not better tiling. The earlier cross-case table mixed mathematical routes, dtypes and sizes and therefore did not establish causation.

[Compile matrix](evidence/matrix-summary.json): 68 unique configurations, 50 within UB; overflow cases were not simulated. Two additional controls checked aligned FP16 exact tile16384 and FP32 exact tile5120/unroll2 (overflow). Fourteen matched perf configurations passed. Static allocation remains enabled and required synchronization remains on. [Review adjudication](REVIEW_ADJUDICATION.md) explains which OpenCode claims were accepted or rejected. Final local qualification samples all six dtype/mode routes with special values, numerical boundaries, guarded multi-core tails and cached repeated execution; it is not full-shape device execution. Three harness-import failures are retained in [history](evidence/history-import-shadowing/README.md) and excluded.

The authored compute-kernel AST is identical to the previous submission; only one host geometry constant changed. [Package manifest](evidence/package.json) pins source/runtime/wheel/archive hashes; QEMU verified the installed evaluator bytes, host dispatch, IR passes and AscendC translation for all 20 cases, not NPU execution. Native Model checks compile and execute reduced inputs. Six compiled specializations belong to one authored GeLU kernel. The exact [integration gate](evidence/tools/local_compile_gate.py) and [source contract](evidence/tools/source_contract.py) are archived, and the verifier imports those snapshots rather than unpublished workspace changes. [Fresh task reconciliation](evidence/task-reconciliation.json) confirms the current 20 official definitions and golden source. Scripts still require the pinned runtime/toolchain and official precision checker described by the previous report; this is not a standalone installer.

Transport note: two HTTP/2 connections were stopped with provably incomplete archive bodies (6.68 MB and 7.93 MB read out of 25.90 MB). After each stop, absence of a job and unchanged credits were reconciled. The final transport uses the identical archive/tag via the public server address and HTTP/1.1, retaining HTTPS certificate validation. No complete/ambiguous upload is retried automatically; there are no further automatic retries. This is one intended evaluation, not multiple completed benchmark submissions.
