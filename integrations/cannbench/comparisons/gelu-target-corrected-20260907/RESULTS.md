# GeLU hardware results

[CANNBench job job_6589259af036](https://cannbench.com/workspace/jobs/job_6589259af036) — 20/20 correct; status `succeeded`.

All times below are hardware microseconds. Speedup = official reference / candidate; ≥1 means at least as fast as the reference. Links may require CANNBench sign-in.

| Case | Shape | dtype | Mode | Candidate µs | Reference µs | Speedup | Accuracy | Implementation |
|---|---|---|---|---:|---:|---:|---|---|
| [level1/gelu_1](https://cannbench.com/workspace/jobs/job_6589259af036) | [1024, 1024] | float16 | none | 11.8700 | 4.4900 | 0.3783× | True | [promoted-exact-erf](candidate/gelu.py#L43) |
| [level1/gelu_2](https://cannbench.com/workspace/jobs/job_6589259af036) | [2048, 2048] | float32 | none | 132.8100 | 15.3700 | 0.1157× | True | [fp32-exact-tail](candidate/gelu.py#L43) |
| [level1/gelu_3](https://cannbench.com/workspace/jobs/job_6589259af036) | [4096, 4096] | bfloat16 | none | 147.8100 | 30.1400 | 0.2039× | True | [promoted-exact-erf](candidate/gelu.py#L43) |
| [level1/gelu_4](https://cannbench.com/workspace/jobs/job_6589259af036) | [8192, 8192] | float16 | tanh | 206.0800 | 172.9300 | 0.8391× | True | [corrected-tanh](candidate/gelu.py#L37) |
| [level1/gelu_5](https://cannbench.com/workspace/jobs/job_6589259af036) | [8192, 8192] | float32 | tanh | 339.3900 | 387.2650 | 1.1411× | True | [corrected-tanh](candidate/gelu.py#L37) |
| [level1/gelu_6](https://cannbench.com/workspace/jobs/job_6589259af036) | [1023, 1023] | bfloat16 | tanh | 5.8400 | 4.4600 | 0.7637× | True | [corrected-tanh](candidate/gelu.py#L37) |
| [level1/gelu_7](https://cannbench.com/workspace/jobs/job_6589259af036) | [1009, 1021] | float16 | none | 11.9100 | 4.4500 | 0.3736× | True | [promoted-exact-erf](candidate/gelu.py#L43) |
| [level1/gelu_8](https://cannbench.com/workspace/jobs/job_6589259af036) | [1537, 769] | float32 | tanh | 8.7500 | 6.1100 | 0.6983× | True | [corrected-tanh](candidate/gelu.py#L37) |
| [level1/gelu_9](https://cannbench.com/workspace/jobs/job_6589259af036) | [363, 367, 373] | bfloat16 | none | 433.1000 | 117.5300 | 0.2714× | True | [promoted-exact-erf](candidate/gelu.py#L43) |
| [level1/gelu_10](https://cannbench.com/workspace/jobs/job_6589259af036) | [2049, 513] | float16 | tanh | 5.8800 | 4.5600 | 0.7755× | True | [corrected-tanh](candidate/gelu.py#L37) |
| [level1/gelu_11](https://cannbench.com/workspace/jobs/job_6589259af036) | [3, 7, 13, 4001] | float32 | none | 37.2700 | 6.0500 | 0.1623× | True | [fp32-exact-tail](candidate/gelu.py#L43) |
| [level1/gelu_12](https://cannbench.com/workspace/jobs/job_6589259af036) | [1000003] | bfloat16 | tanh | 5.8800 | 4.3800 | 0.7449× | True | [corrected-tanh](candidate/gelu.py#L37) |
| [level1/gelu_13](https://cannbench.com/workspace/jobs/job_6589259af036) | [11, 13, 17, 67, 67] | float32 | none | 342.5200 | 37.4550 | 0.1094× | True | [fp32-exact-tail](candidate/gelu.py#L43) |
| [level1/gelu_14](https://cannbench.com/workspace/jobs/job_6589259af036) | [3, 7, 11, 13, 1009] | float16 | tanh | 11.9400 | 7.7800 | 0.6516× | True | [corrected-tanh](candidate/gelu.py#L37) |
| [level1/gelu_15](https://cannbench.com/workspace/jobs/job_6589259af036) | [512, 2049] | float32 | none | 36.4700 | 5.9800 | 0.1640× | True | [fp32-exact-tail](candidate/gelu.py#L43) |
| [level1/gelu_16](https://cannbench.com/workspace/jobs/job_6589259af036) | [255, 8193] | bfloat16 | none | 22.4800 | 6.2300 | 0.2771× | True | [promoted-exact-erf](candidate/gelu.py#L43) |
| [level1/gelu_17](https://cannbench.com/workspace/jobs/job_6589259af036) | [4097, 511] | float16 | tanh | 9.2300 | 6.3100 | 0.6836× | True | [corrected-tanh](candidate/gelu.py#L37) |
| [level1/gelu_18](https://cannbench.com/workspace/jobs/job_6589259af036) | [2, 511, 2049] | float32 | none | 68.7700 | 8.8600 | 0.1288× | True | [fp32-exact-tail](candidate/gelu.py#L43) |
| [level1/gelu_19](https://cannbench.com/workspace/jobs/job_6589259af036) | [4, 255, 2049] | bfloat16 | tanh | 9.2100 | 6.2600 | 0.6797× | True | [corrected-tanh](candidate/gelu.py#L37) |
| [level1/gelu_20](https://cannbench.com/workspace/jobs/job_6589259af036) | [2, 3, 17, 1024, 101] | float32 | none | 331.0900 | 36.1950 | 0.1093× | True | [fp32-exact-tail](candidate/gelu.py#L43) |

See [CSV](case-results.csv) for previous high-level and historical low-level latencies and geometry, and [sanitized summary](hardware-summary.json) for environment and exact aggregates.

Previous high-level job: [job_cdee9a024da2](https://cannbench.com/workspace/jobs/job_cdee9a024da2); 18 common passing cases, 14 faster, geometric mean of old/new latency ratios 2.4447×.

Historical low-level comparison: [job_7b4caccdc21f](https://cannbench.com/workspace/jobs/job_7b4caccdc21f). This is not the same authored API/style or a controlled matched rerun.

Independent geometric mean over 20 positive passing-case reference/candidate ratios: 0.3533×; arithmetic mean: 0.4636×. API aggregates are preserved verbatim in the sanitized summary.
