# GeLU uniform JIT — repaired R submission

[Official CANNBench job: job_4d1a7b6d0381](https://cannbench.com/workspace/jobs/job_4d1a7b6d0381)

This directory preserves the generalized AscTile GeLU implementation and the exact archive uploaded for the repaired R experiment. The device implementation uses AscTile, with the inline AscendC exception limited to Erf/Erfc. The historical pyasc runtime is pinned to `adadd7d66ed0ee16d33d79487bf584899a26ef1e`.

## Files

- [gelu.py](gelu.py): complete candidate source, byte-identical to `cann_bench/gelu.py` inside the submission archive.
- [submission.zip](submission.zip): original uploaded `shipping/R/diagnostic.zip`, renamed without changing its bytes. It contains the candidate, build inputs, supporting modules, and evaluator wheel.
- [SHA256.json](SHA256.json): verified artifact hashes.

## Recorded hardware result

All **20/20 official cases passed**, with **0 anti-cheat failures**. The independently calculated geometric-mean speedup versus the benchmark reference is **0.483579×**; values above 1 indicate faster execution. The API field named `geometric_mean_speedup` reported **0.529936×**, which equals the arithmetic mean and is not used as the geometric mean here.

These are historical hardware results. Publishing these artifacts does not perform a new benchmark or establish qualification under a newer runtime or evaluator. The later PR575 geometry experiment recorded a lower geometric mean of 0.437277×.

## Verification and provenance

Both files were retrieved from the preserved Linux experiment at `integrations/cannbench/comparisons/gelu-uniform-jit-20260916/repair/`. Their SHA256 values match the retained handoff record and the files on the VM. ZIP integrity was checked, and its embedded candidate matches the standalone source exactly.

```text
ba09d1e8722ea07be322776bd0be9364cfb41943298999e9cfe1e47da85ec4c6  gelu.py
77bfd2e7f2661dafe149590be93c8364688455ecf79d394349d9d37cb49d18c3  submission.zip
```
