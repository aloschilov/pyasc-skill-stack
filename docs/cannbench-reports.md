# CANNBench evaluation reports

## GeLU tiling / reuse-allocation follow-up (2026-09-08)

- [Report: all 20 tile shapes, AIV launched/useful counts, reuse_alloc and UB](../integrations/cannbench/comparisons/gelu-tiling-reuse-20260908/README.md)
- [Per-case CSV](../integrations/cannbench/comparisons/gelu-tiling-reuse-20260908/case-results.csv)

Local qualification passed: 14/14 perf configurations, 12/12 numerical/tail checks,
20/20 official compile/dispatch cases. FP32 exact selects tile5120/reuse1;
other routes retain their measured baseline. Hardware evaluation
[job_7389707a7850](https://cannbench.com/workspace/jobs/job_7389707a7850)
succeeded: 20/20 correct, zero anti-cheat failures, score 62.3922.
The six changed FP32 exact cases improved 1.5348× against the previous run;
the fourteen unchanged cases averaged 1.0064×. Independent geometric mean
reference/new improved from 0.3533× to 0.4035×; only case5 exceeds reference
(1.1450×). One submission credit was charged. The report also includes an
aligned local unrolling control, explicitly separated from hardware results.

This index collects the CANNBench evaluation reports and their corresponding
remote runs. CANNBench job pages may require the account used for the private
submissions.

## GeLU upstream diagnostics — 2026-09-07

- [Fresh local checks, OpenCode review and diagnostic submission](../integrations/cannbench/comparisons/gelu-upstream-diagnostics-20260907/README.md)
- Confirmed [JIT option-discovery bug](https://gitcode.com/compiler-team/pyasc/issues/4)
  and [VF dead-store optimization issue](https://gitcode.com/compiler-team/pyasc/issues/5).
- [Exact-GeLU accuracy/API question](https://gitcode.com/compiler-team/pyasc/issues/6),
  with a standalone executed reproducer and verified FP32 bit-distance evidence.
- Serial one/eight-core repeat controls pass; earlier timeouts do not establish a deadlock.
- High-level diagnostic [job_cdee9a024da2](https://cannbench.com/workspace/jobs/job_cdee9a024da2)
  completed: **18/20 correct**, exact-FP32 cases11/20 fail cancellation checks;
  no compile/runtime or anti-cheat failures. Passing-case geometric mean
  **0.16106×**; API-labelled aggregate **0.17034×** is their arithmetic mean.
  One credit spent, four remain. Full case navigation and evidence are in the report.

## GeLU from supplied reference archive — 2026-09-07

- [Candidate, archive provenance and local comparisons](../integrations/cannbench/comparisons/gelu-reference-20260907/README.md)
- 20/20 dispatches compile. Reduced/dense numerical checks pass FP16/BF16
  routes but fail FP32 exact and literal tanh. The prior stable tanh-exp
  control passes the dense FP32 check. No submission or credit spent.

## High-level AscTile GeLU and JIT flags — 2026-09-07

- [Report and case navigation](../integrations/cannbench/comparisons/gelu-asctile-jit-20260907/README.md)
- [Lowering diagnosis](../integrations/cannbench/comparisons/gelu-asctile-jit-20260907/LOWERING.md)
- Pin: `adadd7d66ed0ee16d33d79487bf584899a26ef1e`.
- Native and CPython 3.12/x86_64 runtimes are built and verified at the pin.
  20/20 official dispatches compile on each; all 108 dtype/mode/JIT configurations
  compile with synchronization verification. Dense FP32 exact negative-tail
  checks fail for all 18 JIT settings. No submission or credit spent.
- Historical [i03](https://cannbench.com/workspace/jobs/job_a375a6e244ca) and
  [i04](https://cannbench.com/workspace/jobs/job_7b4caccdc21f) are lower-level
  comparisons, not results for the new high-level candidate.

## Full pilot run

- HTML and PDF reports: archived. `git checkout archive/campaign-2026 -- docs/cannbench-pyasc-skill-stack-report-2026-09-02.html docs/cannbench-pyasc-skill-stack-report-2026-09-02.pdf`
- [Main 9-operator run `job_cd51d6c2ca67`](https://cannbench.com/workspace/jobs/job_cd51d6c2ca67)
- [SwiGLU fallback run `job_0f5fdee4443a`](https://cannbench.com/workspace/jobs/job_0f5fdee4443a)

## Target-intersection comparison and corrective reruns

- [Markdown report](../integrations/cannbench/comparisons/target-intersection-20260902/reports/final-cannbench-reruns.md)
- [HTML report](../integrations/cannbench/comparisons/target-intersection-20260902/reports/final-cannbench-reruns.html)
- [PDF report](../integrations/cannbench/comparisons/target-intersection-20260902/reports/final-cannbench-reruns.pdf)
- [Initial target-derived run `job_79c05b96ed9d`](https://cannbench.com/workspace/jobs/job_79c05b96ed9d)
- [No-skills run `job_75a7fee4ae6f`](https://cannbench.com/workspace/jobs/job_75a7fee4ae6f)
- [With-skills run `job_ae3bfdefd087`](https://cannbench.com/workspace/jobs/job_ae3bfdefd087)
- [Manual corrected run `job_a3e9153a2930`](https://cannbench.com/workspace/jobs/job_a3e9153a2930)
- [Skills-corrected run `job_028f05fd8ec8`](https://cannbench.com/workspace/jobs/job_028f05fd8ec8)

## Current four-operator comparison

- [Markdown report](../integrations/cannbench/comparisons/four-operator-20260903/reports/four-operator-comparison.md)
- [HTML report](../integrations/cannbench/comparisons/four-operator-20260903/reports/four-operator-comparison.html)
- [PDF report](../integrations/cannbench/comparisons/four-operator-20260903/reports/four-operator-comparison.pdf)
- [CANNBench jobs workspace](https://cannbench.com/workspace/jobs)

The four-operator jobs are waiting for daily submission credits. Its report
generator adds direct run links automatically as each `job_id` is recorded.

## Handwritten GeLU deep dive

- [Live deep-dive report](../integrations/cannbench/comparisons/gelu-handwritten-deepdive-20260903/REPORT.md)
- [Iteration 01 — `job_24726123eebe`](https://cannbench.com/workspace/jobs/job_24726123eebe)
- [Iteration 02 — `job_bcf486ff6371`](https://cannbench.com/workspace/jobs/job_bcf486ff6371)

Iteration 02 repaired the low-level C310 launch ABI and proved exact cases
1–3 correct on 950PR. Its large VF/reuse tanh route then hit vector-core
timeout 507034. A safe low-level tanh iteration is prepared for the next credit
reset and remains ahead of the other operator submissions.
