# GeLU performance investigation — 2026-09-07

## Scope and reference

Improve the handwritten GeLU path before spending another CANNBench credit.
Reference: [iteration03 hardware job](https://cannbench.com/workspace/jobs/job_a375a6e244ca),
20/20 correct, official aggregate speedup **0.4748251724×**, score 62.0711356991.
The official aggregate is not replaced by a locally recomputed mean of case ratios.

Both reference and new candidate use pyasc v2 commit
`0a631f70968c3cb7c33ce45330a85768dd5a6f06`.
An isolated native AArch64/CPython3.11 runtime was built from the exact archived
source using the existing LLVM reference installation; the remote package reuses
the proven CPython3.12/x86_64 wheel. The working pyasc-fork library was not replaced.

## Hardware follow-up

Iteration04 was accepted as [job_7b4caccdc21f](https://cannbench.com/workspace/jobs/job_7b4caccdc21f).
This is a private, single-operator GeLU submission, tag
`pyasc-v2-gelu-register-fma-db8192-i04-20260907`.
**Completed successfully on NPU: 20/20 correct, zero anti-cheat failures.**
Official aggregate speedup: **0.474825× → 0.582646×** (+22.7%); score:
**62.0711 → 64.4755**. Submission used one credit: 6 before, 5 after.
Baseline anchors were unchanged between these two runs.

18 of 20 cases are faster than iteration03. FP32 tanh at 8192×8192 reaches
**1.103256×**; FP16 tanh at 8192×8192 reaches **0.930983×**.
The remaining tanh cases reach 0.771–0.884×. Exact mode remains the main bottleneck:
FP32 0.383–0.454× and BF16 0.198–0.266×.
FP16 exact regressed from about 0.41× to 0.33×: the more accurate FP32 register
polynomial is slower than the old native-half erf path for these two cases.
**Overall 1× has not been reached.** The old large-shape tanh timeout did not recur.

Sources: [full results](remote_runs/results.json), [logs](remote_runs/logs.json),
[per-case timings and baseline anchors](hardware-comparison.csv),
[aggregate comparison](hardware-comparison.json).

`candidate_next/gelu.py` is a separate **unsubmitted hybrid** that restores the
already measured iteration03 FP16 exact path and retains iteration04 for the
other five routes. This avoids treating the two FP16 regressions as an improvement.
Its local checks are `next-compile-x86.json` and `next-gate-exact-f16.json`;
it has no measured whole-submission score. The measured iteration04 source/archive
remain immutable. No second tuning credit was spent.

Next substantial performance work should target a shorter exact-mode math path
(especially removal/reduction of division/exp latency), with fresh numerical and
CaModel gates before hardware. A blanket core-count increase or unconditional
replacement of exact by tanh is not a justified fix.

## Case navigation: shapes, dtypes and implementations

All rows below are **measured on NPU and correct in both runs**:
[i03](https://cannbench.com/workspace/jobs/job_a375a6e244ca) and
[i04](https://cannbench.com/workspace/jobs/job_7b4caccdc21f).
Case numbers are the suffixes of `level1/gelu_N`. Shapes/dtypes/modes come from
[CANNBench cases](../../tasks/gelu/cases.yaml); timings and speedups come from
[the per-case comparison](hardware-comparison.csv).
`none` means exact-mode GeLU. Speedup is **CANNBench baseline time / i04 time**,
not i04 versus i03; values above 1× beat that baseline. Times are in microseconds.

<!-- case-navigation:start -->
| Case | Input shape | dtype | Mode | i03 (µs) | i04 (µs) | i04 speedup vs baseline | Implementation |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | 1024 × 1024 | float16 | `none` | 10.85 | 13.35 | 0.336× | [i03](baseline/gelu.py#L189) / [i04](candidate/gelu.py#L130) |
| 2 | 2048 × 2048 | float32 | `none` | 51.03 | 38.99 | 0.394× | [i03](baseline/gelu.py#L32) / [i04](candidate/gelu.py#L134) |
| 3 | 4096 × 4096 | bfloat16 | `none` | 206.00 | 152.00 | 0.198× | [i03](baseline/gelu.py#L73) / [i04](candidate/gelu.py#L132) |
| 4 | 8192 × 8192 | float16 | `tanh` | 253.08 | 185.75 | 0.931× | [i03](baseline/gelu.py#L149) / [i04](candidate/gelu.py#L137) |
| 5 | 8192 × 8192 | float32 | `tanh` | 456.39 | 351.02 | 1.103× | [i03](baseline/gelu.py#L108) / [i04](candidate/gelu.py#L141) |
| 6 | 1023 × 1023 | bfloat16 | `tanh` | 6.30 | 5.31 | 0.840× | [i03](baseline/gelu.py#L149) / [i04](candidate/gelu.py#L139) |
| 7 | 1009 × 1021 | float16 | `none` | 10.98 | 13.41 | 0.332× | [i03](baseline/gelu.py#L189) / [i04](candidate/gelu.py#L130) |
| 8 | 1537 × 769 | float32 | `tanh` | 9.04 | 6.91 | 0.884× | [i03](baseline/gelu.py#L108) / [i04](candidate/gelu.py#L141) |
| 9 | 363 × 367 × 373 | bfloat16 | `none` | 602.00 | 442.58 | 0.266× | [i03](baseline/gelu.py#L73) / [i04](candidate/gelu.py#L132) |
| 10 | 2049 × 513 | float16 | `tanh` | 6.44 | 5.47 | 0.834× | [i03](baseline/gelu.py#L149) / [i04](candidate/gelu.py#L137) |
| 11 | 3 × 7 × 13 × 4001 | float32 | `none` | 15.98 | 13.34 | 0.454× | [i03](baseline/gelu.py#L32) / [i04](candidate/gelu.py#L134) |
| 12 | 1000003 | bfloat16 | `tanh` | 6.44 | 5.39 | 0.813× | [i03](baseline/gelu.py#L149) / [i04](candidate/gelu.py#L139) |
| 13 | 11 × 13 × 17 × 67 × 67 | float32 | `none` | 131.86 | 96.24 | 0.389× | [i03](baseline/gelu.py#L32) / [i04](candidate/gelu.py#L134) |
| 14 | 3 × 7 × 11 × 13 × 1009 | float16 | `tanh` | 13.02 | 10.09 | 0.771× | [i03](baseline/gelu.py#L149) / [i04](candidate/gelu.py#L137) |
| 15 | 512 × 2049 | float32 | `none` | 15.73 | 13.38 | 0.447× | [i03](baseline/gelu.py#L32) / [i04](candidate/gelu.py#L134) |
| 16 | 255 × 8193 | bfloat16 | `none` | 27.54 | 24.09 | 0.259× | [i03](baseline/gelu.py#L73) / [i04](candidate/gelu.py#L132) |
| 17 | 4097 × 511 | float16 | `tanh` | 9.84 | 7.82 | 0.807× | [i03](baseline/gelu.py#L149) / [i04](candidate/gelu.py#L137) |
| 18 | 2 × 511 × 2049 | float32 | `none` | 27.87 | 23.14 | 0.383× | [i03](baseline/gelu.py#L32) / [i04](candidate/gelu.py#L134) |
| 19 | 4 × 255 × 2049 | bfloat16 | `tanh` | 9.98 | 7.67 | 0.816× | [i03](baseline/gelu.py#L149) / [i04](candidate/gelu.py#L139) |
| 20 | 2 × 3 × 17 × 1024 × 101 | float32 | `none` | 130.29 | 91.22 | 0.397× | [i03](baseline/gelu.py#L32) / [i04](candidate/gelu.py#L134) |
<!-- case-navigation:end -->

The i04 links point to each dtype/mode specialization in the submitted source.
Shared implementation: [exact polynomial/FMA](candidate/gelu.py#L15),
[tanh math](candidate/gelu.py#L69), [register casts](candidate/gelu.py#L89),
[tiling and tails](candidate/gelu.py#L119), [host launch](candidate/gelu.py#L144).
All i04 cases use tile8192, unroll2 and up to 72 vector blocks.
The [unsubmitted hybrid FP16 exact path](candidate_next/gelu.py#L145) is **not**
the implementation measured in rows 1 and 7; it has local validation only.

## Findings supported by source and experiments

1. The original target example is not the entire benchmark contract. CANNBench
   needs exact and tanh modes across FP16, BF16 and FP32, arbitrary tails, NaN/Inf.
   Correctness extensions in iteration03 introduced different, slower execution paths.
2. Iteration03 performs separate full-tile vector operations and, for low precision,
   full-tile casts through UB. Its exact FP32/BF16 path calls generic `erfc`.
3. `vf_fusion=True` alone did not produce efficient register-only code for the
   attempted exact formula: [generated source](translated/fused/01.cpp) materializes
   intermediate arrays in UB. The pure-DSL candidate was slower, not selected.
   This finding concerns this expression/pipeline, not every use of VF fusion.
4. Register math, fused multiply-add, in-register casts and overlapping tile work
   each address actual overhead. Increasing the core count alone cannot remove it.
5. Unroll=2 at tile=13824 allocates **331776 bytes UB**, above the local 253952-byte
   capacity. Tile8192 fits: **196608 bytes FP32 / 98304 bytes FP16/BF16**.
6. The first custom exact formula returned NaN for +Inf. A value-based register
   selection fixes that edge case. This was discovered locally, before submission.

The installed C310 `AscendC::Gelu` primitive is a tanh approximation. It cannot
simply replace exact-mode GeLU. Absence of a named GeLU primitive in pyasc also
does not by itself prove why a particular kernel is slow.

## Selected implementation

[candidate/gelu.py](candidate/gelu.py) is **handwritten pyasc with embedded AscendC
register code via the public `asctile.inline_vf` API**, not a pure Python-DSL result
and not an AI-with-skills submission. The host, JIT, allocation, DMA, tails and
launch remain pyasc. No framework/baseline GeLU is called by the kernel.

- Exact mode: cancellation-free erfc approximation, FP32 intermediates, Horner FMA.
- Tanh mode: cancellation-free single-exponential expression.
- FP16/BF16 conversion stays in registers, not separate full-tile UB arrays.
- Tile8192, unroll2, reuse_alloc0; up to 72 vector blocks.
- Disjoint `real_shape` tail stores replace overlapping last-tile writes.
- No input-data-dependent host dispatch or benchmark/checker detection.

Shorter polynomial alternatives were screened in [numerical-gate.json](numerical-gate.json)
but not selected; the conservative stable polynomial was retained.

## Local evidence and limitations

All 20 official dispatches / six unique specializations pass the
[x86/QEMU code-generation and ABI gate](final-compile-x86.json), without a hidden
FFTS argument. This gate alone does not invoke Bisheng or execute a kernel.
Native CaModel tests additionally compile actual C310 binaries with Bisheng.

`final-gate-*.json` records the six final mode/dtype routes, sampling every official
case's value range, two invocations each, plus nonfinite and zero values.
**These are reduced-size numerical tests, not execution of all full benchmark shapes.**
Separate prime-tail, multi-tile and multi-core probes are retained in this folder.
NaN/Inf arithmetic produces expected simulator diagnostic messages; acceptance is
based on actual output comparisons, not absence of those messages.

Illustrative CaModel comparisons, not CANNBench speedups:

- FP32 exact, 13824 values, one block: iteration03 16587 ticks; pure-DSL fused
  tile1024 57875; register NR tile13824 14812; register FMA tile13824 13256.
- FP32 tanh, 13824 values, one block: iteration03 10070; register-only math 8572.
- FP32 exact, 32771 values, tile8192, one block: sequential 33264;
  unroll2 22703. The sequential version also includes the +Inf fix (two register
  instructions); inputs for this timing probe are finite.
- Unroll2 tile10240: 25055 ticks on the same 32771-element probe, so the larger
  tile was not selected.
- Final candidate, 32771 values, four blocks: 12570 ticks, correctness passed.
  This demonstrates scaling locally, not optimality of 72 blocks on real hardware.
- Final candidate, 13824 values, one block: **13813 versus 16587 reference ticks**
  (16.7% fewer ticks / 1.201× relative to iteration03 in this local probe).

Final reduced numerical coverage: **20/20 sampled cases, all six route checks
passed**, with two invocations per route. Final candidate SHA256:
`62cbcd7f9c20266f94eb497caa591e5ce0f7b0852949041bcf5548e952be01d9`.
The final unrolled FP16 tanh stress probe also passed: 32771 values spanning
[-65504,65504], one block, two invocations, exact agreement after FP16 rounding
([evidence](final-pipeline-stress-f16.json)). The self-contained wheel was built,
installed, and its CPython3.12/x86_64 native runtime imported successfully under
QEMU ([evidence](wheel-validation.json)).

CaModel is **Ascend950PR_9599 / CANN9.0**, while the prior remote runner is
**950PR_9589 / CANN9.1**. Simulator ticks are not NPU microseconds. Full-shape
performance and absence of a large-shape timeout still require CANNBench.
Some early timing probes captured a source hash after execution while development
continued; use the immutable candidate and final gate hashes for submission provenance.

## Reproduction and credit safety

Use `sim.sh --package candidate --dtype float32 --mode none --case-group --repeat 2`
for a reduced numerical check; `--size`, `--tile`, `--cores` select timing probes.
The isolated runtime and existing CANN/LLVM installations are prerequisites.
`numerical_gate.py` screens formulas on CPU; it is not part of the submission.

`package_submission.py` requires six passing final gates, coverage IDs 1–20,
current candidate hashes, a passing pinned-v2 x86 gate, and the exact prior runtime
archive hash. It creates an immutable GeLU-only ZIP and `MANIFEST.json`.
`submission.py submit` checks existing jobs first and records upload intent before
POST. An ambiguous upload is **not automatically retried**. `submission.py status`
only collects existing results and never spends a credit.

OpenCode workers: GLM5.2 and Qwen3.7-max provided reviews; the Alibaba Qwen3.8-max
provider returned HTTP401, and the pipeline worker reached its bounded timeout.
Raw worker logs and experimental worker outputs are retained locally, not published.
Unmeasured performance projections and overly broad worker claims are not accepted as evidence here.
The selected implementation and verification were completed by the main agent.
