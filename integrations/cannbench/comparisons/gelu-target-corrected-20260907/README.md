# Corrected target-style AscTile GeLU

Final combined candidate, pyasc v2 pinned to `adadd7d66ed0ee16d33d79487bf584899a26ef1e`.
This is a high-level kernel authored from the target pattern and reviewed through OpenCode;
it is not a new skill/no-skill generation comparison.

**Verified CANNBench: [job_6589259af036](https://cannbench.com/workspace/jobs/job_6589259af036)
succeeded, 20/20 cases correct, zero anti-cheat failures.**
One credit has been charged; nine remain after this submission.
The previous [direct-Erf diagnostic](https://cannbench.com/workspace/jobs/job_cdee9a024da2)
passed 18/20. Do not interpret its hardware results as results for this candidate.

## Hardware outcome

[All 20 shapes, dtypes, modes, timings and implementation links](RESULTS.md)
are available as a navigation table and [CSV](case-results.csv).
[Sanitized result summary](hardware-summary.json) preserves environment and aggregates.

- Large FP32 tanh, shape [8192,8192]: **339.39 µs versus 387.265 µs reference,
  1.1411×**. This is the only case at or above 1× in this run.
- Exact FP32 cases 11 and 20, previously failing accuracy, now pass:
  37.27 µs / 0.1623× and 331.09 µs / 0.1093× respectively.
- Versus the previous high-level run: **14/18** common passing cases are
  faster; geometric mean of previous/current latency ratios is **2.4447×**.
  The remaining four common cases regress: FP32 exact 2/13/15/18 pay for
  the stable-tail computation even when their data do not need it.
- Independent geometric mean of all 20 reference/candidate ratios:
  **0.3533×**; tanh (9 cases) **0.7646×**, exact (11 cases) **0.1879×**.
  The API field named `geometric_mean_speedup` is **0.4635686473** here,
  which equals the arithmetic mean of the 20 ratios. Both are retained;
  they must not be presented as the same metric. CANNBench score: **61.8814**.
- Historical low-level [job_7b4caccdc21f](https://cannbench.com/workspace/jobs/job_7b4caccdc21f)
  remains faster overall: the new high-level kernel improves only **6/20**
  case latencies against it; geometric mean of old/new latencies is 0.6830×.
  Its score was 64.4755. The new result is not claimed to supersede that
  implementation on overall performance.

Accuracy across the official set is achieved. **The overall ≥1× performance
target is not achieved.** The reproducible FP32 exact cost is the additional
continued-fraction arithmetic on every tile, including both sides of the
selection, combined with tile1024. The matched local valid-vs-invalid probe
costs 4966 versus 3521 ticks; this isolates a real arithmetic cost, but does
not attribute the entire hardware gap to one pass. Next work should compare
a publicly exposed accurate Erf/erfc primitive and tune exact FP32 geometry
without restoring cancellation or inserting low-level code. No additional
submission was made automatically.

Hardware environment reported by the job: Ascend950PR_9589, CANN9.1.0,
CPython3.12.13, Torch/torch_npu2.10.0. The Docker label still says CANN9.0.0;
the reported runtime version is preserved separately. Cross-run comparisons
are observational, not controlled repeated A/B measurements.

## Implementation and configuration

[Submitted-source candidate](candidate/gelu.py) preserves `gelu(x, approximate="none")`.
All numerical operations use public AscTile. Torch is limited to allocation and metadata.
No inline AscendC, authored register code, compiler changes or JIT monkeypatches.

- Tanh uses the corrected target expression
  `x / (1 + exp(-sqrt(8/pi) * (x + 0.044715*x³)))`.
  FP16/BF16 compute intermediates in FP32 and cast back.
- FP16/BF16 exact uses `(x*0.5)*(1+erf(x/sqrt(2)))`, with FP32 intermediates.
- FP32 exact keeps that central expression for x≥−3 and uses a six-level
  normal-tail Mills-ratio continued fraction for x<−3
  ([mathematical reference](https://dlmf.nist.gov/7.9.E1)).
  This avoids cancellation; it is not the configured ops-nn Erf algorithm.
  Both expressions execute before the tile selection, so sparse negative tails
  do not eliminate the continued-fraction cost.
- Contiguous core partitions explicitly skip unused end tiles.
  Tail copies use `real_shape` and zero padding; all loops keep synchronization.

Configuration selection depends only on dtype/mode and tensor size, not benchmark case IDs:
FP32 exact tile1024/unroll1/VF on; FP32 tanh tile15872/unroll2/VF off;
FP16/BF16 both modes tile8192/unroll2/VF off. Launch is bounded by
`min(72, ceil(numel/tile))`. Static UB is respectively 49152, 253952,
98304 (low-precision exact) and 131072 B (low-precision tanh).

Common requested flags: `reuse_alloc=1, static_alloc=None, insert_sync=True,
opt_level=3, debug=False`. The platform resolves static allocation to enabled.
Invocation-time VF choices are retained by the integration gate and verified by
the generated specializations. Diagnostic compilation uses
`verify_sync=True, always_compile=True`; timed repetitions are warmed/cached
with debugging disabled.

## Local qualification and integration repairs

[Local result index](local-summary.json): all 18 final jobs passed their
numerical/repetition checks with zero unexpected simulator errors.
One-core warmed medians on inputs spanning [−6,2]: FP32 exact 4966 ticks
(2048 elements), FP32 tanh 9358 (31744 elements), FP16/BF16 exact
10363/10369 and tanh 5144/5144 (16384 elements). Different sizes must not be
compared as equal workloads. Three raw timing repetitions are retained.

Native AArch64/CPython3.11: CANN9.0, CaModel Ascend950PR_9599.
The isolated runtime is an extracted source tree, not a Git checkout:
[467 tracked compiler/Python files](evidence/native-source-integrity.json)
were byte-compared with the pinned Git archive, with no differences.
The self-contained evaluator wheel contains the identical pinned source runtime
for CPython3.12/x86_64; its installed library and module bytes are checked under QEMU.
The image's older name does **not** define the imported runtime: the pinned wheel
is installed and byte-verified in a disposable container.

All 20 official host cases pass dispatch and compilation, resolving to six
unique dtype/mode specializations of one authored kernel. This is one GeLU
operator, not 20 different operator implementations.
[Package verification](evidence/package-x86.json) includes
requested compiler flags, specialization constants and UB.
[Artifact hashes](evidence/package.json) identify source, runtime, wheel and ZIP.
Wheels, ZIPs and raw private job responses are not publication artifacts.

An initial archive was explicitly rejected with WEB-ZIP-007 because source
layout metadata `setup.py` was missing. No job or charge was created (10 credits
remained). The corrected ZIP adds that file without changing the qualified
kernel or wheel. The upload ledger prevents retrying an ambiguous POST;
this layout correction is not a second hardware evaluation.

The [fresh task reconciliation](evidence/task-reconciliation.json) confirms
20 cases, matching shape/dtype/mode/notes and identical golden source. The
API serializes the Inf/NaN range bounds as null; the local YAML preserves them.

The final Model suite covers seeded normal values, dense negative tails,
random negative tails, switch boundaries and adjacent floats, samples of every
official range, NaN/Inf, signed zero and finite dtype limits. Separate guarded
tail tests use eight cores, more than one tile per active partition and cached
repeats. Timings use three cached repetitions after warmup.
These are **sampled compiled-kernel checks**, not full official-shape execution
and not NPU latency.

Two integration defects were addressed before upload:

1. The old local compile gate rejected invocation-time JIT options.
   It now captures/replays them and separates specialization keys.
   Regression checks cover option propagation, kernel keyword arguments,
   rejection of unknown flags and distinct keys.
2. The first combined candidate passed output checks but issued zero-length
   DMA on unused end tiles (`mte_gdma_illegal_burst_len_t0`).
   Its [source is preserved as rejected history](history/gelu_zero_length_dma.py).
   The final loop bounds omit those iterations. Unexpected DMA/memory/sync
   diagnostics block submission even when numerical outputs match.
   CaModel's `vec_err_idata_inf_nan_t0` messages during intentional NaN/Inf
   stress are retained separately, with a passing special-value oracle check.

The official oracle is `torch.nn.functional.gelu` (double and same dtype)
with the CANNBench precision checker, including cancellation and small-value
rules. In particular, GeLU(−Inf)=NaN is intentional, matching Torch.
The source-gate suite passed 36 tests and 111 subtests.

## Why an Erf API issue was filed

[pyasc issue #8](https://gitcode.com/compiler-team/pyasc/issues/8) has an English
main text and a Russian comment. Public
[ops-nn GeLU DAG](https://gitcode.com/cann/ops-nn/blob/f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307/activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h)
uses Vec::Erf for FP32, versus default ErfFast after FP32 promotion for FP16/BF16.
The corresponding installed CANN9.0 Vec::Erf selects
SUBSECTION_POLYNOMIAL_APPROXIMATION. The
[official AscendC documentation](https://asc.gitcode.com/api/SIMD-API/adv_api/math_compute/Erf_interface/Erf.html)
distinguishes that high-accuracy mode from default performance-oriented Padé.

Pinned public `asctile.erf` exposes no algorithm choice. This affects available
accuracy/cost/temporary-UB tradeoffs; allocation flags cannot substitute for it.
The subsection algorithm has **not been benchmarked in this experiment**.
Neither a complete GeLU accuracy fix nor a speed advantage is claimed for it.
Even accurately rounded FP32 Erf may lose tiny tails near −1.
See the [issue evidence](ERF_ISSUE.md) and [review adjudication](REVIEW_ADJUDICATION.md).

## Evidence and limitations

This publication archives the report, candidate and evaluation evidence. It
does not deploy unrelated pending skill/worker changes from the dirty local
workspace. Harness scripts assume the already provisioned pinned runtime and
local integration dependencies; they are not a standalone installer. The
preserved RUN_PLAN.md is the pre-execution plan, not the current status;
this README and hardware-summary.json are authoritative for the completed run.

The primary OpenCode reviewer was `dashscope/qwen3.7-max`, session
`ses_f8329847bffee0yi5gbyJuEoSu`; complete skill/source evidence was supplied
inline with tools denied. [Provenance](workers-submission-v2/provenance.json)
records hashes. The main agent rejected incorrect initial review claims,
including changing the mandated −Inf result and unsubstantiated SciPy execution.
Independent review is not a replacement for tests.

[Earlier target-only local report](LOCAL_TANH_REPORT.md),
[direct-Erf failure](EXACT_CHECK.md), and [exact remedy investigation](EXACT_REMEDY.md)
are historical, scope-specific evidence. Full hardware acceptance and the ≥1×
performance target must be judged by the new CANNBench result, not Model ticks.
