# Local validation ladder

Use the current run's task snapshot and source/runtime manifest. The checkout's
single-operator CI selects its runtime in
`integrations/cannbench/ci/single-runtime.json`; prose and historical runs must
not override that pin. Verify imported runtime bytes against the supplied
wheel before attributing a compiler result to a revision.

Use already provisioned commands within the assigned time/tool budget. Do not
install packages, build an image, or start an unbounded suite unless requested.
Without a configured runtime, perform syntax/static checks and report the
remaining validation as pending.

## Contract and lowering

Check the exact callable signature/defaults and every case's shapes, dtypes,
attributes and list/optional arguments against the selected official task.
Derive the required case count from that snapshot; another operator's count
or a historical full-catalog inventory is not an acceptance criterion.

The CI helper `integrations/cannbench/ci/single_qualify.py` attests the runtime
and calls `workers/local_compile_gate.py`. It replays metadata-only host
dispatch, captures selected JIT launches, and lowers every unique
specialization through codegen, compiler passes, AscendC translation and UB
checks. All selected case routes must pass; zero cases cannot pass.

Inspect the emitted integer widths for tile counts, offsets and DMA byte
strides, including fallback routes. An element count that fits signed 32-bit
does not make `n + tile - 1` or `(n - valid) * itemsize` safe. Widen the
arithmetic where the runtime supports it, or guard the supported input range
before allocation and launch. Check boundary metadata without allocating
huge tensors, and retain every official task case within that range. These
host checks supplement numerical execution; they do not replace it.

This gate does not execute numerical code or prove device launch, correctness
or speed. Data-dependent host reads are unsupported by its tensor stubs. Do
not add candidate branches that detect the validator or catch its missing API
to bypass a real dispatch path. Report that route as blocked and use a
validator that can exercise it before claiming complete coverage.

## Numerical checks

Use the recipe's ordinary and adversarial numerical probe with the configured
simulator. Preserve the official golden's dtype, tolerance, special-value,
shape and list-output semantics. Retain per-case results and candidate hashes.
Mark infeasible cases compile-only, never numerically passed.

Record the precision of the golden's operations as well as its output dtype.
Calling `golden(x.double())` can still compute in FP32 if the golden casts
internally. If the result is then stored in FP64, label it as FP32-computed
and FP64-stored. A separately implemented FP64 reference is a diagnostic with
its own overflow and special-value behavior. Keep it distinct from the
official acceptance oracle.

The optional `integrations/cannbench/workers/run_camodel_smoke.py` contains
reusable small numerical routes. Invoke it only in an already configured
runtime, selecting the relevant operators explicitly:

```bash
python3 integrations/cannbench/workers/run_camodel_smoke.py \
  --candidate-root /absolute/path/bundle/cann_bench \
  --ops gelu,foreach_addcdiv_scalar --suite adversarial \
  --output /absolute/path/outside-checkout/numerics.json
```

Its basic, critical and adversarial suites have different coverage. The
adversarial suite supports only the two operators shown. Inspect simulator
diagnostics as well as comparisons: padded lanes may perform invalid arithmetic
even when `real_shape` prevents their results being stored. Partial smoke
coverage is not complete task qualification, and simulator ticks are not
hardware timings.

## Bounded optimization and hardware acceptance

Establish a correct public `@asctile.jit` baseline before changing allocation
or fusion options. Follow [JIT diagnostics](../../asctile-api-patterns/references/api-jit-options.md)
for fresh processes, isolated caches and requested/effective options. Keep
synchronization enabled. Record source, runtime, geometry, UB use, compilation
and numerical results for each bounded experiment; preserve the last correct
measured incumbent. Do not treat lowering via constructed options as proof
that the decorator/cache path works.

CANNBench submission is explicit and credit-consuming. Only its complete
selected official case set, successful numerics, zero anti-cheat failures and
finite positive device timings establish the hardware result. Preserve the
submitted package identity and report; do not infer qualification from a
queued job, a different runtime, or local compilation. Full evidence stays
outside checkout; accepted qualification gets a durable archive and a concise
reference in repository documentation.

Use terminal per-case timings after the service finishes any automatic
rechecks. Preserve first-pass console times separately because a recheck can
change the accepted timings. Recompute the case geometric mean as
`exp(mean(log(baseline_us / candidate_us)))` over the exact matched cases.
Check the aggregation behind a service summary label. Across runs, compare
case identities, hardware, metric/profile and each reported baseline and
hardware-floor anchor alongside candidate times. A matching catalog version
alone does not establish a matched performance comparison.
Matching anchors permits comparison of the reported ratios. Attributing a
timing change to code also requires paired measurements on the same runner
with matched inputs, runtime and measurement protocol. Preserve timing changes
of unchanged-source controls as observed run variation, without using them as
normalization factors for other kernels.

The Exp + SwiGLU control `job_dd91a091c27a` passed 40/40 cases with score
150.7287. Its final 40-case geometric mean was 1.37914×. The summary field
`geometric_mean_speedup` was 1.52967×, reproduced by taking the geometric mean
of the two operators' arithmetic mean speedups. This control used retained
Exp with a manual cache change and unchanged W75 SwiGLU. Attribute the result
to those submitted bytes. A skill-improvement claim requires a separate
generation comparison. [Terminal control report](https://cannbench.com/workspace/jobs/job_dd91a091c27a)
and [retained evidence](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-l1-performance-44141381187).

Treat `Failed to get acl to npu flow events` as a correlation diagnostic.
Assess timing validity from the terminal numerics, anti-cheat and performance
integrity results, with finite positive timings for actual candidate kernels.
The control above passed those terminal checks despite the warning. Its
published artifacts omitted raw CSVs and per-invocation durations, so repeat
completeness and median aggregation remained independently unverified.
Preserve that limit whenever the underlying records are unavailable.
