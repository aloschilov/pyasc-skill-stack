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

Replay the full official case set with the target's effective AIV capacity,
including device/thread limits, rather than only the simulator's count or a
small-input resource smoke. Record capacity, selected tile/unroll/grid and
kernel binding per case. For AOT packaging, check that every target-selected
specialization is present: a change from 72 to 64 AIVs can select a different
tile or body. An unqualified missing-binding fallback must block promotion of
that route. Even a numerically qualified fallback is separate coverage; its
timing does not measure the intended generated specialization. If the service
does not expose its query values, label injected counts as assumptions and
compare actual reported kernel names before attributing a result.
Different tiles can share a kernel symbol. A matching name does not recover
the specialization key or launch grid; leave that binding unresolved without
the corresponding launch metadata.

Before a new source-AOT submission, qualify the complete submitted ZIP's build
in a fresh, bounded environment, including every specialization and wheel
packaging. Record whole-script wall time, the attributed budget, compiler and
resource identities, and compare every output wheel member with the qualified
inventory. Incremental rebuilds with inherited binaries remain provenance,
not a full cold-build pass; summed per-object times are not elapsed wall time.
Follow the [prepared-package receipt contract](../../../integrations/cannbench/README.md#evaluate-a-prepared-level-1-package).
A locally successful build does not guarantee the same timing on an unknown
service runner; a timeout in one job is not a universal service deadline.

Compare the full launch configuration, not just specialization keys or ELF
hashes. The same binary with a different grid is a distinct numerical check:
block-strided loops and tail ownership can depend on the launch block count.
Include grid-only changes when selecting numerical cases, or explicitly retain
them as compile/dispatch-only coverage until checked at the target grid.

Inspect the emitted integer widths for tile counts, offsets and DMA byte
strides, including fallback routes. An element count that fits signed 32-bit
does not make `n + tile - 1` or `(n - valid) * itemsize` safe. Widen the
arithmetic where the runtime supports it, or guard the supported input range
before allocation and launch. Check boundary metadata without allocating
huge tensors, and retain every official task case within that range. These
host checks supplement numerical execution; they do not replace it.
For a full operator, a guard that rejects another legal input is a documented
coverage limit, not a repair of its address arithmetic. Check the declared
operator domain as well as the public examples before claiming full support.

On a pin that lowers dimensions or pointer-add offsets to int32, a 64-bit
typed C++ pointer does not make earlier scalar expressions safe. For a stream,
widen the ceiling-division intermediate before adding the tile remainder;
consider a bounded tile-relative GM view so the DMA stride uses that tile's
valid extent instead of the whole tensor. Inspect emitted arithmetic and the
launch ABI after the change. This does not establish support beyond the
public API's representable element-offset range.

This gate does not execute numerical code or prove device launch, correctness
or speed. Data-dependent host reads are unsupported by its tensor stubs. Do
not add candidate branches that detect the validator or catch its missing API
to bypass a real dispatch path. Report that route as blocked and use a
validator that can exercise it before claiming complete coverage.

## Coverage beyond public rows

For a full operator submission, replay host dispatch on a small, independently
chosen set of legal shapes outside the public case list. Derive these probes
from the operator contract and implementation boundaries: byte alignment,
tile/slab thresholds, one and many rows, final tails, supported dtypes, split
axes and strides. Keep this set separate from official qualification; do not
reconstruct hidden inputs or add branches keyed to reported case IDs.

Check the actual packaged callable and its native lookup, not only the JIT
source. Record the selected kernel, constexpr key, launch grid, origin and
whether that binding exists. Exact-width specializations may remain useful,
but an AOT package with no JIT compiler also needs a qualified general route
for supported widths absent from its table. Prefer a bounded family of physical
tiles with runtime logical extents where the pinned compiler supports them.
For a retained generic fallback, measure and identify its kernel separately:
correct outputs alone do not show that the intended vectorized path ran.

Run numerical and suppressed-launch checks on representative new routes and
their boundary tails. A dispatch replay proves selection and binding coverage,
not numerics or speed; a synthetic smoke does not prove the hidden dataset.

## Numerical checks

Use the recipe's ordinary and adversarial numerical probe with the configured
simulator. Preserve the official golden's dtype, tolerance, special-value,
shape and list-output semantics. Retain per-case results and candidate hashes.
Mark infeasible cases compile-only, never numerically passed.

If a task export turns nonfinite range endpoints into `null`, recover them
from the revision-bound canonical case specification before sampling. `null`
alone cannot distinguish NaN from positive or negative infinity. Without that
source, mark the input domain unresolved rather than choosing finite defaults.

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

`--candidate-root` must contain `__init__.py` and export the selected Python
wrappers from within that package. Relative imports are supported. The helper
loads this directory in an isolated namespace and records `source_binding`
with the requested and imported source paths and SHA-256 hashes; it rejects
missing packages and exports from another package instead of falling back to
an installed `cann_bench`. A temporary bytecode cache prevents stale `.pyc`
files from substituting for current sources, without modifying existing source
caches. The helper compares the package's Python source hashes before import
and after the run; a mismatch fails the smoke.

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

When optimizing the CANNBench score, compare complete candidates on the same
catalog, hardware and timing protocol. Keep the source hash for each series.
Apply the catalog's scorer to each measured time with that case's
`baseline_perf_us` and `t_hw_us`. With the tasks 2.x weights and Eq.3, a fully
correct 20-case operator earns `30 + 70 * mean(case_scores)`, subject to the
operator cap. To choose the next tile experiment, replay the expected time
change on the cases its dispatch branch handles and inspect the change in
points. Use the published scorer for its behavior near the hardware floor.
[Per-case formula](https://github.com/hicann/cann-bench/blob/53ea611a1b88da922b82812f2721dd0b2fa33476/src/kernel_eval/base/perf_metric.py) ·
[Operator score and weights](https://github.com/hicann/cann-bench/blob/53ea611a1b88da922b82812f2721dd0b2fa33476/src/kernel_eval/benches/cann_scoring.py)

Keep geometric-mean speedup alongside the score. On two complete 950PR
SwiGLU runs, W72 led W75 on geometric mean (1.09109× versus 1.09043×).
W75 led on projected points (69.9746 versus 69.9189). Treat such a small
margin as a reason for paired repeats. Label a local replay as a projection
until the service returns its score for the submitted ZIP.
[W72 timings](https://github.com/aloschilov/pyasc-skill-stack/pull/60#issuecomment-6882053364) ·
[W75 timings](https://github.com/aloschilov/pyasc-skill-stack/pull/60#issuecomment-6888454179)

Distinguish the outer phase's wall-time limit from a reported device/kernel
timeout. A phase that stops during or between expensive checks leaves the
remaining checks incomplete; it does not establish a numerical failure or vector-core
timeout. Preserve completed results and the exhausted budget. Any authorized
continuation gets its own bounded record. With source, runtime, checker and
fixtures unchanged, run only the missing checks; otherwise revalidate affected
completed coverage.

CANNBench submission is explicit and credit-consuming. Only its complete
selected official case set, successful numerics, zero anti-cheat failures and
finite positive device timings establish the hardware result. Preserve the
submitted package identity and report; do not infer qualification from a
queued job, a different runtime, or local compilation. Full evidence stays
outside checkout; accepted qualification gets a durable archive and a concise
reference in repository documentation.

Inspect per-case performance errors even when the terminal job says
`succeeded` and all accuracy checks passed. Report missing times and distinguish
the first device error from later checks skipped after an unrecoverable error.
Keep standard and hidden case sets separate, including their measured counts;
their combined leaderboard score does not establish improvement on either set.

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

For a local-to-service gap, distinguish matching source, matching retained
instruction bodies and the binary actually bound to each timing record.
A matching `.text` dump from another compilation does not establish which
binary was timed. Record whether a digest covers the whole ELF, raw section
bytes or formatted tool output. Check the selected dispatch, grid, launch ABI
and registration/synchronization lifecycle as well as warmup, input restoration
and per-invocation aggregation. Keep host wall time separate from device-kernel
duration; a different host lifecycle alone does not explain a device-time gap.
Retain the terminal service result when raw repeats or binary bindings are
missing. Distinguish a demonstrated resource-dependent dispatch/binding change
from an unmeasured explanation of its timing impact.

The Exp + SwiGLU control `job_dd91a091c27a` passed 40/40 cases with score
150.7287. Its final 40-case geometric mean was 1.37914×. The summary field
`geometric_mean_speedup` was 1.52967×, reproduced by taking the geometric mean
of the two operators' arithmetic mean speedups. This control used retained
Exp with a manual cache change and unchanged W75 SwiGLU. Attribute the result
to those submitted bytes. A skill-improvement claim requires a separate
generation comparison. [Terminal control report](https://cannbench.com/workspace/jobs/job_dd91a091c27a)
and [retained evidence](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-l1-performance-44141381187).

The later [late-gather submission](https://cannbench.com/workspace/jobs/job_5c2ab4c8015d)
also passed 40/40, but scored 124.0239 with unchanged baseline/floor anchors.
Its ten reported `swiglu_fallback` routes match the missing bindings in the
64-AIV dispatch replay. SwiGLU 9 ran that fallback in 1368.64 µs, versus
7.86 µs for the control's retained kernel: this does not evaluate the proposed
late-gather body. The available service artifacts do not expose resource-query values, and
Exp's shared `_exp_kernel` name cannot independently identify its fallback.
Numerical acceptance must not promote such timings as measurements of the
intended optimization.

The [follow-up run](https://cannbench.com/workspace/jobs/job_7ffaa136fa8f)
retained byte-identical SwiGLU source, host wrappers, resource helper, all 23
packaged C++ specializations and their compiler flags. It passed all 20 SwiGLU
cases without a reported fallback; case 11 used `row_single_wave_fp32` at
4.24 µs. Both jobs requested `950pr`, but their reported execution environments
differed. A target-family label and identical source package do not establish
the same effective AIV count, selected specialization or service-built ELF.
Preserve those unknowns instead of assigning every service run a fixed count
or treating this timing recovery as evidence of a source optimization.

Treat `Failed to get acl to npu flow events` as a correlation diagnostic.
Assess timing validity from the terminal numerics, anti-cheat and performance
integrity results, with finite positive timings for actual candidate kernels.
The control above passed those terminal checks despite the warning. Its
published artifacts omitted raw CSVs and per-invocation durations, so repeat
completeness and median aggregation remained independently unverified.
Preserve that limit whenever the underlying records are unavailable.

Separate numerical correctness from a later input-challenge or profiling failure.
If the service reports `no unseen legal input`, inspect the pinned official
input generator before changing the kernel: a constant domain such as `[0, 0]`
cannot supply a distinct input by changing the seed. Reproduce that property on
CPU, retain the failed job and its error, and report the benchmark incompatibility
before another submission. Do not change the official domain, omit the case or
disable the integrity check to obtain a score. Keep any earlier numerical-stage
pass separate from final acceptance; incomplete performance coverage is not a
qualified result, and its missing timings must not be filled from another run.
