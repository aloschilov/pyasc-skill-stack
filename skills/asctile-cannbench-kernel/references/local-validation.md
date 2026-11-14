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

This gate does not execute numerical code or prove device launch, correctness
or speed. Data-dependent host reads are unsupported by its tensor stubs. Do
not add candidate branches that detect the validator or catch its missing API
to bypass a real dispatch path. Report that route as blocked and use a
validator that can exercise it before claiming complete coverage.

## Compare the generated entry on resident tensors

For a short NPU kernel, record the input and output devices together with
the Torch, torch_npu and AscTile wheel versions. Use a compatible Python ABI
to call the unedited generated operator on ordinary NPU tensors. Check its
output against the task golden. A matched CPU-input call through the same
wheel can expose the effect of tensor placement on device-task time. Keep
both calls on the same profiler and repeat them in reverse order. For a continuous diagnostic trace with L2 conditioning, verify one
cache-clean task before each measured invocation and count every target task. Record a source hash
and the observed launch sequence for each run. Separate this comparison
from the full-suite GM, whose rows share the same input and timing protocol.

### Match the benchmark revision and timing protocol

Read the task version from `tasks/suite_manifest.json` and check the evaluator
defaults at the same Git revision. On tasks 2.1.0 at `53ea611`, the default
`restored_io_v2` path in `OpRunner.run_ai_op` prepares fresh input storage,
clears L2 before each capture, and retires the previous invocation's I/O.
The evaluator uses one preflight, three warmups and five separate measured
sessions. It sums all candidate kernel durations within each session and
rounds the median of the five sums to two decimal places. Use that returned
time when dividing the device JSON baseline by the candidate time.

For a local unary qualification, run
`integrations/cannbench/workers/profile_restored_l1.py PACKAGE TASK CASE OUTPUT`
in the matched CANNBench, PyAsc and torch_npu environment. The worker reads
release case YAML and uses the evaluator's case-ID-derived seed and input
distribution. Record this fixture identity alongside the measurement protocol
when comparing with a previous run. The provider's
release `build.sh` detects the device using `acl.get_soc_name()`. Check the
reported SoC. If building with CMake directly, pass the corresponding
`-DNPU_ARCH` value. The 950PR target is `ascend950` (Bisheng `dav-3510`).
Direct CMake and a failed ACL probe both default to `ascend910b` in this revision.
Verify the warmup, cache-clean, clone and zero operators with the matched
provider. Keep the five captures with their validated outputs and the
evaluator, source, case and golden hashes. Compare a full set measured under
one protocol when reporting a GM.

## Numerical checks

Use the recipe's ordinary and adversarial numerical probe with the configured
simulator. Preserve the official golden's dtype, tolerance, special-value,
shape and list-output semantics. Retain per-case results and candidate hashes.
Mark infeasible cases compile-only, never numerically passed.

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
