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
or fusion options. Follow [JIT diagnostics](../../pyasc-api-patterns/references/api-jit-options.md)
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
