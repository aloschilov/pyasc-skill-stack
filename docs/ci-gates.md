# CI gates

The [CI workflow](../.github/workflows/ci.yml) separates ordinary checks from
explicit model and hardware work. No nightly schedule is configured.

| Job | Trigger | Acceptance |
|---|---|---|
| `pr-gate` | PR, push to `main`, manual dispatch | Offline unit, skill syntax, image-pin and reference-drift checks |
| `examples-gate` | Same, after `pr-gate`; 20-minute cap | Pinned simulator verification contract and permanent numerical regression fixtures |
| `merge-gate` | Push to `main`, or manual dispatch with hardware action `none`; 60-minute cap | Every discovered target module compiles; failures **and resource blocks** fail the gate |
| `generated-hardware` | Explicit manual dispatch with generate/repair/continue/resume | Selected operator recipe, exact package, strict hardware result and durable publication |
| release | `v*` tag | Fast suite, skills-only archive and install verification |

## Ordinary PR checks

```bash
bash tests/ci-gate.sh --tier pr
python3 tests/tools/verification_contract.py --results-dir "$RUN_ROOT/contract"
```

The first command is offline. The second needs Docker and the pinned image in
`reference/images/ENV.json`; choose a new directory outside checkout. The matrix
covers historical kernels, tails, negative and weak-positive fixtures, and skill
listings. Expected failures must fail for the contracted reason; numerical
passes require a suppressed-launch control. The separate SwiGlu short-row
regression protects a known memory-layout bug; it is one fixture in this broader
suite, not the definition of the CI pipeline.

Both PR jobs must pass on the current head. Artifacts retain the complete
verification-contract run for 30 days, including failures. Queue time is separate
from execution time. Neither job qualifies model generation or NPU behavior.
Accepted qualification requires [durable evidence](verification-contract.md#storage).

## Compile coverage

`merge-gate` recursively discovers `reference/asctile/target/test_*.py`, including
nested modules, and invokes `run_target_test.py` with backend `Model` and platform
`Ascend950PR_9599` in the pinned simulator image. Modules with a declarative
case list use a representative case; other modules use their collected cases.
This is compile-only, not exhaustive specialization or numerical coverage. The summary separates failed kernels from
resource-blocked checks, but either makes coverage incomplete and the job nonzero.
No discovered modules also fails. See [manual procedure](review-order.md).

## Explicit model and hardware evaluation

The [shared workflow](../.github/workflows/cannbench-generated.yml) accepts an
operator recipe, currently `swi_glu`. Recipes own operator-specific prompts and
numerical probes; the driver owns attribution, packaging, submission, collection
and publication. Unsupported recipe names fail before model invocation or POST.
Only SwiGlu has recorded hardware qualification. Adding a recipe does not qualify
it automatically. See [requirements and recovery](../integrations/cannbench/README.md#one-fresh-candidate-through-ci).

Ordinary PR checks never generate candidates or consume CANNBench credits.
`resume` collects an existing submission and publishes its evidence without
regeneration or resubmission.

## Agent workflows

`pyasc-codegen-workflow`, `pyasc-code-review` and `pyasc-build-run-verify` retain
the design → implementation → fresh-session review → validation → tuning loop.
A caller arranges these handoffs; one model can serve different sessions.
`tests/run-tests.sh --agentic` exercises provider-backed behavior and integration;
`--all` also runs the offline units. Both can incur provider charges.

The bounded CANNBench driver currently automates generation/repair and machine
checks, not a separate reviewer or tuning agent. That automation gap is tracked
with acceptance criteria in [issue #20](https://github.com/aloschilov/pyasc-skill-stack/issues/20).
The installer option `--agents` installs into `~/.agents/skills`; it does not
create an orchestration team. The skill workflow does not depend on the removed
legacy `teams/` prompts.

## Release archives

[Release CI](../.github/workflows/release.yml) builds and install-checks archives
containing `skills/`, `install.sh`, `README.md` and `CHANGELOG.md`. Runtime pins,
references and verification tools require the full checkout. A skills-only
installation is not end-to-end qualification.
