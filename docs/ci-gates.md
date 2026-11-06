# CI gates

The [CI workflow](../.github/workflows/ci.yml) separates ordinary checks from
explicit model and hardware work. Separate generation controllers are scheduled
following authorization on 2026-11-06; NPU remains explicit and unscheduled.
The [generation planner and explicit executor](../tests/generation/README.md)
validate a bounded multi-operator inventory. Ordinary CI exercises only local
provider doubles and fixed numerical fixtures. An explicit manual generation
dispatch can invoke the executor after plan approval. Scheduled qualification
is reported separately from the existence of enabled controllers.

| Job | Trigger | Acceptance |
|---|---|---|
| `pr-gate` | PR, push to `main`, manual dispatch | Offline unit, skill syntax, image-pin and reference-drift checks |
| `examples-gate` | Same, after `pr-gate`; 20-minute cap | Pinned simulator verification contract and permanent numerical regression fixtures |
| `merge-gate` | Push to `main`, or manual dispatch with both hardware and generation actions `none`; 60-minute cap | Every discovered target module compiles; failures **and resource blocks** fail the gate |
| `generated-hardware` | Explicit manual dispatch with generate/repair/continue/resume | Selected operator recipe, exact package, strict hardware result and durable publication |
| `generation-simulator` | Explicit manual `generation_action=plan|execute`; default `none` | Offline plan, or hash-approved fresh simulator qualification and optional compatible-baseline comparison; no NPU |
| `Opt-in generation` | Daily 17:17 UTC; PR planning only, or explicit manual plan/execute | Shared executor and pinned policy baseline; PRs never call a provider |
| `Paired generation experiment` / `Model generation comparison` | Daily cadence checks at 17:47 / 18:47 UTC; at most one execution per 168 hours | Complete predeclared cohorts, retained failures, no NPU |
| `Reviewed generation` | PR planning only, or explicit manual plan/execute | [Fresh writer/reviewer sessions](../tests/generation/review.md), bounded repairs and unchanged-byte simulator qualification; no NPU |
| `Generation coverage observation` | Hourly at minute 43 UTC, or explicit manual dispatch | GET-only archive/freshness inspection; scheduled issue updates use marked destinations; manual notification requires opt-in |
| `Evidence dashboard` | Relevant PRs run offline tests; manual build, optional publish | GET-only projection of pinned public archives; [scope, failure handling and publication](../dashboard/README.md); no generation or device access |
| release | `v[0-9]*` tag | Fast suite, skills-only archive and install verification |

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
suite, not the definition of the CI pipeline. Generation adapters additionally
check three fixed add/reduce/broadcast kernels and three wrong-result variants;
each correct result requires a rejecting suppressed-launch control. A seventh
negative fixture requires an unknown public API to fail as a candidate defect,
not be excused as a broken simulator environment.

Both PR jobs must pass on the current head. Artifacts retain the complete
verification-contract and generation-adapter runs for 30 days, including failures. Queue time is separate
from execution time. Neither job qualifies model generation or NPU behavior.
Accepted qualification requires [durable evidence](verification-contract.md#storage).
The [controller and observer](../tests/generation/README.md#inactive-nightly-controller-and-health-observer)
retain control/health records separately. Their planning/manual green checks do
not establish scheduled coverage or successful notification delivery.

## Compile coverage

`merge-gate` recursively discovers `reference/asctile/target/test_*.py`, including
nested modules, and invokes `run_target_test.py` with backend `Model` and platform
`Ascend950PR_9599` in the pinned simulator image. Modules with a declarative
case list use a representative case; other modules use their collected cases.
This is compile-only, not exhaustive specialization or numerical coverage. The summary separates failed kernels from
resource-blocked checks, but either makes coverage incomplete and the job nonzero.
No discovered modules also fails. See [manual procedure](review-order.md).

The separate [target numerical census](../tests/target-suite/README.md) derives
all 1,155 official case IDs and runs explicit bounded simulator shards with
unchanged assertions and suppressed-launch controls. It is a manual CLI lane;
ordinary PRs run its offline contract tests only. Selected-subset success cannot
make the full-suite completeness gate green: excluded, unrun, unsupported,
failed and resource-blocked cases remain visible in the denominator.

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

`asctile-codegen-workflow`, `asctile-code-review` and `asctile-build-run-verify` retain
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
