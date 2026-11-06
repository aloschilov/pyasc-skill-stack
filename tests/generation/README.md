# Bounded generation experiments

This is the executor slice of [nightly coverage #22](https://github.com/aloschilov/pyasc-skill-stack/issues/22).
The default commands are offline and do not invoke OpenCode, Docker or a provider:

```bash
python3 tests/tools/generation_plan.py --results-dir /outside-checkout/plan
python3 tests/tools/generation_executor.py --plan /outside-checkout/plan/plan.json
```

Planning verifies pinned task/runtime agreement and source drift, and binds case
ASTs, skills, profile, image digest and the executor/verifier source closure.
Validation rejects modified plans, even if someone recomputes the plan checksum.
A plan is `not-run`, not spending authorization or numerical evidence.

## Numerical contract

The inventory has three explicit upstream task anchors. It does not run every
pytest parametrization or reproduce the upstream random-input distribution.

| Task | Shape and dtype | Oracle | Probe acceptance |
|---|---|---|---|
| `vadd_test_2` | 199 elements, float32 | x + y | rtol 1e-5, atol 1e-6; tail |
| `reduce_sum_test_1` | 61 elements to [1], int32 | sum(axis=0, keepdim=True) | exact; padded input |
| `broadcast_test_2` | [1,5,8] to [4,5,8], float32 | broadcast | exact; four cores |

Adapters check shape/dtype/block/operation attributes against the pinned rows
without importing them. Each uses three deterministic input patterns, poisoned
output, and checks both results and unchanged inputs. Candidate code provides a
`launch` function with one JIT kernel call; host arithmetic, Torch fallbacks,
runtime reconfiguration and arbitrary imports are rejected. This intentionally
restricted candidate interface is described in each generated prompt.

The same candidate bytes pass static checks, JIT lowering with suppressed
launches, real **Model simulator** execution, and a separate suppressed-launch
negative control. Comparisons belong to trusted probes, not candidate self-tests.
Missing launches, missing comparisons, wrong results, inconclusive controls,
changed bytes or unavailable runtime cannot pass. These guards do not constitute
a sandbox against deliberately hostile Python; execution uses a networkless
container without credentials or device mounts.

## Explicit execution, budgets and evidence

An authorized manual run uses `--execute --approve-plan <exact plan_sha256>
--results-dir /outside-checkout/new-run`. It needs OpenCode, the locally available
digest-pinned simulator, and `OPENCODE_BASE_URL` / `OPENCODE_API_KEY`. The executor
never pulls an image, submits to CANNBench, or selects an NPU backend. No schedule
is added. Manual CI dispatch is described below. A provider-backed simulator pilot is recorded below; ordinary PR checks
continue to use fixed fixtures and local provider doubles only.

The executor runs serially, reserves the full worst-case output and execution
budget **before** each attempt, and never recycles unused reservations to enlarge
the matrix. Current maxima are 3 attempts total, 8 provider requests per attempt,
8192 completion tokens per request, 300 s generation + 300 s qualification per
attempt: 196,608 output tokens and 1800 execution seconds. Input tokens, provider
prices, setup, cleanup and archival overhead are outside these ceilings. The initial 4096-token response limit was exhausted by a GLM-5.3
reasoning-only response during manual qualification; it was raised under the
user's expanded model-use authorization, keeping the attempt/time caps. Cleanup
has a separate 20 s timeout. OpenCode/SDK/protocol compatibility is checked during
execution; an unavailable client/provider becomes `blocked`, not a mock success.

OpenCode receives selected skills in a fresh Git worktree/private state directory
(no commits are created). The worktree makes its file-edit permission patterns
resolve to the intended project. It runs with shell, delegation, web access and unrelated edits denied. Native session and
skill-load events plus provider-observed model/usage bind attribution.
Matrix schema 2 separately pins the requested alias (`gateway/glm-5.3`) and
expected provider response ID (`glm-5-3-260814`). That reported ID is retained as
model revision metadata, not a claim of immutable model weights. Any unlisted
response ID is rejected; no automatic model fallback is accepted.
A local Chat Completions gate replaces the provider URL/key for this
process and reserves each request before forwarding it. It sets `max_tokens`,
requires complete usage (including reasoning), and rejects additional requests
at the round cap. OpenCode's step setting alone is insufficient because it may
request a final summary. The gate currently supports only
`@ai-sdk/openai-compatible` providers with complete Chat Completions usage and a
matching pinned response-model ID. It buffers each bounded response to verify usage before
allowing the next request. It cannot cancel remote computation or guarantee a
provider's billing behavior; missing/over-limit usage is retained as unknown and
stops subsequent work. No blind request retries or automatic recovery are allowed.

The process-group watchdog kills local children on deadline or cancellation;
Docker cleanup is separately confirmed. All attempts, failed candidates, usage
reservations, commands, hashes, timings and partial cell states stay outside Git.
An unfinished attempt leaves subsequent cells `not-run` and retains a durable
`active.json` latch under `~/.local/state/pyasc-generation`, in addition to an OS
lock preventing concurrent executors. Do not bypass it with another state directory.
Inspect the referenced run's provider usage, commands and container cleanup before
manually reconciling the latch. A new execution requires a new directory and explicit
approval; there is no automatic `resume` or replay of an interrupted attempt.

Keep `private/` (OpenCode's local application state and raw provider responses)
private and out of published archives. Rejected responses retain their hash,
HTTP status/request ID and a fixed validation reason in `usage.json`; raw bytes
are private reconciliation evidence, never a reason to replay a request. Complete
`finish_reason=length` responses retain their measured usage and fail the cell
as budget exhaustion; they are not mislabeled as unknown delivery.
Permanent rejection returns a non-retryable client error and stops the process
group immediately, instead of letting the SDK spend the remaining deadline
retrying a local rejection. `files.json` deliberately excludes it. Provider credentials are not
passed to OpenCode or written into model configuration, request records or logs.
Execution is not isolated from a hostile local user; permissions and the budget
gate protect the ordinary agent workflow, not adversarial credential theft.

## Model-free checks

```bash
python3 -m unittest discover -s tests/unit/python -p 'test_generation*.py'
python3 tests/integration/check-generation-adapters.py \
  --results-dir /outside-checkout/adapter-fixtures
```

The first command covers planning, native attribution, process deadlines/child
cleanup, serial reservations, partial outcomes, locking and usage enforcement
against a loopback fake provider. It needs neither credentials nor Docker.
The second uses the pinned simulator with three correct permanent fixtures and
three deliberately incorrect variants, plus one nonexistent-API negative fixture. Each correct fixture must reject suppressed
launches; each incorrect variant must fail an actual post-launch comparison. The unknown
API must be classified as a candidate failure, not an unavailable simulator.
`examples-gate` retains these reports in a separate artifact. This is verification
of the adapters/executor, **not** newly generated-kernel or hardware qualification.

Remaining #22 work: live alert delivery and an explicitly approved
schedule plus end-to-end scheduled acceptance. The inactive controller and health
observer are described below. #23 skills-effect comparisons and #24 model diversity should
reuse these contracts; #20 owns separate reviewer/tuner handoffs.

## Manual CI dispatch

The existing `CI` workflow routes explicit `generation_action=plan|execute` to
`generation-simulator.yml` after the normal PR checks. Its default is `none`;
pushes and pull requests never invoke a model. Hardware and simulator generation
cannot be selected in the same dispatch. There is no schedule trigger.

```bash
gh workflow run ci.yml --ref <reviewed-branch> -f generation_action=plan
# Inspect plan/plan.json and SUMMARY.md in generation-evidence-<run>-<attempt>.
gh workflow run ci.yml --ref <same-branch> -f generation_action=execute \
  -f generation_plan_sha256=<exact-reviewed-plan-sha256>
```

The runner must provide Docker and the pinned simulator image; the normal
examples gate pulls that image. Execute mode installs OpenCode 1.19.0 for Linux
arm64 from its official release asset with a pinned SHA256. The execution step alone receives
`OPENCODE_API_KEY` and the repository's `OPENCODE_BASE_URL` variable. Its child
OpenCode process still receives only the local gate credential. No CANNBench
credential or NPU backend is passed. Plan-only runs need no provider credential.
The approval digest binds the commit and complete plan: advancing the branch
requires a new plan. Merely dispatching `execute` does not approve a changed plan.

The first successful execution is **qualified**, not automatically accepted as
a baseline or a passing regression comparison. Review its full evidence, retain
the selected archive durably, then pass both baseline inputs on a later execute:

```bash
-f generation_baseline=<RUN_ID/ATTEMPT-or-release/TAG> \
-f generation_baseline_sha256=<SHA256-of-generation.tar.gz>
```

Run references download that exact named Actions artifact; `release/TAG`
downloads `generation.tar.gz` from that exact Release in the same repository.
No `latest` lookup or automatic baseline promotion is used. Archive checksum,
safe extraction, origin and baseline qualification are checked; incompatible
plans/clients stop **before** provider dispatch. The comparison's baseline index
digest is derived from this verified archive. A missing/expired artifact stops
the job; do not silently substitute another run. Actions evidence lasts 90 days,
so an accepted baseline must also live in a durable archive (for example a
reviewed Release asset) with its checksum and locator recorded outside Git.

CI retains raw evidence under the runner's persistent shared work volume:
`$(dirname "$RUNNER_WORKSPACE")/.pyasc-generation/results/<run>-<attempt>/`,
with its stable executor latch in the adjacent `state/` directory. This volume
must be mounted at the same absolute path in the runner and its Docker daemon;
the runner container's `HOME` must not hold the only copy of evidence or the latch.
Local jobs sharing this runner must use that same state directory. CI refuses an
existing default-HOME latch instead of silently switching away from it. Only a
public `generation.tar.gz`, checksum and summary are
uploaded. The archive excludes every `private/` tree and symlinks; unpacking
rejects links, traversal, duplicate files, missing hashes and oversized payloads.
Failed/partial evidence is collected with `always()` when the runner is still
available. Hard termination or runner loss may prevent upload; inspect the
persistent directory and shared executor latch before taking further action.
Re-running a CI job creates a new directory but cannot bypass that latch.

Comparison alerts appear in the job summary and cause a failing check; a
qualification-only result cannot claim regression coverage. Scheduling is still
inactive; the separate health observer does not infer fresh coverage from a
green ordinary CI job.

## Inactive nightly controller and health observer

`generation-nightly.yml` reuses the same executor. PRs only prepare a plan without
provider credentials; explicit manual execute requires its reviewed digest.
`tests/generation/nightly.json` keeps `enabled: false`, no activation timestamp
and no alert destination. Neither new workflow has a cron trigger. The policy's
exact Release baseline and checksum are checked before any generation. The policy
snapshot/hash and controller hash are retained separately from execution inputs;
changing the enable flag does not silently requalify an incompatible executor.

```bash
gh workflow run generation-nightly.yml --ref <reviewed-branch> -f mode=plan
# Inspect generation-control-<run>-<attempt>, then use its exact digest:
gh workflow run generation-nightly.yml --ref <same-branch> \
  -f mode=execute -f approve_plan=<reviewed-plan-sha256>

# GET-only audit of an existing manual run; no provider or Docker calls:
python3 scripts/generation_operations.py monitor \
  --root /outside-checkout/new-observation --inspect-run <RUN_ID/ATTEMPT>
```

The independent `generation-health.yml` runs on GitHub-hosted Ubuntu, not the
self-hosted simulator runner. It is manual-only and notification defaults to
false. Disabled policy yields `disabled`; a manual archive audit can yield only
`verified-manual`, never scheduled acceptance. Monitor exit 1 means an alert;
the workflow retains/delivers it before its final failing check.

When explicitly enabled, scheduled observations filter the dedicated workflow's
`schedule` events on `main` after `enabled_since`. Healthy coverage requires the
exact run/attempt and revision, selected baseline, full numerical proof and a
fresh independent comparison. A recent GitHub success without verified evidence
is an alert. Failed/cancelled runs, more than two hours queued/running, missing
coverage after the 36-hour activation grace period, stale results, expired or
corrupt artifacts and observation failures remain visible. History/inspection
limits fail closed. An in-grace period with no completed result is `pending`,
not fresh coverage. A later verified run can clear older failures.

Alerts use one **dedicated**, preconfigured issue whose body begins with
`<!-- pyasc-generation-health:v1 -->`; do not point this at a development issue.
The notifier never creates issues or mentions people. Explicit notification
updates that issue idempotently, reopens it for an alert, and closes it only on
verified healthy coverage. Unchanged findings cause no edit. A PATCH is followed
by GET confirmation; unknown delivery is reconciled, never blindly repeated.
`delivery.json` is evidence that GitHub stored the update, not proof a person read
a notification. No model or CANNBench credential is available to the observer.

Activation remains a separate user-approved change: qualify/select a baseline
compatible with the **final** executor, configure/test the marked alert issue,
set `enabled_since` and `enabled: true`, and add reviewed cron triggers (proposed
generation `17 3 * * *`, independent health `43 * * * *`, UTC). Update the inactive
workflow regression assertions in that same change. GitHub schedules run on the
[default branch and may be delayed](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule),
so activation also requires an approved merge. An actual scheduled generation,
comparison and alert/recovery delivery must be verified before closing #22.
Observation on the same GitHub platform cannot guarantee delivery during a
GitHub-wide outage; job summaries/artifacts remain the audit trail when available.

## Offline regression policy

`generation-no-lost-pass-v1` compares two retained runs without model, Docker or
network calls. Select a reviewed, complete PASS baseline explicitly by the SHA256
of its `files.json`; do not automatically promote the newest run. Store its archive,
archive checksum, index checksum and locator durably outside Git. A failed candidate
never replaces the baseline. Hashes establish integrity, not independent authenticity.

```bash
python3 tests/tools/generation_compare.py \
  --baseline /outside-checkout/accepted-run \
  --baseline-sha256 <reviewed-files.json-sha256> \
  --candidate /outside-checkout/new-run \
  --report /outside-checkout/comparison.json
```

Both bundles must have indexed, matching public files and source snapshots. PASS
cells additionally need unchanged candidate bytes, command/cleanup receipts,
native session and model/skill attribution, known provider usage, numerical checks
and a rejecting suppressed-launch control. Missing evidence fails closed. Private
wire responses are unnecessary and must remain excluded. Reusing the baseline or
one of its successful sessions is not a fresh comparison.

Compatibility requires the same ordered inventory/case anchors, model/profile,
provider response revision, OpenCode version, runtime digest, budgets, required
gates and non-skill input hashes. Only skill contents and checkout revision may
change; changed skill files are listed. The offline comparison tool itself is
excluded from execution compatibility. All other executor, adapter, verifier,
profile or task changes require a new baseline, even when apparently harmless.
An incompatible run is reported explicitly and receives no comparative score.

The denominator is always every planned cell in the baseline. One prior PASS
becoming `failed` emits `lost-pass` and exit **1**; there is no percentage threshold.
`blocked`/`not-run` cells remain coverage gaps, with exit **2**, never disappearing
from the denominator. A regression with additional gaps lists both. Complete
compatible runs without alerts exit **0**. This is a deterministic regression
signal from one trial per configured attempt, not a statistical skills-effect or
performance claim; repeated/paired measurement belongs to #23.

New executor receipts include UTC `started_at` and `finished_at`. The reporter
emits `stale-coverage` when completion is older than 36 hours by default
(`--max-age-hours` is explicit in the report); missing/future/naive timestamps
emit `unknown-coverage-time`. File modification time is never evidence of fresh
coverage. Legacy runs can be inspected or selected explicitly, but cannot prove
fresh coverage without recorded completion time. This tool produces alert records
and nonzero exits; it does not itself monitor a schedule, send notifications or
prove that a scheduled job ran. The health observer implements the independent
checks and explicit issue-update path, but live delivery and scheduled acceptance
remain unqualified while scheduling is disabled.

## Manual pilot, 2026-11-06

Commit `651233ebd5db074ef25eb810a07891fcce4941a9`, OpenCode 1.19.0,
requested `gateway/glm-5.3`, observed `glm-5-3-260814`: **3/3 fresh candidates**
passed static/JIT/Model numerical checks and rejecting suppressed-launch controls.
Candidate bytes were not edited or replaced between generation and qualification.
The three cells used 15 provider requests and 12,851 completion tokens (including
reasoning); summed cell execution was 312.58 s. Each task used the three input
patterns above. This does not establish general generation quality, performance,
NPU coverage, a baseline regression policy or scheduled/nightly acceptance.

The initial 4096-token pilot exposed incomplete rejection diagnostics, a provider
alias/revision mismatch, an OpenCode worktree permission mismatch, rejection of
bare JIT decorators, an incorrect reduction API call, and reasoning-only token
exhaustion. Failed attempts and their original candidates remain separate. One
original request has unknown usage because its response was not retained by the
old gate; later responses were reconciled from private wire evidence. The final
pilot used the explicitly increased 8192-token response limit, not the old cap.

Maintainer archive `manual-simulator-651233e.tar.gz` is retained outside checkout
on both the development host and Linux VM. SHA256:
`f030e4981d4a60117b3c96ccd292a65002ffde25341ea08c5e4ca1b175cce410`.
It contains 210 hash-verified members including the plan, source snapshot,
candidates, sessions/skill attribution, usage, commands and numerical reports.
Private application state/raw provider responses are excluded. The separate
`development-attempts-20261106.tar.gz` preserves failed attempts and reconciliation;
SHA256 `2854eb3b1068fd3d8878177849a6b7ee9882fe22a104a79f2e8d514259da1ffa`.
Archive locations and CI verification belong in the PR handoff; no run artifacts
are committed to Git.

## Manual CI qualification, 2026-11-06

[Dispatch 42126948369](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/42126948369)
on `53ef52e6d592d06400e1351effa44c8b9776b20d` qualified **3/3 fresh candidates**
with OpenCode 1.19.0 / `gateway/glm-5.3` (reported `glm-5-3-260814`).
Add, reduce-sum and broadcast passed unchanged-byte static/JIT/Model numerics:
three patterns each, nine real simulator launches, 21 output/input comparisons,
and three rejecting suppressed-launch controls. All containers were removed.
Executor wall time was **278.08 s**; the complete qualification job took **336 s**
including provisioning/collection. Usage was **15 requests, 11,323 completion
tokens and 155,935 input tokens**; provider cost was unavailable. Limits were unchanged.
This is `qualified`, without baseline comparison or promotion, NPU/performance
measurement, or scheduled/nightly acceptance.

The public archive has 216 hash-verified members and 87 source hashes checked
against that commit. Maintainer copies outside checkout on the development host
and Linux VM: `~/pyasc-qualification-runs/issue22-ci-20261106/generation-ci-42126948369.tar.gz`.
SHA256: `bb574ef974083bc1352c9c5d347a67047203b5a7aea6998ecd4730125e8f3d67`.
Private response/application state is excluded. The earlier pilot and failed
attempts remain intact; this run does not replace their records.

## Selected baseline and compatible comparison

The reviewed `42126948369/1` run was explicitly selected and published in a
[durable qualification Release](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-generation-42126948369)
before the independent comparison. Its `generation.tar.gz` retains the checksum
above; it is never a moving `latest` reference or an automatically promoted run.

[Comparison dispatch 42129712051](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/42129712051)
on `6a2e82756247f6865b5f1930680ddcbd8895cc33` fetched that Release and passed
`generation-no-lost-pass-v1`: **3/3**, compatible inputs, no reused sessions,
regressions or coverage gaps. All three new candidates passed Model numerics and
suppressed-launch controls. Executor wall time: **246.42 s**; qualification job:
**305 s**. Usage: **15 requests / 7,671 completion / 154,866 input tokens**;
provider cost unavailable. This single same-skills comparison validates the
comparison path and repeatability for these cells, not a skills-effect estimate
or scheduled coverage. The selected baseline remains unchanged.

The same Release retains `comparison-42129712051.tar.gz` (217 checked members),
SHA256 `125a6119a3cb14eb6c430a224db9da4c1b7e0771a898827cf42066d9ac289408`.
Copies also remain outside checkout on Mac and Linux VM under
`~/pyasc-qualification-runs/issue22-comparison-20261106/`. Both public archives
exclude private response/application state; their source hashes and attribution
were independently verified after download.
