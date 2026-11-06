# Paired context experiment

Work for [#23](https://github.com/aloschilov/pyasc-skill-stack/issues/23) uses the
shared bounded executor and numerical adapters. The manual experiment below has
24 auditable trials. Scheduling remains disabled. A complete experiment is not necessarily
a successful kernel in every trial, nor evidence that skills improve results.

## Manual qualification, 2026-11-06

[Run 42175778880](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/42175778880)
on `82f2110e9d00e78f7fcfefdaf704788fed122095` completed all **24 trials**,
with six valid pairs per contrast and no unknown requests or coverage gaps.
Plan SHA256: `cea0ec3ee85610af8c31df3992b58ac109f4341a9337a01317379bfc10b5856d`.
The model was GLM-5.3 (`glm-5-3-260814`), low reasoning, through OpenCode 1.19.0.

| Protocol | Qualified | Candidate rejected | Known budget exhausted |
|---|---:|---:|---:|
| P2 | 0/6 | 4 | 2 |
| P3 | 0/6 | 4 | 2 |
| P4 | 0/6 | 5 | 1 |
| P6 | 5/6 | 1 | 0 |

The five accepted candidates made **15 Model launches**, passed 33 output/input
checks and rejected all five suppressed-launch controls. Candidate bytes remained
unchanged. The P6 rejection was the second add trial's JIT failure; reduction and
broadcast passed twice each. A known budget failure is an observed failed attempt,
not a numerical pass. All 95 execution-source hashes matched the submitted commit.

Executor wall time was **1,983.08 s (33m03s)**; the preceding **648.65 s (10m49s)**
includes workflow queue and setup. The 91 provider requests used **88,958 output**
and **467,615 input tokens**; monetary cost is unknown. A post-run audit of retained
private requests matched their public hashes and found `max_tokens=8192`,
`reasoning_effort=low` and enabled thinking in all 91 requests. Other sampling
settings and seed were unspecified. This does not prove that the provider honored
those settings or identify an immutable weights revision.

P6-P3's observed qualification difference was **+0.8333**, with a conditional 95%
Hoeffding interval **[-0.2756, 1]**. P3-P2 and P4-P3 were zero, with intervals
[-1, 1]. P6-P4 was +0.8333 with the same wide interval, but replaces context
packages rather than isolating skills. These six-pair results include zero within
their intervals; they do not establish a general benefit, performance improvement,
other-model coverage or NPU qualification.

The [durable Release](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-paired-42175778880)
retains `qualification-42175778880.tar.gz`, SHA256
`0ae0f6b13c06ff1f06abff2ad44b665677439f4d4b6556fd8e087e411c2acb74`.
It contains the original public CI archive, strict report, timing, cleanup,
wire-setting audit and observer receipts. Its downloaded bytes and all indexed
members were checked; copies also remain outside checkout on the workstation and VM.
Private provider payloads are excluded. All recorded process groups and simulator
containers were absent after execution, and the executor released its shared latch.

The GET-only local observer independently returned `verified-manual` without
alerts. Hosted observer dispatch returned HTTP 404 because its workflow is not yet
on the default branch; no hosted job was created. Hosted and scheduled observer
qualification remain pending. No schedule, NPU run or automatic baseline promotion
was enabled.

## What the surviving scripts establish

`collect_generative_evidence.py` and `compare_skills_value.py` are historical
schema tools, not this experiment's acceptance path. Their useful intervention
labels survive, but they cannot establish the current paired claim:

- The old collector's P4 default points to a removed baseline AGENTS.md. Its
  workspace builder can expose `golden/` to both controls and treatment, and its
  optional fallback prompt changes the intervention between attempts.
- The old aggregator accepts static-only dashboard PASS, infers missing protocol
  IDs from filenames, and replaces an unknown monetary cost with zero. Its
  per-protocol means and marginal intervals do not require complete matched pairs
  with equal runtime, model, task, evaluator or budgets.
- A model ID plus nonzero tokens is weaker than native session/context provenance,
  unchanged-byte numerical qualification and suppressed-launch controls.

Historical output keeps its original meaning. Do not feed it into the new paired
reporter or describe a regenerated P-label as a replication of those experiments.

## Explicit interventions

Every leg receives the same task, dtype, shapes, block count, required host/device
ABI and restrictions. Those necessities are not counted as extra guidance.

| ID | Added context | Primary comparison |
|---|---|---|
| P2 | None beyond common task/ABI | Minimal control |
| P3 | Shared procedural guidance, without skill content | P3-P2: guidance |
| P4 | P3 plus the separately authored, hashed baseline notes | P4-P3: baseline notes |
| P6 | P3 plus the selected skills loaded through the native skill tool | P6-P3: skills package/retrieval |

P6-P4 replaces two context packages; it does not isolate the skills effect.
The new [baseline notes](context/baseline.md) have their own version and contain no
reference kernel. They do not pretend to restore the removed historical AGENTS.md.
Version 2 uses API names exercised by the qualified fixed fixtures. Version 1
incorrectly advertised `load`/`store` and a reduction shape rule; its pilot runs
are invalid controls and must not be pooled with version 2. The fixture API check
catches unsupported names; semantic changes to these notes still need review.

`paired.json` predeclares add, reduction and broadcast through the existing
matrix, all four legs and two repetitions per task: **24 trials**, six matched
pairs per contrast. SHA256 ordering with a recorded order seed permutes blocks
before execution. The second repetition reverses each task's first leg order,
counterbalancing every within-task comparison. The seed controls order only;
provider sampling has no qualified deterministic seed.

The model/profile/revision, runtime digest, task snapshot, evaluator, tool schemas
and per-trial limits remain equal. Default worst-case reservations are 24 model
attempts, 1,572,864 output tokens and 28,800 execution seconds. Each leg has the
same 900-second generation deadline and 300-second qualification deadline. The
single-run matrix retains its existing limits. Actual input/output
usage and time are recorded separately; input tokens and monetary charges are not
bounded by the output-token cap. No automatic model fallback or hidden repair is
allowed. Additional attempts, if configured, are fresh restarts; reviewer/tuner
repairs belong to #20 and are currently zero.

Specification version 2 explicitly selects `cloud-glm-5.3-low` and the expected
response ID `glm-5-3-260814` for every leg. It inherits tasks/runtime/budgets from
the seed matrix and binds its own profile in the plan. The qualified single-run
matrix keeps `cloud-glm-5.3`. The new profile keeps thinking enabled and requests
`reasoning_effort=low`, a supported GLM-5.3 setting in the
[provider API](https://docs.z.ai/api-reference/llm/chat-completion). An observed
response ID does not establish an immutable provider weights revision.

The `context_path_format=relative-read-call-v1` instruction supplies exact short
project-relative `read` arguments. Context files and permissions stay the same;
the full baseline must still be observed. Older plans retain their original
absolute prompt format and incomplete results. Earlier incomplete cohorts remain
archived outside checkout: an interrupted response with unknown usage
(`42161942577`), a high-reasoning response with contradictory usage counters
(`42167095606`), and a denied incorrectly reconstructed P4 path (`42173378734`).
None is reclassified or pooled with the accepted cohort. Low reasoning does not
repair provider accounting; strict counter checks remain. A changed prompt,
profile or deadline needs a new predeclared cohort after prior-latch reconciliation.
A timeout remains unresolved infrastructure, never a measured zero outcome.

## Isolation and evidence

Trials reuse the same executor lease, provider budget gate and simulator
qualification, with separate project, HOME, XDG directories, caches and native
sessions. A partial/unknown trial stops the remaining inventory and keeps the
shared unresolved-run latch. No new upload ledger or state-directory workaround
is introduced. Re-execution requires a new explicit plan/run after reconciliation.

Controls receive no skills, reference kernels, other candidates or oracle files
in their readable context. Read permissions name only the candidate and selected
context files; commands, networking and directory search remain denied in every
leg. OpenCode's glob/grep permissions match the search expression rather than its
path, and native internal tool-output directories have external-directory
exceptions. Disabling both search tools avoids widening the context through
those exceptions; explicit file reads and native skill loads remain available.
The audit checks native accesses, rejects automatic instruction loads, verifies
full skill/baseline content, and records what reached actual provider requests.
Prompt hashes bind both the supplied text and the exact single-argument quoting
performed by the pinned [OpenCode 1.19 CLI](https://github.com/anomalyco/opencode/blob/v1.19.0/packages/opencode/src/cli/cmd/run.ts#L288).
The audit does not broadly normalize whitespace or remove arbitrary escaping.
Native permission denials are retained as attempted accesses, not as loaded
context. A successful access outside the declared context or an unaccounted search
still invalidates a trial; an empty output or generic tool error does not waive it.
Tool names/argument-schema hashes must match across the experiment. This audits
the pinned OpenCode permission model and observed behavior; it is not a sandbox
against a hostile host administrator or malicious replacement client.

Original candidate bytes, source/context/task/model/prompt hashes, raw native
events, request hashes/audits, usage, negative controls and cleanup receipts remain
outside checkout. Raw provider requests/responses and application state stay in
private directories and must never enter published evidence.

## Paired reporting

The reporter validates each trial against its predeclared plan and exact oracle.
An actual static/JIT/numerical candidate rejection can be a valid zero outcome.
A measured output/request cap can also be a zero qualification outcome when all
provider responses and required context attribution are complete. It is labeled
`budget-exhausted`, never numerical failure or a successful generation. The client
is stopped, and no partial output is qualified. Unknown delivery cannot use this
classification, even if a cap was also reached.

For P2/P3 only, the first tool-bearing response can exhaust its output cap before
OpenCode emits any CLI event. After stopping the client, the executor may bind
that known failure to a read-only backup of its private OpenCode 1.19 database:
exactly one session, the original user prompt and the initial assistant message,
with no assistant/tool parts or candidate. Earlier auxiliary responses must have
no tool schema and finish normally; the capped response was withheld by the gate.
The public receipt contains identity and hashes, while the database stays private.
This receipt establishes only an empty failed session. It cannot replace events,
prove P4/P6 context loads, waive unknown usage, or qualify a candidate. Historical
pilots with missing receipts remain incomplete; they are not retroactively accepted.

Timeouts, unavailable infrastructure, missing attribution, changed tools/prompts,
unresolved usage and not-run trials remain separate from candidate failure.
An invalid or missing control cannot create a claimed gain. Each contrast keeps
its original denominator; any gap suppresses its aggregate effect estimate.

Reports retain every paired qualification difference and descriptive attempt,
time, input/output-token and cost delta. Unknown costs remain `null`. Time means
generation/qualification wall time, not device-kernel performance.

For qualification differences D in [-1, 1], the report gives the conservative
95% Hoeffding interval: mean(D) ± sqrt(2 log(40) / n), clipped to [-1, 1]. This
requires independent paired trials and describes the recorded fixed tasks/model;
it does not cover correlated provider behavior or generalize to other operators
or models. Six pairs normally give a very wide interval, even if all outcomes
agree. The report never automatically declares a general skills benefit.

```bash
# Offline planning only; does not call a model or simulator.
python3 tests/tools/generation_pairs.py plan --root /outside-checkout/paired-plan

# Explicit execution; inspect/approve the exact plan digest first.
python3 tests/tools/generation_pairs.py execute \
  --plan /outside-checkout/paired-plan/plan.json \
  --approve-plan <exact-digest> --root /outside-checkout/new-paired-run \
  --state-dir <the-existing-shared-generation-state-directory>

# Read retained evidence without running snapshot code, model or simulator.
python3 tests/tools/generation_pairs.py report --root /outside-checkout/paired-run
```

The **Paired generation experiment** workflow plans on PRs and defaults to plan
on manual dispatch. To execute, choose `mode=execute` and provide the exact
reviewed `approve_plan` for that revision. It reuses the generation simulator
worker, pinned OpenCode/runtime provisioning, shared latch and public archive;
the 600-minute job limit includes the eight-hour maximum execution reservation.
An incomplete experiment fails the CI job. A complete experiment may contain
candidate failures; inspect its paired report rather than interpreting green CI
as a skills benefit. This workflow has no cron; its periodic policy is disabled.

The existing **Generation coverage observation** workflow can inspect a retained
paired run with `experiment=paired` and an explicit `inspect_run=RUN_ID/ATTEMPT`.
It only fetches evidence, revalidates all pairs and checks freshness and origin;
it does not download the single-run historical baseline or execute snapshot code.
Green workflow status without complete, intact evidence is insufficient. The
explicit-run result remains `manual-inspection`, cannot qualify a schedule or send a schedule
alert, and makes no general skills-benefit claim. The equivalent local command is:

```bash
python3 scripts/generation_operations.py monitor --paired \
  --inspect-run RUN_ID/ATTEMPT --root /outside-checkout/paired-observation
```

## Prepared periodic use (disabled)

The `periodic` policy in `paired.json` reuses the controller, shared executor latch
and health monitor. It defaults to disabled, a 168-hour minimum interval, a
192-hour freshness window and a ten-hour stalled-run threshold. PRs only plan;
manual execution still requires the exact reviewed digest regardless of policy.

Future scheduled execution is main-only and must be enabled explicitly, with an
activation timestamp and a separately marked alert issue. Before preparation and
again before the execution worker, GET-only history checks enforce the interval.
Any recent scheduled attempt consumes it, including failed or cancelled work;
scheduled reruns are rejected. Missing, duplicated or truncated history blocks
spending. A successful controller-only skip is distinguished through its exact
job receipt (`prepare` succeeded, `paired` skipped); it does not reset the interval
or establish fresh generation coverage. Thus a future schedule may poll more
often than it executes. At most 20 recent prior runs are inspected before refusing
spending. A new explicitly approved manual run remains possible after reconciling
the shared latch. There is no second ledger or automatic repair campaign.

Without `inspect_run`, the paired monitor observes scheduled coverage only when
this policy is enabled. It verifies complete matched evidence (candidate failures
may be valid experimental outcomes), reports missing/failed/stalled/stale runs,
and uses an alert issue starting with `<!-- pyasc-generation-health:paired:v1 -->`.
It cannot overwrite the single-run alert channel. Both policies and workflow cron
triggers remain off; the future scheduled health workflow selects both policies,
and an inactive policy stays disabled without sending notifications.

The manual experiment and local evidence observer are qualified above. Hosted
observer qualification, actual periodic activation and scheduled-run qualification
remain separate work; activation requires a separately reviewed change. Offline
controller tests do not establish scheduled coverage. No NPU spending is implicit.
