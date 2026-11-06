# Controlled model comparison

This is the model axis of the [paired experiment](paired.md), using its executor,
credential binding, session/context audit and numerical adapters. One complete
manual comparison is qualified below. The periodic policy is enabled after authorization on 2026-11-06;
the controller checks cadence daily and explicit manual execution requires plan approval.

## Manual qualification, 2026-11-06

[Run 42209794583](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/42209794583)
on `68bffbc6bc4400d4c25e369cb4a009d6a724a2a1` completed all 24 predeclared
trials with known usage and no coverage gaps. The denominator includes all
13 candidate failures; no candidates were repaired or selected after the run.

| Model | P3: tool schema, no selected skills | P6: four selected skills |
| --- | --- | --- |
| DeepSeek V4.1 Flash | 0/6 qualified | 6/6 qualified |
| GLM-5.3 | 0/6 qualified | 5/6 qualified |

The 11 accepted candidates produced 33 JIT and 33 actual Model launches,
75 output/input numerical checks and 11 rejecting suppressed-launch controls.
The failures were two static contract violations and eleven JIT failures from
unsupported APIs or tensor-address operations. The executor took 2,130.112573 s;
the generation job took 2,178 s, including setup and collection. Its wait after
plan preparation was 3 s. There were 88 provider requests (47 DeepSeek, 41 GLM),
133,420 output and 954,823 input tokens. Monetary cost is unavailable.
All 100 source hashes and the archived report were independently rechecked;
the local GET-only observer returned `verified-manual`. All 118 own process
groups and 22 qualification containers were absent, and the executor lease
was released. This cohort did not qualify hosted observation or alerts; subsequent manual
controller acceptance is recorded below. Actual cron delivery remains unverified.

Each contrast has six fixed task/repetition pairs. P6−P3 is +1.0 for DeepSeek
(conditional 95% interval −0.1089 to 1) and +0.8333 for GLM (−0.2756 to 1).
GLM−DeepSeek is 0 under P3 (−1 to 1) and −0.1667 under P6 (−1 to 0.9422).
All four conservative Hoeffding intervals include zero. This small fixed-task
comparison establishes the measurement path, not general skills benefit,
model ranking, independent providers, kernel performance or NPU qualification.

[Durable qualification archive](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-models-42209794583):
`qualification-models-42209794583.tar.gz`, SHA256
`1244eaefad5760fdc1a3e955fcbc267023af71d3bb6e2a9f8d597d9652c78edb`.
Copies are retained on both qualification hosts outside checkout. The archive
contains unchanged candidates, plan/source hashes, public request metadata,
reports, observer, timings and cleanup; private provider bodies are excluded.

## Separate unsuccessful attempts

Earlier attempts remain preserved outside checkout on both qualification hosts.
Their UNKNOWN requests retain their original accounting and 24,000-token
reservations; they were abandoned without replay after own-process cleanup.
They are never pooled into the accepted comparison. The CLI-download failure
made no provider request.

| Run | Outcome | Archive SHA256 (`model-RUN-blocked.tar.gz`, except the preflight archive) |
| --- | --- | --- |
| 42189411115 | GLM-5.2 identity drift; 1 failed, 1 unknown, 22 unrun | `807ecc9bdce1dfd6b757a9c206d6b3bc7aace86588d9629f575f04f0d747f72b` |
| 42192207099 | GLM-5.2 identity drift; 1 failed, 1 unknown, 22 unrun | `736f35a8ee975566254c0903dd6e15c7a91610edfeafd8d07320e078e9af9a68` |
| 42195347301 | Incomplete response; 5 qualified, 10 failed, 1 unknown, 8 unrun | `759c3c48fbc0e9032d080d94e792529311ec7d3b3f3c84ca5391020b4f7d0f96` |
| 42202423456 | CLI download timeout; `model-42202423456-preflight.tar.gz` | `7018fadc2baf26e24b26f3990b981eec41892b91c897b3eefff07500be4ffcc4` |
| 42204519269 | Incomplete response; 1 qualified, 1 failed, 1 unknown, 21 unrun | `37db7a1304355b121993bb80272340b0e677b5a373182ea82fdd335e0738e6b0` |

The accepted cohort uses the independently configured DeepSeek profile and a
repaired incremental HTTP reader, with regression controls. It does not
reinterpret missing bytes or relax identity checks in earlier attempts.

## Fixed protocol

`models.json` selects two distinct requested and expected response identities:

| Arm | Profile | Requested | Expected response |
| --- | --- | --- | --- |
| deepseek41 | cloud-deepseek-v4.1-flash-gateway-low | gateway/deepseek-v4.1-flash | deepseek-v4-1-flash-261017 |
| glm53 | cloud-glm-5.3-low | gateway/glm-5.3 | glm-5-3-260814 |

Both use the same gateway transport, not independent providers. The separately
named gateway profile does not change existing provider templates.
Immutable provider/weights revisions are unknown; a response alias is not one.

The fixed matrix is add, reduction and broadcast × P3/P6 × two models × two
repetitions: 24 fresh OpenCode sessions. P3 has the tool schema but no selected
skills; P6 supplies the same four selected skills as the context experiment.
Within each task, the second repetition reverses the entire model/context
order. The seed controls ordering, not model sampling.

Every arm uses the same task hashes, pinned Model runtime, evaluator, one attempt,
eight requests at most, 24,000 output tokens per response including reasoning,
900 seconds for generation and 300 for qualification. Worst-case reservations
are 4,608,000 output tokens and 28,800 execution seconds. Input tokens are measured,
not capped; provider cost stays null when unavailable. Queue/setup/collection
are outside the execution bound. The CI job has a 600-minute deadline.

The periodic profiles configure `thinking=enabled`, `reasoningEffort=low` and a 256,000
token OpenCode context window. The earlier qualified cohort above used high reasoning; its results are not pooled
with this operational profile. Two subsequent high-reasoning controller attempts
stopped on incomplete GLM streams; low reasoning is a new explicit condition for
both arms, not a claim that the transport fault is fixed. Existing high profiles
remain available. Task inventory, repetitions, token/time budgets and qualification
criteria are unchanged. This is client configuration, not a verified
provider capacity or a promise that reasoning levels have identical semantics.
Temperature, top-p and seed are unspecified. Public evidence records the actual
serialized allowlist and unknown defaults; it does not claim provider compliance.
The reporter rejects dropped requested options or different serialized settings
between arms. The new plan disables auxiliary title/summary agents in both arms
and requires a tool schema on every request; generation uses the exact per-arm
response ID above, never an alias allowlist. This is supported by the pinned
[OpenCode 1.19 agent configuration](https://github.com/anomalyco/opencode/blob/v1.19.0/packages/opencode/src/agent/agent.ts).
A gateway repeating identical cumulative usage on the final choice and the
following usage-only chunk is counted once, only with one response ID, one
finish event and matching counters. Conflicting counters remain unresolved.
The gate drains bounded `read1` fragments and retains received body bytes
privately, including a partial final HTTP chunk. A streaming reply is usable only if those unchanged
bytes contain the final SSE `DONE`, one finish, the exact model and valid usage;
the HTTP framing error remains recorded. Missing application completion or an
incomplete non-streaming body stays UNKNOWN and prevents another request.
No fallback model, alternate credentials, or hidden repair is used.

Each unchanged candidate must pass static validation, JIT, actual Model execution
with numerical output/input checks, and a rejecting suppressed-launch control.
These checks are simulator qualification, not NPU or kernel performance evidence.
Every repetition is retained, including known candidate and budget failures.
Unavailable providers/models and unknown responses stop the experiment as blocked;
they cannot become numerical failures or a skills benefit. A partial cohort is
not resumed by changing its model: preserve it, reconcile outstanding work, then
declare a separate plan/run. Never pool cohorts or select only the best candidate.

Reports show P6−P3 separately for each model and the model difference separately
under P3 and P6, six predeclared pairs per contrast. Conditional uncertainty has
the same independence and small-sample limitations as the context experiment.
No global skills benefit, model ranking or hardware performance is implied.

```sh
# Offline; keep evidence outside checkout.
python3 tests/tools/generation_pairs.py plan --experiment models \
  --root /outside-checkout/model-plan

# A new directory and exact plan approval are required for execution.
python3 tests/tools/generation_pairs.py execute --experiment models \
  --plan /outside-checkout/model-plan/plan.json --approve-plan <digest> \
  --root /outside-checkout/new-model-run --state-dir /outside-checkout/shared-state

# GET-only verification of one completed manual CI run.
python3 scripts/generation_operations.py monitor --models \
  --inspect-run RUN_ID/ATTEMPT --root /outside-checkout/model-observation
```

`generation-models.yml` plans on PRs and defaults to `mode=plan` on manual
dispatch. Execution uses the shared simulator workflow and durable executor
lease. Its daily cadence check runs at 18:47 UTC (21:47 Moscow), with a
168-hour minimum execution interval, main-only origin and dedicated alert issue
#39 marked `<!-- pyasc-generation-health:models:v1 -->`. Model and context
coverage use separate histories and cannot satisfy each other's observers.
`generation-health.yml` accepts `experiment=models` for manual inspection.
Hosted/scheduled observation is not qualified merely by these offline tests.

## Integrated controller acceptance

The current integrated manual run and hosted failure/recovery receipts are
recorded in the [shared acceptance result](README.md#manual-controller-acceptance-2026-11-06).
The user accepted manual dispatch instead of cron for closure. The earlier
cohort above keeps its original scope; actual cron delivery remains unverified.
