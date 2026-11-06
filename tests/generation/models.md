# Controlled model comparison

This is the model axis of the [paired experiment](paired.md), using its executor,
credential binding, session/context audit and numerical adapters. It is not a
second campaign runner. The manual comparative run is **not yet qualified**;
the policy is disabled and the workflows execute only when explicitly dispatched.

Two GLM-5.2 cohorts stopped on response-identity drift before a comparison was
possible. Each retained one candidate failure, one unresolved request and 22
unrun trials. The second, with auxiliary agents disabled, still received the
short alias instead of the expected dated ID. These are gaps, not model or
skills scores. Completed delivery and cleanup were reconciled; original
UNKNOWN accounting and 24,000-token reservations remain unchanged.

| Incomplete cohort | External archive | SHA256 |
| --- | --- | --- |
| [42189411115](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/42189411115) | `model-42189411115-blocked.tar.gz` | `807ecc9bdce1dfd6b757a9c206d6b3bc7aace86588d9629f575f04f0d747f72b` |
| [42192207099](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/42192207099) | `model-42192207099-blocked.tar.gz` | `736f35a8ee975566254c0903dd6e15c7a91610edfeafd8d07320e078e9af9a68` |

Both archives are retained outside the checkout on the qualification Mac and VM.
The current plan selects the separately configured DeepSeek model instead of
relaxing identity checks or pooling the incomplete cohorts. A small tool request
observed its response ID below; that is discovery, not OpenCode qualification.

`models.json` selects two distinct requested and expected response identities:

| Arm | Profile | Requested | Expected response |
| --- | --- | --- | --- |
| deepseek41 | cloud-deepseek-v4.1-flash-gateway | gateway/deepseek-v4.1-flash | deepseek-v4-1-flash-261017 |
| glm53 | cloud-glm-5.3 | gateway/glm-5.3 | glm-5-3-260814 |

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

Both profiles configure `thinking=enabled`, `reasoningEffort=high` and a 256,000
token OpenCode context window. This is client configuration, not a verified
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
lease. The periodic policy is disabled and there is no cron. Future activation
requires its own review, main-only origin, cadence/health checks and a dedicated
alert issue marked `<!-- pyasc-generation-health:models:v1 -->`. Model and context
coverage use separate histories and cannot satisfy each other's observers.
`generation-health.yml` accepts `experiment=models` for manual inspection.
Hosted/scheduled observation is not qualified merely by these offline tests.
