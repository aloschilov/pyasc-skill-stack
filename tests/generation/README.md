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

A future authorized manual run uses `--execute --approve-plan <exact plan_sha256>
--results-dir /outside-checkout/new-run`. It needs OpenCode, the locally available
digest-pinned simulator, and `OPENCODE_BASE_URL` / `OPENCODE_API_KEY`. The executor
never pulls an image, submits to CANNBench, or selects an NPU backend. No schedule
or generation CI trigger is added. Actual provider-backed qualification remains
pending; ordinary PR checks use fixed fixtures and local provider doubles only.

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

Remaining #22 work: provider-backed manual qualification, compatible baseline
comparisons/regression policy, and explicitly approved schedule plus end-to-end
scheduled acceptance. #23 skills-effect comparisons and #24 model diversity should
reuse these contracts; #20 owns separate reviewer/tuner handoffs.
