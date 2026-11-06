# OpenCode profiles

Tracked templates make generation settings explicit instead of inheriting an
unknown global model configuration. The current bounded entry point is
`tests/tools/generation_acceptance.py`; it uses `cloud-glm-5.3` by default.
Set `PYASC_MODEL_PROFILE` to select another template. The older generative
collector also uses the same renderer; its campaign is not a current CI lane.

## Credentials and rendering

The default profile requires `OPENCODE_BASE_URL` (an OpenAI-compatible endpoint)
and `OPENCODE_API_KEY`. Supply both through your existing credential mechanism.
It selects `gateway/glm-5.3`, with a 32768-token output limit and enabled thinking, requesting high reasoning
effort. The endpoint must actually offer that model; a template does not
establish provider availability.

```bash
python3 tests/tools/opencode_profile.py --profile cloud-glm-5.3 \
  --skills on --require-credentials --out /path/to/project/opencode.json
```

Create the project directory first and link its `skills/` directory using the
repository installer, as shown in the root README. Missing required variables
are reported before a provider call. `--help` lists the available profiles.

Templates using `{env:VAR}` leave resolution to OpenCode; `${VAR}` is expanded
by the renderer. Rendered configs can therefore contain credentials. Keep them
outside the repository and out of evidence archives; do not print them into CI
logs. Use `--out /dev/null` for a credential-presence check.

### Shared executor binding

The shared generation executor selects its profile and exact expected response
model in the versioned generation matrix. Its plan binds the profile bytes,
selected provider, declared endpoint/credential references and model settings.
It resolves only those references: a DashScope profile cannot borrow
`OPENCODE_API_KEY`, and an Ollama profile cannot inherit a cloud credential.
Missing variables, changed profile bytes or unsupported provider configuration
stop before commands and requests. No model fallback is performed.

Bounded profiles must select one explicitly declared provider using
`@ai-sdk/openai-compatible`, an explicit output limit, and `baseURL`/`apiKey`
options. Endpoint values may be fixed URLs or `{env:NAME}` / `${NAME}` references;
credentials must use a reference, except the public `ollama` placeholder.
References use uppercase environment-variable names. The gate owns the resolved
credential; the isolated OpenCode process receives only a local gate token.
Secrets and values resolved from the environment are not added to the public plan.

Historical templates without an output limit still need an explicit bounded
configuration and actual qualification. This binding does not establish model
availability or multi-model coverage. The exact response model ID is checked
separately; an immutable provider revision remains unknown unless independently
established. A profile alias is not treated as such a revision.

Retained results from before this binding contract remain readable, with no claim
that their credential binding was validated. New-source receipts require the
binding fields; removing them is an error. Execution always requires a plan
rebuilt and approved against the current source, including its profile binding.

Each new request also retains an allowlisted public `request_settings` record:
the effective token cap, explicitly supplied sampling/seed and reasoning fields,
and the names of fields present on the wire. Omitted fields stay `null` with no
claim about provider defaults; these are sent settings, not evidence that the
provider honored them. Raw prompts, tool arguments and arbitrary request metadata
are excluded. Older receipts lacking this record do not establish those settings.

## Profile families

| Templates | Configuration |
|---|---|
| `cloud-glm-5.3` | Gateway profile (see qualification scope below); `OPENCODE_BASE_URL`, `OPENCODE_API_KEY` |
| `cloud-default` and other `cloud-*` templates | Historical DashScope configurations; `DASHSCOPE_API_KEY` |
| `local-*` templates | Ollama configurations; `OLLAMA_BASE_URL` |

The templates are retained for explicit use. Historical model access checks,
nightly jobs and benchmarks do not establish that these alternatives work now.
See the template itself for its model identifier and limits. A different model
or runtime requires its own qualification.

## External directory access

A project may link skills from a checkout elsewhere on disk. The renderer
resolves that checkout's actual path and grants access to it. OpenCode uses the
last matching permission rule: the catch-all denial comes first, scoped grants
next, and explicit template restrictions last. Additional roots can be granted
with `--allow-root DIR`; no personal workspace layout is required.

`--skills off` removes the rendered discovery setting. It does not isolate
skills from a user's global configuration and is not, by itself, a controlled
skills-on/off experiment.

## Reasoning and qualification scope

GLM-5.3 uses [forced thinking](https://docs.z.ai/guides/capabilities/thinking-mode).
Use only `reasoningEffort: "high"` in this OpenCode profile. OpenCode 1.19.0
[translates it to wire field `reasoning_effort`](https://github.com/anomalyco/opencode/blob/v1.19.0/packages/llm/src/protocols/openai-chat.ts);
configuring both names is redundant. A loopback HTTP serialization probe against
the installed 1.19.0 CLI confirmed enabled thinking and `reasoning_effort=high`;
it did not invoke a remote model. A gateway may still translate these options,
so retain actual session/usage evidence rather than infer an internal budget.

The [archived hardware qualification](../../../docs/hardware-lane.md) used the
then-current low-effort profile. Changing the request to high effort does not
requalify that configuration or rewrite the archived run. The bounded CANNBench
CI caller overrides the output cap to 24000 tokens, including reasoning; it
retains the model response and blocks incomplete candidates instead of treating
a token-limited response as success.
