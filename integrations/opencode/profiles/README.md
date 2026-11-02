# OpenCode profiles

Tracked templates make generation settings explicit instead of inheriting an
unknown global model configuration. The current bounded entry point is
`tests/tools/generation_acceptance.py`; it uses `cloud-glm-5.3` by default.
Set `PYASC_MODEL_PROFILE` to select another template. The older generative
collector also uses the same renderer; its campaign is not a current CI lane.

## Credentials and rendering

The default profile requires `OPENCODE_BASE_URL` (an OpenAI-compatible endpoint)
and `OPENCODE_API_KEY`. Supply both through your existing credential mechanism.
It selects `gateway/glm-5.3`, with a 32768-token output limit and reasoning
disabled. The endpoint must actually offer that model; a template does not
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

## Profile families

| Templates | Configuration |
|---|---|
| `cloud-glm-5.3` | Current qualified gateway profile; `OPENCODE_BASE_URL`, `OPENCODE_API_KEY` |
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
