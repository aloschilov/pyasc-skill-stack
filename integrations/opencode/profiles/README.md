# opencode profile templates

These JSON files are templates rendered by
[`tests/tools/opencode_profile.py`](../../../tests/tools/opencode_profile.py).
Each template provides the per-project `opencode.json` for one model /
provider combination.

Two entry points consume them, and both go through that one module so they
cannot disagree:

- [`collect_generative_evidence.py`](../../../tests/tools/collect_generative_evidence.py),
  via `--model-profile <name>`.
- [`tests/integration/test-kernel-generation.sh`](../../../tests/integration/test-kernel-generation.sh),
  via `PYASC_MODEL_PROFILE` (default `cloud-glm-5.3`).

The shell test used to write no config at all and inherit the host's
`~/.config/opencode/opencode.json`. That is how the truncation recorded in
[`docs/verification-status.md`](../../../docs/verification-status.md) went
unexplained for so long: the global config on the affected machine caps
`gateway/glm-5.3` at `output: 8192`, and nothing in the repository said so.
A harness that inherits an unknown ceiling is not measuring what it claims.

You can render a profile by hand to see exactly what a run will use:

```bash
python3.10 tests/tools/opencode_profile.py --profile cloud-glm-5.3 --skills on
```

## Template substitution

Two placeholder spellings appear, resolved at different times:

| Form | Resolved by | When |
|------|-------------|------|
| `${VAR}` | `opencode_profile.py`, via `string.Template.safe_substitute` | at render time; unset names are left verbatim |
| `{env:VAR}` | opencode itself | when it loads the rendered config |

Either way, an unset variable degrades to an empty string somewhere
downstream and surfaces as an opaque provider-side authentication error.
Pass `--require-credentials` (the shell test always does, and the collector
prechecks before starting a sweep) to get the variable named up front
instead:

```
ERROR: .../cloud-glm-5.3.json references environment variables that are not set:
  - OPENCODE_API_KEY
  - OPENCODE_BASE_URL
```

## External directory access

The test projects are temporary directories whose `skills/` and `golden/`
entries are symlinks back into this repository, so from opencode's point of
view the skill stack is an *external* directory. Templates close that door
with `"external_directory": {"*": "deny"}`, which would also close it on the
thing under test — the agent would run with no skills while the harness
reported `skills_mode=on`, a silent wrong answer rather than a crash.

`opencode_profile.grant_repo_access()` therefore injects an `allow` for this
checkout's root, resolved at render time, ahead of the catch-all deny. Add
further roots with `--allow-root DIR`. An exact key already present in a
template still wins, so a profile can deliberately deny a path the grant
would otherwise open.

This replaced a hardcoded `~/workspace/pyasc-skill-stack/*` in
`cloud-default.json`, which was one developer's layout and granted nothing on
any other machine.

This is how the local-model profiles target an Ollama sidecar without baking a
hostname into the repo:

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1 \
  python3.10 tests/tools/collect_generative_evidence.py \
    --op abs --dtype float16 \
    --model-profile local-qwen-coder-7b \
    --skills-mode on
```

The cloud DashScope profiles use the same mechanism for the API key, so the
key is never committed to the repo:

```bash
DASHSCOPE_API_KEY=sk-... \
  python3.10 tests/tools/collect_generative_evidence.py \
    --op abs --dtype float16 \
    --model-profile cloud-glm-5.1 \
    --skills-mode on
```

The collector merges the `skills.paths` setting on top of the template at
runtime — keep templates skills-agnostic so the same file works for both
`--skills-mode on` and `--skills-mode off`.

## Profiles

| File | Purpose |
|------|---------|
| `cloud-glm-5.3.json` | **The default.** Routes to `glm-5.3` through an OpenAI-compatible gateway at `{env:OPENCODE_BASE_URL}` with `{env:OPENCODE_API_KEY}`. Two settings carry the weight: `limit.output: 32768` and reasoning disabled (`thinking.type: disabled`, `reasoningEffort: low`). Together they address the truncated generations in `docs/verification-status.md`, where an 8192-token ceiling inherited from a global config cut kernels off mid-file. |
| `cloud-default.json` | The former default, kept as the DashScope reference. Pins `model: dashscope/glm-5` explicitly (so the resolved model never drifts with the global default) and embeds the DashScope provider + `${DASHSCOPE_API_KEY}` inline, so the rendered `opencode.json` is self-contained. No longer the collector default: it needs a `DASHSCOPE_API_KEY` that no runner here holds, so as a default it could not complete a run. |
| `cloud-glm-5.1.json` | Routes every request to `glm-5.1` on Alibaba DashScope's OpenAI-compatible endpoint. Requires `DASHSCOPE_API_KEY` in the environment (substituted into `${DASHSCOPE_API_KEY}`). |
| `cloud-qwen3.7-max.json` | Same as `cloud-glm-5.1.json` but pins `qwen3.7-max`. Requires `DASHSCOPE_API_KEY`. |
| `cloud-glm-5.2.json` | Same as `cloud-glm-5.1.json` but pins `glm-5.2`. Cross-vendor comparison leg in `cloud-dashscope-gate` (callable with the current key). Requires `DASHSCOPE_API_KEY`. |
| `cloud-deepseek-v4-pro.json` | Same pattern, pins `deepseek-v4-pro`. **Held out of CI:** the current DashScope key returns `Model.AccessDenied` (verified 2026-07-24). Template kept so the leg can be enabled once access is granted. |
| `cloud-kimi-k2.7-code.json` | Same pattern, pins `kimi-k2.7-code`. **Held out of CI:** `Model.AccessDenied` on the current key. Template kept for when access is granted. |
| `cloud-minimax-m2.5.json` | Same pattern, pins `MiniMax-M2.5` (exact casing preserved). **Held out of CI:** `Model.AccessDenied` on the current key. Template kept for when access is granted. |
| `cloud-qwen3-coder-next.json` | Same pattern, pins `qwen3-coder-next`. **Held out of CI:** `Model.AccessDenied` on the current key. Template kept for when access is granted. |
| `local-qwen-coder-7b.json` | Routes every request to `qwen2.5-coder:7b` served by Ollama at `${OLLAMA_BASE_URL}` (default `http://127.0.0.1:11434/v1` via opencode CLI defaults). |
| `local-llama-3.1-8b.json` | Routes every request to `llama3.1:8b` served by the same Ollama endpoint. |
| `local-qwen3-coder-30b.json` | Routes every request to `qwen3-coder:30b` served by the same Ollama endpoint. This is the local model confirmed to clear the agentic harness (see `docs/local-model-feasibility.md` on `archive/campaign-2026`); it needs a large-memory Ollama host (e.g. Apple-silicon Metal), and the CI `nightly-gate` / `local-stability-gate` run it against the Mac's native Ollama via `host.docker.internal`. |
| `local-gpt-oss-120b.json` | Routes every request to `gpt-oss:120b` on the same Ollama endpoint. Comparison leg for `local-stability-gate` alongside qwen3-coder:30b (~68 GB; requires `ollama pull gpt-oss:120b` on the Mac host). |

A `yellowzone-glm-5.2.json` was listed here until v0.1.0. The file itself was
never part of this tree — it documented an internal gateway reachable only
from one network — so the row described a profile no `--model-profile` call
could ever resolve.
