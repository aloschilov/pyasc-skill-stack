# opencode profile templates

These JSON files are templates loaded by
[`tests/tools/collect_generative_evidence.py`](../../tests/tools/collect_generative_evidence.py)
via the `--model-profile <name>` flag. Each template provides the per-project
`opencode.json` for one model / provider combination.

## Template substitution

The collector reads the file as text, substitutes `${VAR}` patterns with the
current process environment (via `string.Template.safe_substitute`), and then
parses the result as JSON. Unset variables expand to empty strings.

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
| `cloud-default.json` | The canonical cloud reference profile. Pins `model: dashscope/glm-5` explicitly (so the resolved model never drifts with the global default) and embeds the DashScope provider + `${DASHSCOPE_API_KEY}` inline, so the resolved per-project `opencode.json` is fully self-contained — it no longer depends on a global `~/.config/opencode/opencode.json`. The template also adds baseline permissions. This is the profile the nightly P2/P3/P4/P6 decomposition runs against, so glm-5 is represented here and is not duplicated as a separate `cloud-dashscope-gate` leg. |
| `cloud-glm-5.1.json` | Routes every request to `glm-5.1` on Alibaba DashScope's OpenAI-compatible endpoint. Requires `DASHSCOPE_API_KEY` in the environment (substituted into `${DASHSCOPE_API_KEY}`). |
| `cloud-qwen3.7-max.json` | Same as `cloud-glm-5.1.json` but pins `qwen3.7-max`. Requires `DASHSCOPE_API_KEY`. |
| `cloud-glm-5.2.json` | Same as `cloud-glm-5.1.json` but pins `glm-5.2`. Cross-vendor comparison leg in `cloud-dashscope-gate` (callable with the current key). Requires `DASHSCOPE_API_KEY`. |
| `cloud-deepseek-v4-pro.json` | Same pattern, pins `deepseek-v4-pro`. **Held out of CI:** the current DashScope key returns `Model.AccessDenied` (verified 2026-07-24). Template kept so the leg can be enabled once access is granted. |
| `cloud-kimi-k2.7-code.json` | Same pattern, pins `kimi-k2.7-code`. **Held out of CI:** `Model.AccessDenied` on the current key. Template kept for when access is granted. |
| `cloud-minimax-m2.5.json` | Same pattern, pins `MiniMax-M2.5` (exact casing preserved). **Held out of CI:** `Model.AccessDenied` on the current key. Template kept for when access is granted. |
| `cloud-qwen3-coder-next.json` | Same pattern, pins `qwen3-coder-next`. **Held out of CI:** `Model.AccessDenied` on the current key. Template kept for when access is granted. |
| `local-qwen-coder-7b.json` | Routes every request to `qwen2.5-coder:7b` served by Ollama at `${OLLAMA_BASE_URL}` (default `http://127.0.0.1:11434/v1` via opencode CLI defaults). |
| `local-llama-3.1-8b.json` | Routes every request to `llama3.1:8b` served by the same Ollama endpoint. |
| `local-qwen3-coder-30b.json` | Routes every request to `qwen3-coder:30b` served by the same Ollama endpoint. This is the local model confirmed to clear the agentic harness (see `docs/local-model-feasibility.md`); it needs a large-memory Ollama host (e.g. Apple-silicon Metal), and the CI `nightly-gate` / `local-stability-gate` run it against the Mac's native Ollama via `host.docker.internal`. |
| `local-gpt-oss-120b.json` | Routes every request to `gpt-oss:120b` on the same Ollama endpoint. Comparison leg for `local-stability-gate` alongside qwen3-coder:30b (~68 GB; requires `ollama pull gpt-oss:120b` on the Mac host). |
| `yellowzone-glm-5.2.json` | Routes every request to `GLM-5.2` on the internal **Turing Yellow Zone** OpenAI-compatible gateway (`http://7.150.6.255:3300/v1`). Requires `YELLOWZONE_API_KEY` in the environment (substituted into `${YELLOWZONE_API_KEY}`). Used on the remote NPU box reached via `gcpty` — see [`docs/opencode-remote-yellowzone-setup.md`](../../docs/opencode-remote-yellowzone-setup.md) for the full install/config walkthrough (CA bundle fix, standalone binary, proxy-bypass launcher). |
