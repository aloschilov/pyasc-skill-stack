# Local-model feasibility for the pyasc skill stack (2026-06-11)

> Empirical spike answering: *can a locally-hosted LLM (Ollama) drive the
> opencode skill-stack workflow and generate a verifiable asc2 kernel?* This
> documents the root-cause of the long-standing local-model `F10_no_artifact`
> failures and what is required to unblock them. It is a feasibility probe, not
> a full sweep.

> **Update (2026-06-13): the RTX 4090 box (`192.168.0.229`) has been retired
> from CI and is no longer accessible.** The findings below are kept as the
> historical record of the spike. CI no longer runs on the 4090: `pr-gate` and
> `merge-gate` moved to GitHub-hosted `ubuntu-latest`, and the nightly
> generative gates (`nightly-gate`, `cloud-dashscope-gate`) now run alongside
> the local-stability/perf gates on the arm64 Mac.

## TL;DR

- The local `F10_no_artifact` failures are **not** primarily context
  truncation. The dominant blocker is **tool-calling**: `qwen2.5-coder`
  (7b/14b/32b) emits tool/skill calls as plain text instead of structured
  `tool_calls`, so opencode's agent loop stops on the first turn and never
  writes `kernel.py`.
- The underlying **coding capability exists**: a direct probe (model fed the
  task + the golden reference, no agent loop) had `qwen2.5-coder:14b` produce a
  kernel that **passes the CANN simulator** (static pass, score 16/16,
  runtime PASS).
- The unblock is a model that is **simultaneously tool-capable AND a strong
  agentic coder**. On the shared RTX 4090 the only model that fit current free
  VRAM and emits structured tool calls (`llama3.1:8b`) is too weak an agent
  (invokes the skill, then refuses / hallucinates).
- **Confirmed working: `qwen3-coder:30b` on the 128 GB Apple-silicon Mac drove
  the full skills-on agent loop end-to-end and produced a CANN-simulator-PASSING
  kernel** (Overall: pass, 23 tool calls, 151.7 s). This is the first local
  model to clear the agentic harness on this stack.

## Hardware context

| Host | Spec | Role | Constraint observed |
|------|------|------|---------------------|
| 4090 box (`192.168.0.229`, `IntelligentSystems`) | RTX 4090, 24 GB; 3x `pyasc-ci-runner` containers + GitLab | Linux CI runners; Ollama runs *inside* the runner containers | A foreign job (`malashe+ python tts_rnn_train.py`, ~12.9 GB, 1h+) left only ~11 GB free during the spike |
| Mac (`198.18.0.1`) | Apple **M5 Max**, **128 GB** unified memory, macOS 26.5.1 | arm64 runner (perf gate) | Unified memory removes the VRAM-contention ceiling; Ollama must run natively (Docker Desktop's Linux VM cannot use the Metal GPU) |

## Method

For each (model, skills-mode) the harness
[`tests/tools/collect_generative_evidence.py`](tests/tools/collect_generative_evidence.py)
drives `opencode run` against an isolated project for `abs/float16` (the
easiest, most-copyable cell), then runs static + semantic + (where available)
CANN-simulator verification. A throwaway opencode profile points the `ollama`
provider (`@ai-sdk/openai-compatible`) at the host's Ollama `/v1` endpoint.

To raise context past Ollama's 4096 default (the harness has no `num_ctx`
knob), a derived tag is created with a Modelfile:

```
FROM <base-tag>
PARAMETER num_ctx 16384
PARAMETER num_predict 4096
```

A "tier ladder" grades how far each run gets:

- **Tier 0** infra OK (model on GPU, no timeout)
- **Tier 1** `kernel_found` (moved off `F10_no_artifact`)
- **Tier 2** static + semantic valid
- **Tier 3** simulator PASS (a genuinely useful kernel)

## Findings on the RTX 4090 (shared, ~11 GB free)

### 1. VRAM gating

With ~11 GB free, none of the target larger models fit fully on the GPU:

| Model | Approx need (weights + KV@16K) | Fit in ~11 GB? |
|-------|-------------------------------|----------------|
| `codestral:22b` | ~15 GB | No (weights ~13 GB alone) |
| `qwen3-coder:30b` | ~20 GB | No |
| `qwen2.5-coder:32b` | ~23 GB | No |
| `qwen2.5-coder:14b` @ f16 KV | ~13 GB | Spills 24% to CPU |
| `qwen2.5-coder:14b` @ **q8_0 KV** | ~10 GB | 94% GPU / 6% CPU (1 layer) |

`qwen2.5-coder:14b` was the largest that fit, and only with `q8_0` KV-cache
quantization (`OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0`).

### 2. The context-truncation theory is refuted (for this case)

The earlier "default 4096 context truncates the prompt" hypothesis does not
explain the failure. At `num_ctx=16384` the skills-on run used only **~7,705
input tokens** — well within the window — and still produced no kernel.

### 3. Root cause: tool-calling

Sending an identical trivial tool to Ollama and inspecting the response:

| Model | Endpoint | `tool_calls`? | Behaviour |
|-------|----------|---------------|-----------|
| `qwen2.5-coder:14b` | `/v1` (openai-compat, used by opencode) | **No** | emits `{"name":...,"arguments":...}` as text |
| `qwen2.5-coder:14b` | `/api/chat` (native) | **No** | same |
| `qwen2.5-coder:7b` | `/api/chat` | **No** | same |
| `llama3.1:8b` | `/api/chat` and `/v1` | **Yes** | structured tool call |

The qwen2.5-coder Ollama template instructs the model to wrap calls in
`<tool_call>…</tool_call>`; the model emits **bare JSON without the wrapper**,
so Ollama's parser cannot extract it. An explicit forcing system prompt did
**not** fix this. This is a model-side limitation of qwen2.5-coder, independent
of opencode.

### 4. Coding capability exists (direct probe)

Bypassing the agent loop — feeding `qwen2.5-coder:14b` the task plus the golden
`abs_f16.py` as reference and asking for a complete `kernel.py`:

- static verify **pass**, score **16/16 accepted**, **simulator PASS**
  ("Kernel executed on Model/Ascend950PR_9599, verification passed").
- The diff vs. golden showed the kernel body reused verbatim **but the test
  harness correctly adapted to the task's requested shapes** (`[1,128]`,
  `[4,2048]`, `[32,4096]`) — task comprehension, not blind copy.

So the model *can* produce a passing kernel; it just cannot *drive the agent
loop* to do so.

### 5. End-to-end re-run with a tool-capable model

Switching the profile to `llama3.1:8b-ctx16k` (confirmed `tool_calls=YES`, 100%
on GPU):

| Run | `tool_use` fired? | What happened | Kernel |
|-----|-------------------|---------------|--------|
| `qwen2.5-coder:14b`, skills-on | No (call as text) | opencode stops turn 1 | none (F10) |
| `llama3.1:8b`, skills-on | **Yes — skill invoked** | model then refuses: "I cannot assist with completing your workflow for you" | none (F10) |
| `llama3.1:8b`, skills-off | Yes (`webfetch`) | chats/hallucinates ("Ascend910 MLU", "C++ HAL") instead of writing a file | none (F10) |
| `qwen2.5-coder:14b`, **direct probe** | n/a (no loop) | full kernel | **simulator PASS** |

The tool-calling fix is **necessary and real** (the skill workflow fired for
the first time) but **not sufficient with a small tool-capable model**:
`llama3.1:8b` is too weak an agent to complete the multi-step workflow.

## Findings on the Mac (Apple M5 Max, 128 GB) — `qwen3-coder:30b`

**This is the unblock, confirmed working end-to-end.** Running the *actual*
skills-on agent loop (not the direct probe) with `qwen3-coder:30b-ctx32k`:

- Tool-calling: `tool_calls=YES` via `/v1` (proper structured call with an `id`).
- Loads **100% on GPU (Metal)**, 21 GB, `num_ctx=32768` — no spill (128 GB
  unified memory removes the VRAM ceiling).
- Agent ran **151.7 s** and drove the full workflow: **23 tool calls** —
  `skill`x4, `bash`x7, `read`x4, `grep`x2, `write`x2, `edit`x1, `todowrite`x3.
- Produced `kernel.py` **and** `design.md`.
- **static pass, semantic pass, score 16/16 accepted, simulator runtime
  PASS → Overall: PASS.**
- Token usage: 474,410 input (cumulative across 22 turns of skill content +
  tool results), 5,218 output.

| Run | tool_use | Workflow | Kernel |
|-----|----------|----------|--------|
| `qwen3-coder:30b`, skills-on, **Mac** | **23 calls** (skill/bash/read/grep/write/edit) | **completes** | **simulator PASS (Overall pass)** |

So a single tool-capable agentic coder, given enough memory, takes the local
skill stack all the way from prompt to a CANN-verified kernel. The kernel uses
the correct asc2 API/tiling and a model-authored docstring (i.e. not a verbatim
golden copy).

### Notes / caveats (Mac)

- The harness `script -qc` wrapper is **util-linux-only**; macOS ships BSD
  `script`. A one-line platform shim
  (`script -q /dev/null sh -c <cmd>` on Darwin) was applied to the Mac's
  throwaway repo mirror to run. **The canonical repo is still Linux-only here**
  — worth upstreaming if the Mac becomes a supported runner.
- Ollama must run natively on macOS (`brew install ollama`); Docker Desktop's
  Linux VM cannot use the Metal GPU.
- `python3.10` was shimmed to Homebrew `python3.13` (collector hardcodes
  `python3.10`); PyYAML installed with `--break-system-packages`.
- Runtime verification used the arm64 sim image
  (`ghcr.io/aloschilov/pyasc-sim-perf:py3.11-arm64`, set via `PYASC_SIM_IMAGE`).

## Conclusions

- The historical 0%/`F10` local results conflate two distinct issues. The
  decisive one is **tool-calling**, not context or raw capability.
- `qwen2.5-coder` at any size is a dead end for the *agentic* skill harness on
  Ollama: strong coder, cannot emit structured tool calls.
- A usable local model must be **tool-capable + strong-coder + agentic** in one.
  `qwen3-coder:30b` satisfies all three: on the 128 GB Mac it ran the full
  skills-on workflow and produced a simulator-verified kernel. It is VRAM-gated
  on the shared 4090 (needs a quiescent GPU window) but trivial on the Mac.
- **Recommended local target: `qwen3-coder:30b`.** For the 4090, run it only
  when the GPU is free (~20 GB); on the Mac it is the default choice. Avoid the
  `qwen2.5-coder` family for the agentic harness (tool-calling dead end).
- If the Mac is to be a supported local runner, upstream the macOS `script`
  shim and a non-hardcoded Python interpreter in
  [`tests/tools/collect_generative_evidence.py`](tests/tools/collect_generative_evidence.py).

## Reproduction

The spike used throwaway artifacts (derived `-ctx16k` tags, temp profiles,
kept project dirs) under `/home/loschilov_aa/ar/work-1/spike` (4090) and
`/Users/aloschilov/ar/spike` (Mac). Core commands:

```bash
# raise context (Ollama has no per-request num_ctx via openai-compat)
printf 'FROM <base>\nPARAMETER num_ctx 16384\nPARAMETER num_predict 4096\n' > Modelfile
ollama create <base>-ctx16k -f Modelfile

# tool-calling check (the decisive diagnostic)
curl -s http://127.0.0.1:11434/v1/chat/completions -d @tool_probe.json | jq '.choices[0].message.tool_calls'

# end-to-end via the evidence harness
python3.10 tests/tools/collect_generative_evidence.py \
  --op abs --dtype float16 --skills-mode on \
  --opencode-config <profile>.json --model-profile <label> \
  --max-attempts 2 --timeout 1200 --runtime --runtime-backend docker --allow-dirty-pyasc
```
