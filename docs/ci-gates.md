# CI Gate Tiers

Three gate tiers ensure fast feedback on PRs while reserving expensive checks for merge and nightly runs.

## Tiers

| Tier | Trigger | Time budget | What runs |
|------|---------|-------------|-----------|
| **pr** | Every push / PR | < 30s | L1 unit tests + JIT verification of golden kernels |
| **merge** | Merge to main | < 5 min | PR tier + simulator execution of golden kernels |
| **nightly** | Scheduled (daily) | 15-30 min | Merge tier + L2 behavior + L3 agentic integration |

## Entry point

```bash
bash tests/ci-gate.sh --tier pr        # Fast PR gate
bash tests/ci-gate.sh --tier merge     # Merge gate (includes simulator)
bash tests/ci-gate.sh --tier nightly   # Full nightly run
```

## PR gate (`--tier pr`)

Runs in under 30 seconds. Suitable for pre-commit hooks and PR checks.

1. `run-tests.sh --fast` -- L1 structural and content validation (skills, agents, teams)
2. JIT verification of all golden kernels via `pytest_verify_kernel.py` -- confirms pyasc JIT compilation works without needing the simulator
3. `check_evidence_paths.py` -- rejects machine-specific `/home/` or `/Users/` paths in committed perf evidence JSON

No network, no simulator, no opencode required.

## Merge gate (`--tier merge`)

Runs in under 5 minutes. Requires CANN simulator environment.

1. Everything in PR gate
2. Simulator execution of all golden kernels via `run_and_verify.py --mode simulator` -- confirms numerical correctness with `np.testing.assert_allclose`

Requires: `source $HOME/Ascend/cann/set_env.sh` and `LD_LIBRARY_PATH` set. See [cann-setup.md](cann-setup.md).

The GitHub Actions `merge-gate` job runs **natively on the self-hosted arm64
Mac runner** (no x86 emulation), pulling the arm64 leg of the multiarch
`pyasc-sim` image. It is still sharded via the 4-shard matrix, though on the
single Mac runner the shards currently serialize; the matrix is retained while
we gather wall-clock metrics before deciding on the shard count.

`bfloat16` goldens are **first-class** here: the native arm64 pyasc build has
full bf16 IR lowering, so `*_bf16.py` goldens (e.g. `layer_norm_v4_bf16`) are
sharded and verified in the same loop as every other capability cell -- they
are part of the blocking merge-gate, not a separate step. (The old GitHub-hosted
amd64 build lacked bf16 IR lowering, which is why bf16 used to be carved out into
a non-blocking `perf-gate` step; that carve-out is gone.)

## Nightly gate (`--tier nightly`)

Runs in 15-30 minutes. Requires opencode CLI and CANN simulator.

1. Everything in merge gate
2. `run-tests.sh --all` -- L2 behavior tests (agent trigger correctness, premature action detection) and L3 integration tests (full agent-in-the-loop kernel generation)

Requires: opencode CLI on PATH, CANN simulator environment.

The GitHub Actions `nightly-gate` job runs the Phase 0 protocol-axis matrix
(P2/P3/P4/P6) against the remote DashScope **`cloud-default`** profile (glm-5),
gated on the `DASHSCOPE_API_KEY` secret. It is **report-only** (no P6 hard-fail
threshold). Cloud inference needs no host Ollama, so the legs only serialize on
the Mac runner for the camodel docker sim verify. The
**`cloud-dashscope-gate`** job is also enabled. It runs a cross-vendor
comparison matrix of three DashScope models — the two incumbents
(`glm-5.1`, `qwen3.7-max`) plus `glm-5.2` — each with the full skills on+off
A/B. Four further flagships (`deepseek-v4-pro`, `kimi-k2.7-code`,
`MiniMax-M2.5`, `qwen3-coder-next`) were evaluated for the comparison set but
the current DashScope key returns `Model.AccessDenied` for them (verified
2026-07-24); their profile templates exist and are ready to add to the matrix
once model access is granted. Every listed profile is measured over the
**same** unified generative cell list (see "Unified kernel list" below), so all
dashboard cards compare an identical set of kernels. This grows the leg from
2×2=4 to 3×2=6 serialized cloud runs on the single Mac nightly runner
(`continue-on-error: true`); it is gated on the `DASHSCOPE_API_KEY` secret and
self-skips if unset. It uses the same **1200 s** per-attempt agent budget as
`nightly-gate` and `local-stability-gate` (previously a tighter 420 s that lost
even trivial cells like `abs/float32` to `exit 124` timeouts), so the on/off A/B
is apples-to-apples across every gate.

The **`local-stability-gate`** compares **`qwen3-coder:30b`** vs
**`gpt-oss:120b`** (skills on/off). Both models must be pre-pulled on the Mac's
native Ollama (`ollama pull qwen3-coder:30b`, `ollama pull gpt-oss:120b`); legs
skip cleanly when a model is missing.

### Host memory (128 GB Mac)

Every CI job now runs natively on the single self-hosted arm64 Mac runner
(`pr-gate`, `merge-gate`, `perf-gate`, `nightly-gate`, `local-stability-gate`,
`skills-value-report`), so they all serialize on that one runner and share the
128 GB host with the ~46 GB Parallels VM (the dev Linux box), the Docker Desktop
VM (camodel sims), and macOS. `gpt-oss:120b` alone is ~68 GB resident and `qwen3-coder:30b` is
~18 GB, so **two co-resident models would overrun the host and thrash swap.**
Because Ollama keeps a model warm for `keep_alive` (5 min default), the
local-model legs (`local-stability-gate`) run a **Free host Ollama memory** step
([tests/tools/free_ollama_memory.py](../tests/tools/free_ollama_memory.py))
that unloads any model a prior leg left warm, bounding the peak Ollama
footprint to the single model the upcoming leg loads. (`nightly-gate` runs on
cloud DashScope and loads no local model.)

Recommended host-side belt-and-braces (set on the Mac's native Ollama, which
CI cannot configure): `OLLAMA_MAX_LOADED_MODELS=1` and a short
`OLLAMA_KEEP_ALIVE` (e.g. `1m`). If the Parallels VM does not need to be up
during a nightly, shrinking its RAM reservation frees the most headroom for
`gpt-oss:120b`.

## Perf gate (`perf-gate`, report-only)

A separate **nightly, non-blocking** GitHub Actions job (`continue-on-error:
true`) that measures perf-vs-AscendC for every demo cell and publishes the
result to the dashboard. It is **not** part of `ci-gate.sh`; it runs only on the
schedule / `workflow_dispatch tier=nightly`, alongside `nightly-gate`.

For each cell the harness ([tests/tools/demo_vector_ops.py](../tests/tools/demo_vector_ops.py)
`--all`) builds the canonical `ops-math`/`ops-nn` AscendC reference and the
generated pyasc kernel on the same `Ascend950PR_9599` camodel, then computes
`ratio = ref_ticks / gen_ticks`. The demo cell list is **derived automatically**
from every `perf_ratio_demo` block in `capabilities.yaml` (via
[tests/tools/load_capability_cells.py](../tests/tools/load_capability_cells.py)),
so adding a new kernel only requires updating capabilities + harness op wiring,
As of the unified-coverage pass, **every** generative cell carries a
`perf_ratio_demo` block, so the perf gate measures all 19 cells (up from 11) —
there is a perf ratio for every operation row, no "—" placeholders. Four new
canonical `ops-nn` references were wired for this (`aclnnGelu`,
`aclnnLeakyRelu`, `aclnnMatmul` via `mat_mul_v3`, `aclnnSoftmax` via
`softmax_v2`) in [tests/tools/perf/ascendc_ref_runner.py](../tests/tools/perf/ascendc_ref_runner.py);
their sources are already baked into the perf image's `/opt/ops-nn`, so no image
rebuild is needed. Because the perf image strips `.git` from the vendored
sources, `ascendc_ref_runner._ensure_thirdparty_siblings` creates the `ops-base`
/ `ops-tensor` sibling dirs that `ops-nn`'s third-party cmake modules look for,
so their pinned-SHA `git checkout` step is bypassed and ops-nn references build
offline.
not a hand-maintained `CELLS` table. The 0.70 gate is **reported, never enforced**,
so documented honest misses (`apply_adam` ~0.46, `batch_norm_v3` ~0.10) stay
green.

- **Image:** runs inside the docker_full perf image
  `ghcr.io/<owner>/pyasc-sim-perf:py3.11`, which extends `pyasc-sim` with the
  vendored `ops-math`/`ops-nn` reference repos, the `pyasc-v2-eval` tree, and
  the `dav_3510`/`Ascend950PR_9599` simulators. Built **manually** on the host
  that has the private clones via
  [docker/build-perf-image.sh](../docker/build-perf-image.sh)
  (`docker/build-perf-image.sh --push`); CI only `docker pull`s it.
- **Output:** `evidence/perf-vs-ascendc/*.json` + an aggregated
  `evidence/perf-summary.json` (via
  [tests/tools/perf/aggregate_perf.py](../tests/tools/perf/aggregate_perf.py)),
  uploaded as the `evidence-perf` artifact.
- **Commit + publish:** the single-writer `skills-value-report` job merges the
  `evidence-perf` artifact, re-aggregates, and commits `perf-summary.json` +
  `perf-vs-ascendc/*.json` to `main`. The `pages.yml` `evidence/**` trigger then
  redeploys the dashboard, whose perf panel renders `perf-summary.json` (falling
  back to each cell's curated `perf_ratio_demo` in `capabilities.yaml` when no
  measured summary is present, e.g. local/dev renders).
- **Compiler SIMD-fusion A/B (same job):** the `perf-gate` job also runs
  [tests/tools/demo_vf_fusion.py](../tests/tools/demo_vf_fusion.py) `--all`, which
  recompiles each generated kernel with `--cce-simd-vf-fusion` **off vs on** on
  the same camodel (verifying the "pyasc2 uses no micro-api; the compiler does
  fusion" positioning — see
  [docs/perf-vs-ascendc-demo.md](perf-vs-ascendc-demo.md)). It is also
  report-only.
  [tests/tools/perf/aggregate_vf_fusion.py](../tests/tools/perf/aggregate_vf_fusion.py)
  writes `evidence/vf-fusion-summary.json` (per-cell `ticks_off`/`ticks_on`/
  `fusion_speedup` + an `improved`/`neutral`/`regressed` verdict); both it and
  `evidence/vf-fusion/*.json` ride the same `evidence-perf` artifact and are
  committed by `skills-value-report`. The dashboard renders them in a **Compiler
  SIMD fusion** panel. (bf16 golden verification no longer lives here -- it is a
  first-class part of the blocking `merge-gate`; see Merge gate above.)

Committed perf evidence uses **repo-relative paths** only (`golden/kernels/...`,
`evidence/perf/_build_cache/logs/...`). The PR gate runs
[tests/tools/check_evidence_paths.py](../tests/tools/check_evidence_paths.py) to
reject `/home/` and `/Users/` prefixes in perf/vf-fusion JSON.

The GitHub Actions `nightly-gate` (and local-stability matrix legs) discover
generative cells from `capabilities.yaml` via
[tests/tools/list_generative_cells.py](../tests/tools/list_generative_cells.py)
(every cell with a non-empty `prompt` on `Ascend950PR_9599`).

### Unified kernel list

All model legs — `nightly-gate`, `local-stability-gate`, and
`cloud-dashscope-gate` — enumerate their kernels from the **same**
`list_generative_cells.py` call, so the generative cell list is a single source
of truth (currently **19 cells**). Divergent dashboard denominators (e.g. some
cards showing 12 cells, others 15) are therefore never a config drift; they are
**stale evidence** from earlier eras when the matrix had fewer cells, because
the aggregator counts each profile's own evidence files rather than the
canonical list. To keep every profile comparable, a full-refresh nightly
re-measures all profiles (cloud + local) over the current list in one run so
every card reads N/N with the same N. The perf surface is unified the same way:
every generative cell now carries a `perf_ratio_demo` block, so the perf gate
reports a ratio for all 19 cells.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PYASC_PYTHON` | `python3.10` | Python interpreter with pyasc |
| `ASCEND_HOME_PATH` | (from set_env.sh) | CANN toolkit root |
| `LD_LIBRARY_PATH` | (must include simulator) | Simulator libraries |
| `NODE_TLS_REJECT_UNAUTHORIZED` | `0` (for opencode) | Bypass TLS issues |
| `PYASC_PERF_IMAGE` | `ghcr.io/<owner>/pyasc-sim-perf:py3.11` | docker_full perf image (perf-gate) |
| `OPS_MATH_HOME` / `OPS_NN_HOME` | `/opt/ops-math` / `/opt/ops-nn` (in perf image) | Canonical AscendC reference repos |

## Exit codes

- `0` -- all checks passed
- `1` -- one or more checks failed
- `2` -- environment prerequisites missing (e.g., simulator not available for merge tier)
