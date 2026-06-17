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

The GitHub Actions `merge-gate` job runs on GitHub-hosted **amd64**
(`ubuntu-latest`), sharded across 4 runners for speed, and verifies every
golden **except** `bfloat16` ones. The amd64 leg of the multiarch `pyasc-sim`
image ships a pyasc build with no bf16 IR lowering
(`asc/language/core/dtype.py` raises `Unsupported DataType name: bfloat16`),
so `bfloat16` goldens (e.g. `layer_norm_v4_bf16`) cannot compile there.

Those bf16 goldens are verified inside the **`perf-gate`** job on the
**self-hosted arm64 runner** (same host as perf measurement), where the
multiarch image resolves to its arm64 leg and bf16 is supported. The step is
**non-blocking** (`perf-gate` has `continue-on-error: true`) and runs **only on
the nightly schedule / `workflow_dispatch`**, never on push.

## Nightly gate (`--tier nightly`)

Runs in 15-30 minutes. Requires opencode CLI and CANN simulator.

1. Everything in merge gate
2. `run-tests.sh --all` -- L2 behavior tests (agent trigger correctness, premature action detection) and L3 integration tests (full agent-in-the-loop kernel generation)

Requires: opencode CLI on PATH, CANN simulator environment.

**TEMPORARY (CORC: local-polishing):** the GitHub Actions `nightly-gate` job
runs the Phase 0 protocol-axis matrix (P2/P3/P4/P6) against the Mac's local
**`qwen3-coder:30b`** Ollama model (`local-qwen3-coder-30b` profile), not the
remote DashScope `cloud-default` profile. It is **report-only** (no P6
hard-fail threshold) while skills are polished locally. The
**`cloud-dashscope-gate`** job (glm-5.1, qwen3.7-max) is temporarily disabled.

The **`local-stability-gate`** compares **`qwen3-coder:30b`** vs
**`gpt-oss:120b`** (skills on/off). Both models must be pre-pulled on the Mac's
native Ollama (`ollama pull qwen3-coder:30b`, `ollama pull gpt-oss:120b`); legs
skip cleanly when a model is missing.

### Host memory (128 GB Mac)

All arm64 jobs (`nightly-gate`, `local-stability-gate`, `perf-gate`) serialize
on the single self-hosted Mac runner and share the 128 GB host with the
~46 GB Parallels VM (the dev Linux box), the Docker Desktop VM (camodel sims),
and macOS. `gpt-oss:120b` alone is ~68 GB resident and `qwen3-coder:30b` is
~18 GB, so **two co-resident models would overrun the host and thrash swap.**
Because Ollama keeps a model warm for `keep_alive` (5 min default), each leg
runs a **Free host Ollama memory** step
([tests/tools/free_ollama_memory.py](../tests/tools/free_ollama_memory.py))
that unloads any model a prior leg left warm, bounding the peak Ollama
footprint to the single model the upcoming leg loads.

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
  SIMD fusion** panel. The same job also verifies **bf16 golden kernels** on
  arm64 (see Merge gate above).

Committed perf evidence uses **repo-relative paths** only (`golden/kernels/...`,
`evidence/perf/_build_cache/logs/...`). The PR gate runs
[tests/tools/check_evidence_paths.py](../tests/tools/check_evidence_paths.py) to
reject `/home/` and `/Users/` prefixes in perf/vf-fusion JSON.

The GitHub Actions `nightly-gate` (and local-stability matrix legs) discover
generative cells from `capabilities.yaml` via
[tests/tools/list_generative_cells.py](../tests/tools/list_generative_cells.py)
(every cell with a non-empty `prompt` on `Ascend950PR_9599`).

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
