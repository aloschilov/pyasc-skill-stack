# Nightly A/B evaluation — feature branches (2026-08-07)

Triggered the `tier=nightly` workflow_dispatch on the feature branches and
compared against main's last nightly baseline. **Status at write-up time:
ci-nightly-stabilization nightly still in the generative-evidence phase
(~18-24h total); this file captures the partial results + the diagnosis and
will be updated when the run concludes.**

## Branches / PRs

| Branch | HEAD | PR | State |
|---|---|---|---|
| `main` (baseline) | `886275f` | — | last nightly **cancelled** at 24h28m (run `30527114992`, 8d ago) |
| `ci-nightly-stabilization` | `0ef15e7` | [#3](https://github.com/aloschilov/pyasc-skill-stack/pull/3) | **ready** — old-API goldens, image-compatible |
| `m3-inplace-atomic` | `51f1404` | [#4](https://github.com/aloschilov/pyasc-skill-stack/pull/4) | **draft** — blocked on image bump (atomic_add golden uses `global_tensor`) |
| `guidance-gm-barrier` | `ee2df79` | [#5](https://github.com/aloschilov/pyasc-skill-stack/pull/5) | **draft** — blocked on image bump (goldens use the 4d1db41d API) |

## Baseline (main, last recorded nightly)

- Run `30527114992`: `status=completed`, `conclusion=cancelled` (24h28m — hit
  GitHub's 24h queued-job limit).
- Recorded evidence `evidence/skills-value-summary.json` (2026-07-31):
  `partial_run=true`; `nightly-gate (P2)` + `nightly-gate (P3)` = **success**;
  `local-stability-gate (off)` = **cancelled**; cloud legs trimmed out.

## Feature run — `ci-nightly-stabilization` (run `31195012048`)

Triggered `workflow_dispatch tier=nightly` on `ci-nightly-stabilization`.
Snapshot at T+51m:

| Job | Status | Notes |
|---|---|---|
| `pr-gate` | ✓ 20s | static checks + capabilities validation |
| `merge-gate` (4 shards) | ✓ 7-12m each | **golden-verify PASS** on the current image (old-API goldens) |
| `nightly-gate` (P2, P3) | running | generative evidence, cloud-default |
| `local-stability-gate` (off, on) | running | local qwen3-coder-30b |
| `cloud-dashscope-gate` (glm-5.1, glm-5.2, qwen3.7-max) | running | cloud generative |
| `perf-gate` | running | perf-vs-AscendC |

### A/B verdict (ci-nightly-stabilization vs main)

- **Golden-verify (merge-gate): identical** — expected, since this branch only
  touches `ci.yml` + docs, not the goldens. Both pass on the current image.
- **Generative evidence / harness completion: verdict pending (~18-24h).** The
  hypothesis under test: the stabilization fixes (container-leak guard,
  rebase-before-push, 2-runner fan-out, host-sleep note) prevent the
  resource-exhaustion / push-failure modes that left main's nightly partial
  (`local-stability-gate` cancelled). Watch whether
  `local-stability-gate` completes this time (it cancelled on main).
- The fixes do **not** shrink the generative matrix, so if the run still cancels
  at ~24h, the root cause is matrix size vs the 24h queued-job budget (not
  container leaks) — see "What went wrong" below.

## Per-branch verdict

### `ci-nightly-stabilization` (PR #3) — ready
- merge-gate ✓ (golden-verify). Generative-evidence running.
- All fixes are image-compatible (old API). **Mergeable now**; the nightly
  result (once complete) is additive evidence, not a merge blocker.

### `guidance-gm-barrier` (PR #5) — draft, blocked
- Goldens migrated to the 4d1db41d API (`global_tensor`/`copy_in`/`copy_out`,
  `gm_barrier`, `TensorLocation`). **18/19 PASS** on the freshly-built
  `pyasc-sim:py3.11-arm64` (origin/v2 `4d1db41d`) — verified locally, NOT via
  nightly.
- Not nightly-tested: the CI multiarch `pyasc-sim:py3.11` still points at
  `7095b6fd` (parallel=, `tensor/load/store`), so these goldens would FAIL the
  merge-gate (removed API). The new arm64 leg (`py3.11-arm64`, 4d1db41d) is
  built+pushed but **not merged** into the multiarch manifest (see blockers).
- `batch_norm_v3_f32` still NaNs at `invstd=rsqrt(var+eps)` on 4d1db41d —
  needs deeper per-kernel debug (likely a 2D-axis `reduce_sum` semantics/precision
  change); the other 18 pass.

### `m3-inplace-atomic` (PR #4) — draft, blocked
- Adds `add_inplace` + `atomic_add` M3 operators + 3 reference files.
- `atomic_add` golden uses the fork target-test API (`asc2.global_tensor`/
  `copy_in`/`atomic_add`) which only exists on origin/v2 ≥ 4d1db41d — it would
  FAIL the merge-gate on the current (`7095b6fd`) image. `add_inplace` golden
  uses the old API and is fine.
- `atomic_add` reverted `confirmed`→`pending` (the prior confirmation was a
  hand-formatted ad-hoc run, not the nightly collector).

## What went wrong / diagnosis

1. **Main's nightlies cancel at ~24h (queued-job limit).** Root cause: the
   generative matrix (cells × model legs × `--max-attempts`) is too large for the
   self-hosted runner's 24h queued-job budget even after the P2/P3 trim + the
   two-runner fan-out. `local-stability-gate` was the leg most often
   starved+cancelled on main. `ci-nightly-stabilization`'s container-leak guard
   addresses a *secondary* cause (leaked `pyasc-sim` containers busy-looping →
   runner resource exhaustion → slowdown → 24h overrun), not the primary
   matrix-size cause. **If the feature run still cancels at ~24h, the matrix
   must be trimmed further (drop a cell or a leg), not just hardened.**
2. **API/image transition split.** `origin/v2` was force-rewritten to
   `4d1db41d`: `asc2.range` renamed `parallel`→`gm_barrier` (inverted) AND
   `asc2.tensor/load/store` removed (→ `global_tensor/copy_in/copy_out`),
   `TileLocation`→`TensorLocation`. The CI image is still pinned to
   `7095b6fd` (old API). So `guidance-gm-barrier` + `m3-inplace-atomic` goldens
   (new/fork API) can't be nightly-tested on the current image — they need the
   image bumped to `4d1db41d`.
3. **Multiarch manifest not merged.** The arm64 `4d1db41d` leg is built+pushed
   (`pyasc-sim:py3.11-arm64`) but the multiarch `py3.11` manifest still points
   at the old arm64+amd64 (7095b6fd). Blockers to merging: (a) `batch_norm_v3_f32`
   NaNs on 4d1db41d; (b) the amd64 leg can't be rebuilt (RTX 4090 box retired;
   needs an amd64 host or a GHA amd64 build). Until merged, the nightly gates
   (arm64 Mac) keep using the old arm64 leg.
4. **`batch_norm_v3_f32` NaN** — `invstd=rsqrt(var+eps)` goes NaN because
   `var_c = sumsq_c*inv_count - mean_c*mean_c` goes negative. The carried
   accumulator loop now has `gm_barrier=True`; the simulator appears not to
   model overlap (other norm goldens pass with overlap-on accumulators), so the
   NaN is a different 4d1db41d change (likely 2D-axis `reduce_sum` semantics or
   precision). Needs per-kernel debug.

## Recommended sequence

1. Merge **PR #3** (`ci-nightly-stabilization`) — ready, image-compatible.
2. Debug `batch_norm_v3_f32` NaN on 4d1db41d (1 kernel).
3. Rebuild the **amd64** leg from `PYASC_GIT_REV=4d1db41d` (amd64 host / GHA),
   then `docker/build-sim-image.sh --merge` to publish the multiarch manifest.
4. Once the manifest is on 4d1db41d, re-run `tier=nightly` on
   `guidance-gm-barrier` (the real A/B for the golden migration) and on
   `m3-inplace-atomic`; then promote PRs #4/#5 from draft.
5. If the nightly still cancels at ~24h after step 1, trim the generative matrix
   (the primary cancel cause) rather than only hardening the runner.

## Live monitoring

- ci-nightly-stabilization nightly: `gh run view 31195012048`
  (https://github.com/aloschilov/pyasc-skill-stack/actions/runs/31195012048)
