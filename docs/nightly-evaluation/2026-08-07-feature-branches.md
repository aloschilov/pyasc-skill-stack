# Nightly A/B evaluation — feature branches (2026-08-07)

Triggered the `tier=nightly` workflow_dispatch on the feature branches and
compared against main's last nightly baseline. **UPDATE (run concluded):
ci-nightly-stabilization nightly finished in ~8h wall — 10/12 jobs PASS
vs main's 24h28m cancel. PR #3 merged to main (`abaeba6`). Two residual
failures diagnosed below (local-stability `on` 8h timeout; skills-value-report
commit failed after a mid-run branch deletion).**

## Final results — ci-nightly-stabilization run 31195012048 (concluded ~8h)

| Job | Result | Time |
|---|---|---|
| `pr-gate` | ✓ | 20s |
| `merge-gate` (4 shards, golden-verify) | ✓ | 7-12m each |
| `nightly-gate` (P2) | ✓ | 42m |
| `nightly-gate` (P3) | ✓ | 3h14m |
| `local-stability-gate` (off) | ✓ | 2h4m |
| `cloud-dashscope-gate` (glm-5.1) | ✓ | 4h3m |
| `cloud-dashscope-gate` (glm-5.2) | ✓ | 5h14m |
| `cloud-dashscope-gate` (qwen3.7-max) | ✓ | 6h21m |
| `perf-gate` | ✓ | 7h19m |
| `local-stability-gate` (on) | ✗ | 8h0m24s — **hit `timeout-minutes: 480`** |
| `skills-value-report` | ✗ | "Commit evidence" exit 1 (partial-nightly + mid-run branch deletion) |

Run conclusion: `cancelled` (the local-stability `on` per-job 8h cap fired).

### A/B verdict — ci-nightly-stabilization vs main
- **Decisive improvement.** Main's last nightly (`30527114992`) cancelled at
  **24h28m** with only nightly-gate P2/P3 succeeding (local-stability
  cancelled). ci-nightly-stabilization finished in **~8h wall** with **10/12
  jobs PASS** — including the cloud-dashscope legs + perf-gate that main
  never reached. The container-leak guard + 2-runner fan-out removed the
  resource-exhaustion slowdown that previously overran 24h.
- **PR #3 merged** (`abaeba6`) — it was good to go (pr-gate ✓; merge-gate ✓
  via the nightly dispatch; mergeable/clean).
- The run did **not** hit the 24h queued-job limit — the remaining cancel is
  the **per-job 8h cap on `local-stability-gate (on)`**, a different failure
  mode than main's. Fix: raise `local-stability-gate` `timeout-minutes` 480→
  600 (10h) — the run wall would still be ~10h, well under 24h. (The `on` leg
  is ~4× slower than `off` because skills-ON generation is heavier on the
  local qwen3-coder-30b model.)

### Residual failures (diagnosed)
1. `local-stability-gate (on)` — `exceeded the maximum execution time of
   8h0m0s` (`timeout-minutes: 480` in ci.yml). The skills-ON leg needs >8h.
   Not a regression; a tuning cap that's too tight for this leg. **Recommended
   fix:** bump to 600.
2. `skills-value-report` "Commit evidence, summary, and capabilities" exit 1.
   The partial-nightly guard correctly fired ("Committing skills-value-summary
   only; per-cell evidence + capabilities.yaml NOT updated"). The commit/push
   then failed — because **PR #3 was merged with `--delete-branch` while the
   nightly was still running**, so the evidence `git push` targeted the
   just-deleted `ci-nightly-stabilization` ref. Not a code bug; an operational
   timing artifact. On a main-branch nightly this won't recur.

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

---

## Confirmation run — main nightly 2026-08-11 (run `31365678492`)

After merging PR #3 (`abaeba6`) + the `local-stability-gate` timeout bump
480→600 (`a6a261c`), triggered `tier=nightly` on `main` to confirm the fixes
hold on the default branch. **Result: SUCCESS — all 13 jobs ✓, ~18h wall,
full evidence committed (`f3d4ffd`, `partial_run: false`).**

| Job | Result |
|---|---|
| `pr-gate` | ✓ |
| `merge-gate` (4 shards, golden-verify) | ✓ |
| `nightly-gate` (P2, P3) | ✓ |
| `local-stability-gate` (off) | ✓ |
| `local-stability-gate` (on) | ✓ (the leg that hit the 8h cap before; 10h cap cleared it) |
| `cloud-dashscope-gate` (glm-5.1, glm-5.2, qwen3.7-max) | ✓ all 3 |
| `perf-gate` | ✓ |
| `skills-value-report` | ✓ (evidence committed to main; rebase-before-push worked) |

**No issues to fix.** The stabilization fixes are confirmed on `main`:
- container-leak guard + 2-runner fan-out → no 24h overrun (main's previous
  nightly cancelled at 24h28m; this one finished in ~18h).
- `local-stability-gate` 480→600 → the `on` leg completes (was the lone
  per-job timeout failure on the ci-nightly-stabilization run).
- rebase-before-push → `skills-value-report` evidence commit succeeded on main
  (the ci-nightly-stabilization run's commit failed only because PR #3 was
  merged with `--delete-branch` mid-run).

Net: main now has a green nightly + fresh full evidence. Remaining open work
is the API/image transition (PRs #4/#5 blocked on the pyasc-sim bump to
`4d1db41d` + the `batch_norm_v3_f32` NaN debug), not the harness.

