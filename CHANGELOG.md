# Changelog

Releases are git tags. Tagging `v*` builds the archives and publishes them —
see [.github/workflows/release.yml](.github/workflows/release.yml). There is no
`release/<date>/` directory; a dated copy of a tree drifts from the tree.

## Unreleased

Close-generation-loop audit fixes (local qualification and PR CI passed; see
[verification status](docs/verification-status.md) for the tested revision):

- Harness-owned runpy execution, counted launches and an explicit Docker backend.
- Numerical verification requires an attributable failed comparison in the
  suppressed-launch control, not merely a failing process. Timeout, missing
  reports and unrelated exceptions leave the result at `executed`.
- The regression matrix includes original run-2 as an expected compilation
  failure; run-1/run-3 and the fixed tail-bearing template are positive controls.
- Strict fresh generation requires static acceptance, codegen and numerical
  execution. Runtime failure or unavailability cannot become a passing cycle.
- Multi-shape scoring recognizes a direct verification loop over tuple sizes
  as well as list sizes; thresholds and historical evidence remain unchanged.
- Run manifests, logs and generated sources live outside the checkout. CI
  uploads artifacts; accepted qualifications need a retained archive beyond CI
  artifact expiry. Historical evidence is unchanged.


## v0.2.0

The release that ran the gate the first release only described.

### The generation gate runs

v0.1.0 shipped with the agent-generation path recorded as `blocked` and
therefore untested. It has now been run, and the numbers are in
[`docs/verification-status.md`](docs/verification-status.md) with the
artifacts in `docs/evidence/generation-gate-20261028/`: three runs, a kernel
and a design document every time, 7/7 static checks every time, scores of
12, 12 and 11 against a threshold of 12.

Reported as `passed`, *marginally*. Two of three cleared the bar by exactly
zero margin, and the only dimension separating a pass from a failure was
whether the agent happened to verify one tensor shape or two. No generated
kernel has been **executed** — `asctile` is not importable on the host and
`run_and_verify.py` has no docker path — so this is a statement about the
source the agent writes and nothing further.

Running the gate for the first time found two defects that only running it
could find:

- `verify_kernel.py` recognized `@asc.jit` and `@asc2.jit` but not
  `@asctile.jit`, the spelling the skills and the prompt both mandate. Every
  correctly written kernel failed `jit_decorator` and cascaded into three
  further "skipped (no JIT function)" failures. It now accepts all three plus
  dotted forms such as `@asc.experimental.asctile.jit`.
- The prompt opened by asking the agent to check its environment. Finding no
  importable `asctile`, it spent 22 steps and 26 tool calls excavating
  unrelated checkouts under `/home` and `/tmp`, wrote nothing, and exhausted
  900 s. The prompt now states the environment as fact; the same run then
  completed in 129 s.

The 8192-token output ceiling diagnosed in v0.1.0 was real but was neither
the only obstacle nor the first one encountered.

### What `generation (executed)` is actually waiting on

`not-run` previously read as though executing a generated kernel needed
machinery nobody had built. It does not. `run_docker_verify()` already runs
`run_and_verify.py --mode simulator` inside the pinned image; the integration
test just does not call it.

Run by hand against the pinned dev overlay, the gap turns out to be four
lines. As generated, every kernel omits `config.set_platform()`, so it asks
for a real device and dies on `SetDeviceWrapper returned 107001` — with or
without the `-r`/`-v` arguments the harness passes, which none of them parse,
so the failure is byte-identical either way. Add the launch contract and
nothing else, and the agent's own vector add **executes on the simulator and
matches `torch`**.

The kernel logic was never the problem. `docs/verification-status.md` carries
the three-row comparison.

This also turned up a defect in the stack itself:
`skills/pyasc-codegen-workflow/templates/kernel-template.py` — the file that
teaches the launch contract, and which the agent demonstrably reads — does
not compile against the pinned runtime. Codegen rejects the
`if valid > tile_size:` guard inside its `asctile.range` loop with
`UnsupportedSyntaxError`. The generated kernels sidestep it with
`min(size - offset, tile_size)`. Nothing in the test suite compiles the
template, which is why this went unseen. Recorded, not fixed.

### One source of truth for opencode profiles

`tests/tools/opencode_profile.py` is new, and both generation entry points go
through it. `tests/integration/test-kernel-generation.sh` previously wrote no
`opencode.json` at all and silently inherited the host's global config —
including the 8192 cap that the same repository documented as the blocker. A
harness that inherits an unknown ceiling is not measuring what it claims to.

- The collector's default profile moved from `cloud-default` to
  `cloud-glm-5.3`. The old default named a DashScope model behind a key no
  runner here holds, so as a default it could not complete a run.
- A missing credential now fails up front and names the variable, instead of
  surfacing minutes later as a provider-side 401.
- `--model-profile` is validated against the profiles on disk. A typo used to
  fall back to a config naming no model, which completed and attributed the
  evidence to whatever the host's default was.
- The repo root is granted `external_directory` access at render time,
  replacing a hardcoded `~/workspace/pyasc-skill-stack/*` that granted
  nothing on any other machine. Without it a deny-by-default profile also
  denies the symlinked skill stack, and the agent runs with no skills while
  the harness reports `skills_mode=on`.
- `collect_generative_evidence.py` resolves the **dev overlay** image, as CI
  and `verify_pyasc_image.sh` already did. It was the last caller still
  falling back to the minimal image, which has no torch, so runtime
  verification died on `import torch` and reported it as a kernel defect.

### `blocked` is no longer spelled `skipped`

`tests/run-tests.sh` recorded a timed-out test as a skip. "Skipped" means we
chose not to run it; a timeout means it ran and we learned nothing. There is
now a `blocked` bucket, reported separately, emitted as `"status": "blocked"`
in `--output json`, and exiting 2 — distinct from 1 for a real failure.

The timeout that triggered this was itself a collision: the per-file budget
and the agent timeout inside the generation test were both 300 s, and the
outer timer starts first, so the agent's own timeout handling and every
verification phase after it were unreachable. Integration tests now get their
own budget (`PYASC_INTEGRATION_TIMEOUT`, default 1800 s) above the agent's
(`PYASC_AGENT_TIMEOUT`, default 900 s).

### Smaller corrections

- The generation test reported scores as `/ 10` against a threshold of `8.5`
  while `score_kernel.py` used 16 and 12. It now reads both from the scorer's
  own JSON, and distinguishes a hard-gate rejection from a low score.
- `$PYTHON run_and_verify.py ... || true` set `$?` to 0 unconditionally, so
  every run — including a crash — reported "Runtime verification passed".
- An absent `opencode` CLI exited 1, reporting a missing optional dependency
  as a product defect. It now skips.
- The "no prompt" error named `capabilities.yaml`, a file pruned at v0.1.0
  and listed in `.gitignore`. It now says so and gives the restore command.
- `PYASC_KEEP_PROJECT=1` retains the generated tree. A generation is not
  reproducible, so an artifact discarded on exit cannot be recovered by
  re-running.

## v0.1.0

First production release. The repository had become a record of a
kernel-generation evaluation campaign — 3183 tracked files, most of them
per-run evidence — with the actual product, ten skills, making up 1.4% of it.
This release is that product and the things it needs to be trusted.

### The skill stack

Ten skills, installed by `install.sh`:

| Skill | Covers |
|---|---|
| `pyasc-api-patterns` | AscTile v2 API and kernel patterns |
| `pyasc-syntax-constraints` | syntax, typing, `ConstExpr`, loop and kernel constraints |
| `pyasc-codegen-workflow` | end-to-end design, implementation, review, provenance |
| `pyasc-build-run-verify` | building and running on QEMU gates, camodel, NPU |
| `pyasc-code-review` | contract, correctness, memory-safety and evidence review |
| `pyasc-target-operator` | authoring and extending the upstream target suite |
| `pyasc-cannbench-kernel` | CANNBench operators and performance configuration |
| `pyasc-docs-search` | local-first index of APIs, tests, examples |
| `pyasc-env-check` | runtime, toolkit, CANN and provenance verification |
| `pyasc-task-focus` | attention management across long kernel tasks |

### The runtime is upstream's, pinned by digest

This repository stopped building simulator images and started consuming the
ones upstream already builds, tests and publishes.

- `reference/images/ENV.json` pins one cell of the matrix published by
  `compiler-team-ru/pyasc-images`: CANN 9.2.0, Ubuntu 22.04, CPython 3.11,
  recorded as a multi-arch **digest**, never a tag.
- CI runs the *development overlay*, `ghcr.io/compiler-team-ru/pyasc-sim-dev`
  — the same image with CPU torch and pytest already installed, which the
  target suite needs and the minimal image deliberately omits. The repository
  used to pip-install both onto every runner; it no longer installs either.
  `ENV.json` records the overlay's digest alongside the exact `torch` and
  `pytest` versions it ships, and `verify_pyasc_image.sh` asserts them.
- The overlay is published per base, so `check-image-pin.sh` asserts that
  `.dev_image.base_report` equals `.matrix.release_tag`. The vendored suite,
  the wheel inside the image and the runtime therefore always derive from one
  upstream commit. `sync-image-pin.sh` refuses to pin a base that has no
  overlay yet: the overlay line follows new bases on a delay, and a pin CI
  cannot run is worse than no new pin.
- `sync-image-pin.sh` resolves the newest `matrix-*` release by tag prefix
  rather than through GitHub's `latest` pointer, which now 404s because the
  dev-overlay and dev-test releases share the repository. The pin records a
  per-release asset URL, and `check-image-pin.sh` fails if it ever points at
  `latest` again.
- Removed the repository's own image machinery: four Dockerfiles, two build
  scripts, the `build-sim-image` workflow, and the 26-file
  `docker/pyasc-overlay/` patched-source tree that predated the published
  wheel. All on `archive/campaign-2026`. The upstream image is 7 GB against 24.
- CI no longer logs in to ghcr.io. The packages allow anonymous pulls, and
  `GITHUB_TOKEN` cannot read another org's packages anyway — a stale credential
  turns a public pull into `denied`, so the pull forces an empty docker config.

### The target-suite pin is resolved

- `PIN.json` points at `github.com/compiler-team-ru/pyasc` — reachable
  anonymously, unlike the gitcode remote — at `d0f64a63`, the same commit the
  pinned image was built from. 24 modules, up from 18.
- The suite now has subpackages (`matmul_v3/`, `kimi_k3/`), and
  `test_matmul_v3*.py` moved into one of them while `test_add_inplace.py`
  disappeared upstream — so the merge-gate's hardcoded skip for it is gone.
  Sync, drift check and compile loop all recurse instead of globbing
  `target/*.py`.
- `sync-target-suite.sh` no longer gates on a hardcoded "rename commit": that
  SHA exists only on gitcode, so against any other mirror it reported a valid
  ref as predating the rename. The guard is now by content, which is how the
  rest of the script already worked.
- `conftest.py` was rebuilt from upstream's. The local copy had silently
  dropped the `asctile_xfail` marker and its `apply_xfail` fixture — which
  `test_rms_norm.py` needs for five UB-overflow cases — and had stopped
  registering `--backend` and `--platform` on the theory that the runtime
  provides them. It does not: `pyasc-dev` declares no entry points and ships
  no pytest plugin.

### CI actually runs

Worth stating because for a while it did not. Every job targets
`[self-hosted, arm64]`, and the only registered runner was labelled
`cannbench-vm`. Nothing matched, so runs sat queued indefinitely — and a
workflow that cannot be scheduled looks exactly like one that was never
triggered. An arm64 runner now serves it (`infra/self-hosted-runner/`), and
both gates report green.

Bringing it up surfaced a failure mode that reads as the wrong problem: the
prebuilt runner image carried v2.335.1, which GitHub has retired. A retired
runner connects, registers, prints "Listening for Jobs", is told it cannot
receive messages, deregisters, and loops — so it never shows online and the
symptom looks like a bad credential. Auto-update cannot fix it, because the
update travels over the channel the runner is barred from. The Dockerfile now
records this, with the build-and-ship recipe.

### Gates report `blocked` separately from `failed`

All 24 modules compile-check on the CI runner; 0 fail. Two harness defects were
fixed on the way, both of which made a module look broken when nothing had been
checked at all: a `-k` selector that assumed one of the two upstream id
conventions, and whole-module runs that exhausted memory because it accumulates
across compiles in one interpreter (now batched, 24 cases per process).

`test_kl_div.py` is killed at exit 137 on a 39 GB development box and passes on
the 128 GB CI runner — same digest, same commit, more memory. That differential
is what identifies the cause; the exit code never could, since a timeout or a
container limit produces the same 137. It is a host-capacity limit, not a fact
about the kernel, and the gate says so rather than failing the job.

### The tree is portable

Four tools defaulted external checkouts to paths inside one developer's home
directory, which on any other machine fails with a `FileNotFoundError` naming a
stranger's account — a worse failure than not starting. They now resolve
through `external_root()`, which says what is missing and what to set.
`tests/unit/tools/test-portability.sh` keeps that true and runs in `--fast`.

### Added

- `install.sh --target DIR` installs into an explicit directory instead of the
  hardcoded `~/.cursor/skills` and `~/.config/opencode/skills`.
- `reference/asctile/target/` — the Huawei-authored AscTile target suite,
  vendored from upstream and pinned by `reference/asctile/PIN.json`. This is
  the correctness reference.
- `scripts/sync-target-suite.sh` re-vendors the suite. Upstream moves it
  between versions (`python/test/asc2/target` → `python/test/asctile/target` →
  `experimental/asctile/python/test/target`), so it finds the suite by content
  — modules using both `asctile.jit` and `asctile.global_tensor` — rather than
  by a path that goes stale. It defaults to the installed runtime's own source
  commit, read from wheel metadata.
- `scripts/check-target-pin.sh` and `scripts/check-image-pin.sh` detect drift
  offline, both wired into `tests/run-tests.sh --fast`.
- `docs/evidence-sources.md` — what counts as evidence: the target suite for
  correctness, CANNBench for performance, TTK for the CANN baseline, and the
  rule that **no model-written kernel is a reference**. It also records
  upstream's five-state qualification mapping, including the rule that an
  interrupted cell stays `blocked` rather than being inferred into `failed` or
  `not-run`.
- `docs/upstream-image-requests.md` — the standing ask to the image-build side,
  version-controlled so the questions and their answers stay on the record.
- `skills/pyasc-cannbench-kernel/references/ttk.md` — the TTK manifest's 42
  columns, how a CANNBench case maps onto a row, and why the manifest is not
  reproducible from a clean clone yet.
- `.github/workflows/release.yml` and this changelog.

### Changed

- `.github/workflows/ci.yml`: 1085 → ~140 lines. `pr-gate` runs static checks
  and L1 units; `merge-gate` compile-checks the target suite instead of the
  model-written goldens.
- `tests/ci-gate.sh` keeps only the `pr` tier.
- `reference/asctile/conftest.py` resolves the AscTile runtime across both
  layouts (`asc.experimental.asctile` from the published wheel, top-level
  `asctile` from a legacy source build) and fails with one readable diagnostic
  when neither is usable.
- `skills/pyasc-target-operator` points at the local pinned suite and
  describes content-based discovery rather than a hardcoded upstream path.
- The README advertised three wheels and glibc 2.35. There are eight, and the
  wheels are `manylinux_2_28` — glibc 2.28. The table is gone rather than
  corrected: the matrix moved twice in three days, so the README now shows how
  to read the live `wheel-manifest.json` instead.
- `docker/opencode-profiles/` moved to `integrations/opencode/profiles/`.

### Removed

The campaign scoreboard: `evidence/` (2800+ per-run JSONs), `capabilities.yaml`,
`golden/kernels`, `golden/tutorials`, `golden/docs`, `teams/`, 18 dated reports
in `docs/`, and the validators that asserted over them. Tracked files went from
3183 to 548.

Also retired: the GitHub Pages dashboard and the `dashboard/` tree it was built
from. The site published campaign evidence that this release prunes, so keeping
it live would have contradicted the rest of the repository.

Nine superseded branches were pruned — seven `codex/*`, plus
`pyask-skill-stack` and `productionize/mvp-v0.1.0` — and the two campaign-era
pull requests that rode on them were closed as superseded.

Nothing was destroyed. Tracked state is on `archive/campaign-2026` (tag
`archive-campaign-2026`), uncommitted worktree edits on
`archive/pre-cleanup-worktree-20261023`, and never-committed campaign WIP on
`archive/untracked-campaign-20261023`. All three are on origin, and
`.gitignore` carries the recovery command.

The nine pruned branch tips were reachable from none of those, so deleting
them would have destroyed content. They are preserved under the
`archive-branches-20261028` tag: one empty-tree commit with nine parents, so a
single ref keeps all nine histories reachable. Its message lists every branch,
SHA and date. Restore one with `git checkout -b <name> <sha>`.

### Known limits

Stated rather than papered over:

- The target suite is **compile-checked**, not executed. `--compile-only` mocks
  kernel launch and the numerical assertions, so a green sweep means 24 modules
  build — not that any of them computes the right answer. A full camodel run is
  the first honest correctness number this stack will have, and it is a v0.2.0
  gate.
- Upstream's own full-suite qualification for the pinned cell is `incomplete`:
  2824 passed and 0 failed so far, but every module under `target/` — the
  subtree this repository vendors and gates on — is still `not-run`. There is
  real numerical evidence upstream, and none of it covers our gate yet.
- The agent-generation path is `blocked` and therefore untested. See
  `docs/verification-status.md`; the cap is declared outside this repository.
- CANNBench task definitions are tracked, but the scorer that turns them into
  ratios was campaign-era and was archived. Reproducible scoring is v0.2.0.
- The TTK manifest generator read a path outside the repository, so the
  baseline cannot be regenerated here. Treat TTK-backed ratios as `blocked`.
- Hardware and performance are `not-run`. Everything above is a simulator
  result.
