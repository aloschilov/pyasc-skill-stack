# Changelog

Releases are git tags. Tagging `v*` builds the archives and publishes them —
see [.github/workflows/release.yml](.github/workflows/release.yml). There is no
`release/<date>/` directory; a dated copy of a tree drifts from the tree.

## Unreleased

Stop building a simulator image, and start consuming the one upstream already
builds, tests and publishes.

### Runtime is pinned by digest, from upstream

- `reference/images/ENV.json` pins one cell of the matrix published by
  `compiler-team-ru/pyasc-images`: CANN 9.2.0, Ubuntu 22.04, CPython 3.11,
  recorded as a multi-arch **digest**, never a tag. `scripts/sync-image-pin.sh`
  moves the pin; `scripts/check-image-pin.sh` runs in `--fast` and fails if the
  pin, `PYASC_SIM_IMAGE` in CI, and `reference/asctile/PIN.json` stop agreeing.
- Removed the repository's own image machinery: four Dockerfiles, two build
  scripts, the `build-sim-image` workflow, and the 26-file `docker/pyasc-overlay/`
  patched-source tree that predated the published wheel. All on
  `archive/campaign-2026`. The upstream image is 7 GB against 24 GB.
- CI no longer logs in to ghcr.io. The packages allow anonymous pulls, and
  `GITHUB_TOKEN` cannot read another org's packages anyway — a stale credential
  turns a public pull into `denied`, so the pull forces an empty docker config.
- `tests/tools/verify_pyasc_image.sh` now asserts the pulled repo digest
  against the pin and that `asc.experimental.asctile.jit` imports. It no longer
  runs `git -C /opt/pyasc rev-parse`, which cannot work: the image installs the
  published wheel and carries no source checkout.

### The target-suite pin is resolved

- `PIN.json` no longer says `"commit": "unresolved"`. It points at
  `github.com/compiler-team-ru/pyasc` — reachable anonymously, unlike the
  gitcode remote — at `d0f64a63`, the same commit the pinned image was built
  from. 24 modules, 1155 cases, up from 18 and 891.
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

### Gates report `blocked` separately from `failed`

23 of 24 modules compile-check against the pinned image; 0 fail. Two harness
defects were fixed on the way, both of which made a module look broken when
nothing had been checked at all: a `-k` selector that assumed one of the two
upstream id conventions, and whole-module runs that OOMed because memory
accumulates across compiles in one interpreter (now batched, 24 cases per
process). The one remaining blocked module, `test_kl_div.py`, allocates a
7.4 GiB input tensor; that is a property of the host, not of the kernel, and
the gate says so rather than failing the job. See `docs/verification-status.md`.

### Documentation corrected against reality

- The README advertised three wheels and glibc 2.35. There are eight, and the
  wheels are `manylinux_2_28` — glibc 2.28. The table is gone rather than
  corrected: the matrix moved twice in three days, so the README now shows how
  to read the live `wheel-manifest.json` instead.
- Recorded that the image ships neither `torch` nor `pytest`, and that its own
  verification scope is "vector-add compile and simulator smoke, 128 float32
  elements" — not correctness qualification.
- `docker/opencode-profiles/` moved to `integrations/opencode/profiles/`.

### Pinned against two kinds of drift

Upstream now publishes a development overlay line,
`ghcr.io/compiler-team-ru/pyasc-sim-dev`, which bakes CPU Torch and pytest into
the simulator image. It is not adopted here yet — the overlays are built on
`1.1.1.dev20261023173239` while this repository pins the newer
`1.1.1.dev20261024081924`, and running a suite from one commit against a
runtime from another is precisely what `check-image-pin.sh` exists to stop. The
ask for a matching overlay, and four other questions for the image-build side,
are recorded in `docs/upstream-image-requests.md`.

What changed in the meantime:

- The CI torch/pytest overlay was installed unversioned, so every run resolved
  whatever was newest and two runs a month apart compiled against different
  stacks. It is now `--no-deps` against an explicit version list, keyed on a
  stamp file so a cached overlay cannot outlive a version bump. The versions
  are upstream's own qualified set — `torch==2.7.1+cpu`, `pytest==8.4.2` — so
  adopting `pyasc-sim-dev` later changes where the packages come from and
  nothing about which packages they are.
- Compile-loop containers now use upstream's documented run environment
  (`HOME`, `PYTHONDONTWRITEBYTECODE`, `PYASC_CACHE_DIR`, `PYASC_DUMP_PATH`),
  which matters because `/workspace` is mounted read-only.
- `sync-image-pin.sh` resolves the newest `matrix-*` release instead of
  GitHub's `latest` pointer. The dev-overlay and dev-test releases share the
  repository, so `releases/latest/download/matrix.json` now 404s; the pin
  records a per-release asset URL, and `check-image-pin.sh` fails if it ever
  points at `latest` again.
- `docs/evidence-sources.md` records what the dev line certifies. Upstream's
  own wording: `published` means environment-qualified, not that the full suite
  passed. Suite status for the pinned cell is `pending`, `executed: 0` — which
  in this repository's vocabulary is `not-run`.

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

### Added

- `install.sh --target DIR` installs into an explicit directory instead of the
  hardcoded `~/.cursor/skills` and `~/.config/opencode/skills`.
- `reference/asctile/target/` — the Huawei-authored AscTile target suite,
  vendored from upstream and pinned by `reference/asctile/PIN.json`. 24 modules,
  1155 parametrised cases. This is the correctness reference.
- `scripts/sync-target-suite.sh` re-vendors the suite. Upstream moves it
  between versions (`python/test/asc2/target` → `python/test/asctile/target` →
  `experimental/asctile/python/test/target`), so it finds the suite by content
  — modules using both `asctile.jit` and `asctile.global_tensor` — rather than
  by a path that goes stale. It defaults to the installed runtime's own source
  commit, read from wheel metadata.
- `scripts/check-target-pin.sh` detects drift between the vendored snapshot and
  `PIN.json` offline, wired into `tests/run-tests.sh --fast`.
- `docs/evidence-sources.md` — what counts as evidence: the target suite for
  correctness, CANNBench for performance, TTK for the CANN baseline, and the
  rule that **no model-written kernel is a reference**.
- `skills/pyasc-cannbench-kernel/references/ttk.md` — the TTK manifest's 42
  columns, how a CANNBench case maps onto a row, and why the manifest is not
  reproducible from a clean clone yet.
- `.github/workflows/release.yml` and this changelog.

### Changed

- `.github/workflows/ci.yml`: 1085 → 120 lines. `pr-gate` runs static checks
  and L1 units; `merge-gate` compile-checks the target suite instead of the
  model-written goldens.
- `tests/ci-gate.sh` keeps only the `pr` tier.
- `reference/asctile/conftest.py` resolves the AscTile runtime across both
  layouts (`asc.experimental.asctile` from the published wheel, top-level
  `asctile` from a legacy source build) and fails with one readable diagnostic
  when neither is usable.
- `skills/pyasc-target-operator` points at the local pinned suite and
  describes content-based discovery rather than a hardcoded upstream path.

### Removed

The campaign scoreboard: `evidence/` (2800+ per-run JSONs), `capabilities.yaml`,
`golden/kernels`, `golden/tutorials`, `golden/docs`, `teams/`, 18 dated reports
in `docs/`, and the validators that asserted over them. Tracked files went from
3183 to roughly 290.

Nothing was destroyed. Tracked state is on `archive/campaign-2026` (tag
`archive-campaign-2026`), uncommitted worktree edits on
`archive/pre-cleanup-worktree-20261023`, and never-committed campaign WIP on
`archive/untracked-campaign-20261023`. All three are on origin, and
`.gitignore` carries the recovery command.

### Known limits

Stated rather than papered over:

- The target suite is **compile-checked**, not executed. A full camodel run is
  the first honest correctness number this stack will have, and it is a v0.2.0
  gate.
- `PIN.json` records the commit as `unresolved`. Running the sync for real
  needs network access to `gitcode.com`; the pin is honest about not knowing
  rather than inventing a commit.
- CANNBench task definitions are tracked, but the scorer that turns them into
  ratios was campaign-era and was archived. Reproducible scoring is v0.2.0.
- The TTK manifest generator read a path outside the repository, so the
  baseline cannot be regenerated here. Treat TTK-backed ratios as `blocked`.
