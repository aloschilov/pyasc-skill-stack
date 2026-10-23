# Changelog

Releases are git tags. Tagging `v*` builds the archives and publishes them —
see [.github/workflows/release.yml](.github/workflows/release.yml). There is no
`release/<date>/` directory; a dated copy of a tree drifts from the tree.

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
  vendored from upstream and pinned by `reference/asctile/PIN.json`. 18 modules,
  891 parametrised cases. This is the correctness reference.
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
