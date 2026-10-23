# CI gates

Two gates, plus a release workflow. Both live in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

| Gate | Trigger | Duration | What it checks |
|---|---|---|---|
| `pr-gate` | every push and PR | seconds | L1 unit tests and static checks |
| `merge-gate` | push to `main`, manual dispatch | ~30 min | compile-check of the AscTile target suite on the camodel |
| release | `v*` tag | minutes | `pr-gate` plus archive build and install verification |

## `pr-gate`

```bash
bash tests/ci-gate.sh --tier pr      # what CI runs
bash tests/run-tests.sh --fast       # the same thing, directly
```

Deterministic structure and content checks over `skills/` and `agents/`, the
runtime-backend dispatch logic, and an offline drift check comparing the
vendored target suite against `reference/asctile/PIN.json`. No Docker, no
CANN, no network, no provider calls.

This is the gate a release tag must pass.

## `merge-gate`

Pulls the pinned `pyasc-sim` image from ghcr.io and compile-checks every
module in `reference/asctile/target/` with
`tests/tools/run_target_test.py --backend Model --platform Ascend950PR_9599`.

Compile-only is deliberate. Each module carries roughly 100 parametrised
cases and the camodel is slow, so the blocking gate verifies that every case
generates and compiles; full numerical execution is a manual or nightly run.
See [review-order.md](review-order.md) for running it by hand.

Runs natively on the self-hosted arm64 runner (no x86 emulation). The image
tag is version-coupled to an immutable pyasc commit — never point it at a
mutable tag, because a moving runtime silently invalidates every result
below it.

`test_add_inplace.py` is skipped: 4 of its 20 cases exceed the 192 KB UB
budget on large shapes even under `--compile-only`. Pending a shape trim or a
representative-case selection block.

## Release

Tagging `v*` triggers [`release.yml`](../.github/workflows/release.yml): run
the fast suite, build `tar.gz` and `zip` of exactly `skills/`, `install.sh`,
`README.md` and `CHANGELOG.md`, assert the unpacked archive contains those
four entries and nothing else, assert `install.sh --target` links every skill
from the unpacked archive, then publish with notes from the tag.

## What is not here any more

The campaign-era `perf-gate`, `nightly-gate`, `local-stability-gate`,
`cloud-dashscope-gate` and `skills-value-report` jobs read from `evidence/`
and scored model-written kernels in `golden/`. Both were removed from `main`,
so the jobs were retired with them; `pages.yml` published a dashboard from a
directory that was never tracked. All of it is on the
`archive/campaign-2026` branch:

```bash
git show archive/campaign-2026:.github/workflows/ci.yml
```

Restoring any of it means first restoring an evidence source for it to read,
and that source should be CANNBench rather than checked-in JSON — see
[evidence-sources.md](evidence-sources.md).

## Local test tiers

```bash
bash tests/run-tests.sh --fast    # L1 only (what CI gates on)
bash tests/run-tests.sh --all     # adds L2 behavior and L3 integration
```

`tests/behavior/` makes real `opencode` provider calls, so `--all` costs
money and time and is not run in CI.
