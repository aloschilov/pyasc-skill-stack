# CI gates

Three jobs, plus a release workflow. The CI jobs live in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

| Gate | Trigger | Duration | What it checks |
|---|---|---|---|
| `pr-gate` | PR, push to `main`, manual dispatch | seconds | L1 unit tests and static checks |
| `examples-gate` | same triggers, after `pr-gate` | minutes | shipped kernels and verification contract on pinned simulator |
| `merge-gate` | push to `main`, manual dispatch | minutes; 60-minute cap | compile-check of the AscTile target suite on the camodel |
| release | `v*` tag | minutes | fast unit suite plus archive build and install verification |

## `pr-gate`

```bash
bash tests/ci-gate.sh --tier pr      # what CI runs
bash tests/run-tests.sh --fast       # unit portion only; no skill-block syntax check
```

Deterministic structure and content checks over `skills/`, the
runtime-backend dispatch logic, and an offline drift check comparing the
vendored target suite against `reference/asctile/PIN.json`. No Docker, no
CANN, no network, no provider calls.

The release workflow runs its fast-unit subset. Neither has a device or provider dependency.

## `examples-gate`

Runs `tests/integration/test-kernel-verification-contract.sh` using the
digest-pinned development image in `reference/images/ENV.json`. The matrix
covers unchanged historical kernels, the tail-bearing template, negative and
weak-positive fixtures, and complete skill listings. Expected failures must
fail for the contracted reason; real numerical passes need a suppressed-launch
control. No NPU or provider call is involved.

```bash
python3 tests/tools/verification_contract.py --results-dir "$RUN_ROOT/contract"
```

Choose a new directory outside checkout. CI uploads the complete run directory
even on failure, retains artifacts for 30 days and caps the job at 20 minutes.
Accepted qualification also needs a checked durable archive outside expiring
Actions storage. See [the storage contract](verification-contract.md#storage).
Both PR jobs must succeed on the current PR head before software acceptance;
self-hosted runner queue time is separate from job execution time. Neither job
is evidence that David is free or usable. See [the roadmap](roadmap.md).

## `merge-gate`

Pulls the pinned `pyasc-sim-dev` image from ghcr.io and compile-checks every
module in `reference/asctile/target/` with
`tests/tools/run_target_test.py --backend Model --platform Ascend950PR_9599`.

Compile-only is deliberate. Each module carries roughly 100 parametrised
cases and the camodel is slow, so this job attempts compilation without
numerical execution. Full numerical execution requires a separate explicit
run. No schedule is configured in the current CI workflow.
See [review-order.md](review-order.md) for running it by hand.

Runs natively on the self-hosted arm64 runner (no x86 emulation). The image
is pinned by digest and tied to a recorded pyasc commit. A mutable image tag
would make results incomparable across runs.

The workflow discovers modules recursively and does not skip
`test_add_inplace.py`. Resource-blocked modules are reported separately; the
current merge job fails on compilation failures but only reports resource
blocks. A green merge job therefore does not by itself prove that every
upstream case compiled. Read its summary and the dated
[verification status](verification-status.md).

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
directory that was never tracked.

The published site is retired too, not merely unbuilt. Deleting the workflow
would have left `https://aloschilov.github.io/pyasc-skill-stack/` live and
frozen, still serving the campaign evidence this repository pruned — an
orphaned page that nothing could refresh or correct. GitHub Pages is disabled
for the repository, so there is no stale public claim outliving its source.

The unused dashboard generator, partial-nightly detector and artifact merger
were also removed after confirming that no current workflow or tool calls them.
Their historical documentation links to preserved Git revisions.

The original campaign is on the `archive/campaign-2026` branch:

```bash
git show archive/campaign-2026:.github/workflows/ci.yml
```

Restoring any of it means first restoring an evidence source for it to read,
and that source should be CANNBench rather than checked-in JSON — see
[evidence-sources.md](evidence-sources.md).

## Local test tiers

```bash
bash tests/ci-gate.sh --tier pr   # full offline PR gate
bash tests/run-tests.sh --fast    # L1 portion only
bash tests/run-tests.sh --all     # adds L2 behavior and L3 integration
```

`tests/behavior/` makes real `opencode` provider calls, so `--all` costs
money and time and is not run in CI.
