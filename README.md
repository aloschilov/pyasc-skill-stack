# pyasc-skill-stack

Reusable Agent skill modules for pyasc kernel development on Huawei Ascend NPUs using the **asctile** high-level tile-based API.

**Scope:** writing an asctile kernel, verifying it, tuning its performance, and configuring CANNBench — and nothing else. Ten skills, one installer, one pinned correctness reference.

With skills installed, a short prompt like _"Develop an abs operator for float16, shapes [1,128], [4,2048], [32,4096]"_ is enough — the agent handles environment setup, design, implementation, review, and verification autonomously.

**What counts as evidence:** the Huawei-authored target suite in `reference/asctile/target/` proves correctness, CANNBench produces the performance ratio, and the TTK manifest defines the CANN-side baseline that ratio is measured against. **No model-written kernel is a reference** — for correctness, for tuning, or as a baseline. See [docs/evidence-sources.md](docs/evidence-sources.md).

## Target users

- Ascend NPU application developers
- pyasc operator developers
- Contributors who wish to extend the skill set

## Quick start

### Prerequisites

These must be available on the host before starting:

- `opencode` CLI installed
- CANN Toolkit installed somewhere on disk (see [docs/cann-setup.md](docs/cann-setup.md))
- Simulator libraries for `Ascend950PR_9599` (shipped with CANN — the only platform the stack targets)
- Linux on `x86_64` or `aarch64`, glibc 2.28 or newer (the wheels are `manylinux_2_28`)
- CPython 3.10, 3.11 or 3.12 — see the support matrix below

> **Runtime contract.** The skills target the **`asctile` programming model inside PyAsc**, obtained from the published `pyasc-dev` wheel:
>
> ```bash
> python3.10 -m venv .venv && source .venv/bin/activate
> python -m pip install --pre pyasc-dev
> ```
>
> `asctile` is compiled into the wheel at build time. Setting `PYASC_SETUP_EXPERIMENTAL=1` during `pip install` of an existing wheel does nothing — it is a *source build* flag.

### Support matrix

`pyasc-dev` is a development release, built and published upstream by
[`compiler-team-ru/asctile`](https://github.com/compiler-team-ru/asctile) as a
matrix of CPython version x architecture. Wheels are `manylinux_2_28`
(glibc 2.28 or newer), so they are independent of the CANN version and run on
Ubuntu 22.04 and 24.04 alike. No macOS, no Windows.

**This table is not maintained here, because it would be wrong within a day.**
The matrix genuinely moves between publications: as of
`1.1.1.dev20261023173239` the two CPython 3.9 cells were marked incompatible
(upstream used `dataclass(slots=True)`, which needs 3.10), and one publication
later, at `1.1.1.dev20261024081924`, all eight cells built. Read the live state
from the `wheel-manifest.json` attached to the newest release:

```bash
curl -fsSL https://github.com/compiler-team-ru/asctile/releases/latest/download/wheel-manifest.json \
  | jq '{version, source_sha, status, published: (.targets|keys), incompatible: (.incompatible_targets|keys)}'
```

It records, per cell, the wheel's sha256, its dependency hashes, the pinned
LLVM revision and the manylinux base-image digest it was built in.

`pip` resolves the newest version that has a wheel for your platform, so
different hosts legitimately land on different versions. When that matters,
pin the exact version this repository is verified against — see
[reference/images/ENV.json](reference/images/ENV.json).

The wheel supplies the compiler and runtime only. **CANN, its drivers, and any hardware are installed separately** and are not bundled. It also does not depend on `torch`, which the target suite needs; install that yourself.

### Step 1. Clone the repository

```bash
git clone git@github.com:aloschilov/pyasc-skill-stack.git
cd pyasc-skill-stack
```

### Step 1b. Install host prerequisites (Ubuntu, optional)

If you're starting from a clean Ubuntu 22.04 / 24.04 host (x86_64 or aarch64), the repo ships a small idempotent helper that checks for and offers to install everything needed for Options A and B:

```bash
bash scripts/install-host-deps.sh                              # interactive: shows plan, asks once
bash scripts/install-host-deps.sh --yes                        # non-interactive (CI)
bash scripts/install-host-deps.sh --skip-docker --skip-opencode  # apt packages only
```

Covers: apt packages (`build-essential`, `git`, `curl`, `ccache`, `clang`, `lld`, `python3.10`, `python3.10-venv`, `python3-pip`), Docker Engine via the official `download.docker.com` repo, and the `opencode` CLI via the upstream installer.

Out of scope for the script: the CANN Toolkit itself (see [docs/cann-setup.md](docs/cann-setup.md)) and any pyenv-managed Python.

### Step 2. Configure paths (once per shell)

The repo does not assume a fixed location for CANN or for your `pyasc` checkout — both are expressed as environment variables with sensible defaults:

```bash
export CANN_HOME="${CANN_HOME:-$HOME/Ascend/cann}"
export PYASC_SRC="${PYASC_SRC:-$HOME/workspace/pyasc}"
```

Common CANN install locations you may need to point `CANN_HOME` at:

- `$HOME/Ascend/cann` (default user install)
- `$HOME/Ascend/ascend-toolkit/latest`
- `$HOME/Ascend/cann-<version>` (e.g. `cann-9.0.0`)
- `/usr/local/Ascend/ascend-toolkit/latest` (system-wide install, and the location used inside the Docker image)

`PYASC_SRC` is only needed for Options C and D below.

### Step 3. Install `pyasc` (with `asctile`) — choose one option

#### Option A — Published wheel (recommended)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --pre pyasc-dev
```

`--pre` is required: every `pyasc-dev` release is a development version, so `pip` ignores them all without it.

Install `pyasc-dev` into a **dedicated virtual environment**. It provides the same `asc` import package as the stable `pyasc` distribution, so the two must not be installed together.

#### Option B — Docker (self-contained, for a clean machine)

This repository no longer builds simulator images. It consumes one cell of the
matrix published by
[`compiler-team-ru/pyasc-images`](https://github.com/compiler-team-ru/pyasc-images),
which builds Ubuntu x CANN x CPython x architecture, smoke-tests every cell and
publishes the result as `matrix.json` next to the images. Pull by digest, never
by tag — the exact digest this repo is verified against lives in
[reference/images/ENV.json](reference/images/ENV.json):

```bash
IMAGE=$(jq -r .image.ref reference/images/ENV.json)
docker pull "$IMAGE"
docker run --rm -it \
  --ulimit nofile=1048576:1048576 \
  -v "$(pwd)":/workspace -w /workspace \
  "$IMAGE"
```

The images allow anonymous pulls. If you get `denied`, you are most likely
logged in to `ghcr.io` as an account without access to the `compiler-team-ru`
org — the daemon sends that credential instead of pulling anonymously. Pull
with an empty config to work around it:

```bash
cfg=$(mktemp -d) && echo '{}' > "$cfg/config.json"
DOCKER_CONFIG="$cfg" docker pull "$IMAGE"
```

> The `--ulimit nofile=1048576` is required: the CANN simulator opens hundreds of dump files per core (16 cores × tens of files), and the default Docker `nofile` limit of 1024 causes runtime errors like `Failed opening file ... Too many open files`.

All remaining steps (sourcing CANN, running `opencode`, running kernels) happen **inside the container**.

Two things the image does **not** give you:

- **`torch`.** The upstream target suite imports it, but neither the image nor
  `pyasc-dev` depends on it. Install a CPU build yourself before running the
  suite; CI does this in a separate step.
- **Correctness qualification.** Each cell is verified only to the extent
  recorded in `matrix.json`'s own `scope` field: *"Upstream vector-add compile
  and simulator smoke, 128 float32 elements. No hardware or broad numerical
  qualification."*

To move onto a different cell or a newer publication, run
`scripts/sync-image-pin.sh` rather than editing the digest by hand;
`scripts/check-image-pin.sh` runs in `--fast` and fails if the pin, the CI
workflow and the target-suite pin stop agreeing.

#### Option C — Local build from `pyasc` source (for compiler developers)

Requires CANN and a C++ toolchain on the host. Clone `pyasc` anywhere (`$PYASC_SRC` tells the rest of the workflow where it lives) and build it into a venv:

```bash
git clone https://gitcode.com/cann/pyasc.git "$PYASC_SRC"
( cd "$PYASC_SRC" && git checkout v2 )   # or the exact commit you need

python3.10 -m venv .venv
source .venv/bin/activate

LLVM_ARCHIVE_BASE=https://oaitriton.blob.core.windows.net/public/llvm-builds
case "$(uname -m)" in
    x86_64)  LLVM_URL="${LLVM_ARCHIVE_BASE}/llvm-86b69c31-ubuntu-x64.tar.gz"   ;;
    aarch64) LLVM_URL="${LLVM_ARCHIVE_BASE}/llvm-86b69c31-ubuntu-arm64.tar.gz" ;;
    *) echo "Unsupported arch: $(uname -m)" >&2 ; return 1 ;;
esac
mkdir -p "$HOME/.llvm"
curl -fsSL "$LLVM_URL" | tar -xz --strip-components=1 -C "$HOME/.llvm"
export LLVM_INSTALL_PREFIX="$HOME/.llvm"

PYASC_SETUP_EXPERIMENTAL=1 pip install "$PYASC_SRC"
```

`PYASC_SETUP_EXPERIMENTAL=1` is **required**: without it the build omits `asc.experimental.asctile` entirely and every skill in this repo will fail to import. It only has an effect while compiling from source.

See the upstream [`pyasc` build-from-source guide](https://gitcode.com/cann/pyasc) (file `docs/installation/build-from-source.rst`) for both prebuilt LLVM archive URLs and optional build flags (`PYASC_SETUP_CCACHE`, `PYASC_SETUP_CLANG_LLD`, etc.).

#### Option D — Shortcut: existing system-wide `pyasc` dev install

If `asctile` is already importable from the base `python3.10` (for example, you maintain a `pip install -e "$PYASC_SRC"` globally), create a venv that inherits system site-packages:

```bash
python3.10 -m venv --system-site-packages .venv
source .venv/bin/activate
```

This is the fastest path but depends entirely on your pre-existing Python environment — prefer A or B for a reproducible setup.

### Step 4. Source CANN and start OpenCode

Same commands for every option (for Docker, run them inside the container — `CANN_HOME` is preset there):

```bash
source "$CANN_HOME/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
python -c "from asc.experimental import asctile; print('asctile OK:', asctile.__file__)"
opencode
```

The import path is `asc.experimental.asctile`. A bare `import asctile` works only for some source builds and **fails with the published wheel**, so prefer the fully-qualified form everywhere.

To see which upstream commit your runtime was built from:

```bash
python -c "import importlib.metadata as m; print(m.version('pyasc-dev'))"
python -c "import importlib.metadata as m; d=m.metadata('pyasc-dev')['Description']; import re; print(re.search(r'Source: .([0-9a-f]{40}).', d).group(1))"
```

Skill discovery is handled by `opencode.json` in the repo root — no additional installation step is needed.

Then give the agent a short prompt:

```text
Help me develop an abs operator that supports float16 data type.
The shape is mainly [1,128], [4,2048], [32,4096].
```

The agent will autonomously walk through environment check, design, implementation, review, and verification. Do not intervene unless the agent hits an external platform issue.

### Step 5. Verify the result

After the agent finishes, run the generated kernel manually:

```bash
source "$CANN_HOME/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
python3.10 <kernel_project>/kernel.py -r Model -v Ascend950PR_9599
```

`Backend.Model` is the simulator, not hardware. A pass here is a simulator pass and reports should say so. A worked, runnable example is in [`skills/pyasc-build-run-verify/references/execution-example.md`](skills/pyasc-build-run-verify/references/execution-example.md).

## What the agent does

When skills are discovered correctly, the agent:

1. Loads `pyasc-codegen-workflow` and follows a 4-phase workflow (Phase 0 → 1 → 2 → 3)
2. Initializes the kernel project directory
3. Retrieves documentation from the asctile kernel references and the pinned target suite
4. Writes a design document with asctile API selection and syntax checks
5. Implements `kernel.py` using the asctile kernel template and reviewed patterns
6. Conducts self-review and acceptance review against pyasc constraints
7. Runs verification (simulator or JIT fallback) and writes a verification record
8. Delivers a runnable kernel with all workflow artifacts

## What the agent should not require from you

If skills are loaded correctly, the agent should not:

- ask you to manually read internal skill files
- ask you to create `design.md`, `kernel.py`, or review documents
- ask you to run internal phase scripts
- require a long "management" prompt with all phases listed
- stop at the first error without attempting to fix it

If the agent requires any of the above, there is likely a problem with skill discovery or the `AGENTS.md` configuration.

## Project structure

```
pyasc-skill-stack/
├── skills/                           # Skill modules (agent reads these) — the product
│   ├── pyasc-codegen-workflow/       # Core 4-phase workflow (asctile)
│   ├── pyasc-api-patterns/           # asctile API usage patterns
│   ├── pyasc-syntax-constraints/     # Supported syntax inside @asctile.jit
│   ├── pyasc-docs-search/            # Documentation index
│   ├── pyasc-build-run-verify/       # Build, run, verification
│   ├── pyasc-code-review/            # Code review checklist
│   ├── pyasc-target-operator/        # Authoring and extending the target suite
│   ├── pyasc-cannbench-kernel/       # CANNBench operators and performance config
│   ├── pyasc-env-check/              # Environment verification
│   └── pyasc-task-focus/             # Task tracking
├── reference/
│   ├── asctile/                      # THE correctness reference
│   │   ├── target/                   # Huawei-authored suite (incl. subpackages)
│   │   ├── PIN.json                  # upstream, path, commit, per-file sha256
│   │   └── conftest.py               # upstream's, plus the import-layout alias
│   └── images/
│       └── ENV.json                  # pinned simulator image digest + provenance
├── scripts/
│   ├── sync-target-suite.sh          # re-vendor the suite (finds it by content)
│   ├── check-target-pin.sh           # offline drift check, runs in --fast
│   ├── sync-image-pin.sh             # re-pin onto a cell of the upstream matrix
│   └── check-image-pin.sh            # offline pin/workflow agreement, runs in --fast
├── integrations/cannbench/           # CANNBench tasks: 11 operators
├── tests/
│   ├── run-tests.sh                  # Test runner
│   ├── ci-gate.sh                    # CI entry point (pr tier)
│   ├── unit/                         # L1: structure/content checks
│   ├── behavior/                     # L2: trigger/action checks (makes provider calls)
│   └── integration/                  # L3: end-to-end workflow
├── docs/                             # evidence-sources, verification-status, review-order, …
├── install.sh                        # symlinks skills into agent directories
├── CHANGELOG.md
└── opencode.json                     # Skill discovery config
```

The campaign that produced this repository — evidence JSON, model-written `golden/` kernels, `capabilities.yaml`, `teams/` and dated reports — was removed from `main` and preserved on the `archive/campaign-2026` branch (tag `archive-campaign-2026`). Recover any of it with `git checkout archive/campaign-2026 -- <path>`.

## Skills library

| Skill | Purpose |
|-------|---------|
| `pyasc-codegen-workflow` | 4-phase workflow: environment → design → implementation + review → verification |
| `pyasc-api-patterns` | asctile API usage patterns, tiling with ceildiv, `ConstExpr` guidance |
| `pyasc-syntax-constraints` | Python syntax support/restrictions inside `@asctile.jit` |
| `pyasc-docs-search` | Local-first documentation and tutorial search |
| `pyasc-build-run-verify` | JIT compilation, simulator execution, output verification |
| `pyasc-code-review` | Code review against pyasc constraints |
| `pyasc-target-operator` | Authoring and extending the upstream target suite |
| `pyasc-cannbench-kernel` | CANNBench operators, local compile gate, performance configuration |
| `pyasc-env-check` | Python, pyasc, CANN, numpy environment checks |
| `pyasc-task-focus` | Task focus and attention management |

### Install the skills globally

```bash
./install.sh                      # symlink all skills into Cursor + opencode
./install.sh --claude             # also link ~/.claude/skills
./install.sh --agents             # also link ~/.agents/skills
./install.sh --target ./my-dir    # link into one explicit directory instead
```

Each skill stays in this repo; `install.sh` creates symlinks under
`~/.cursor/skills` and `~/.config/opencode/skills`. Re-run after adding,
removing, or moving skills (stale links are pruned automatically).
`--target DIR` *replaces* the default destinations rather than adding to them,
which is what the release archive is verified against.

## Testing

```bash
bash tests/run-tests.sh --fast    # L1: structure and content, no network, seconds
bash tests/ci-gate.sh --tier pr   # what CI runs on every PR
bash tests/run-tests.sh --all     # adds L2/L3
```

`--fast` is the gate: it must pass before a release tag builds. Note that
`tests/behavior/` makes real `opencode` provider calls, so `--all` costs money
and time.

Running the correctness reference itself is separate — see
[docs/review-order.md](docs/review-order.md) — and needs CANN plus an
installed `pyasc-dev` wheel.

See [tests/README.md](tests/README.md) for details.

## Releases

Tags, not directories. Pushing a `v*` tag runs the fast suite, builds
`tar.gz` and `zip` containing exactly `skills/`, `install.sh`, `README.md` and
`CHANGELOG.md`, verifies that the archive installs, and publishes it. See
[.github/workflows/release.yml](.github/workflows/release.yml) and
[CHANGELOG.md](CHANGELOG.md).

## Headless run (CI / scripted)

For non-interactive verification, use `opencode run` with a pseudo-TTY wrapper:

```bash
source "$CANN_HOME/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
script -qc 'opencode run "Help me develop an abs operator that supports float16 data type. The shape is mainly [1,128], [4,2048], [32,4096]." --format json' /dev/null
```

The `script -qc` wrapper provides the pseudo-TTY that `opencode run` requires in headless environments.

## Verification status

What has been verified and what is still blocked is tracked in
[docs/verification-status.md](docs/verification-status.md), updated in place
rather than by date. Gate statuses are `passed | failed | blocked | not-run`;
a missing runtime, provider or device is an honest `blocked`, never a
fabricated pass.

Current headline: install, skill discovery, compile and numerical gates pass
on the simulator with a hand-written kernel. The **generation** gate — an
agent producing the kernel from a plain-language request — is `blocked` on a
locally configured 8192-token output ceiling, diagnosed in that document.
Hardware and performance are `not-run`.

The evaluation protocol taxonomy, failure taxonomy and evidence schema are in
[docs/evaluation-methodology.md](docs/evaluation-methodology.md). The
capability dashboard that once rendered from `capabilities.yaml` and
`evidence/*.json` was retired with them; it is on `archive/campaign-2026`.

## License

This project is a port of the [CANN Skills](https://gitcode.com/cann/skills) architecture. See the original project for license terms.
