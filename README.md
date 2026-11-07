# pyasc-skill-stack

Skills for writing, reviewing and verifying PyASC kernels with the
`asc.experimental.asctile` API. The repository contains the skills, a pinned
upstream correctness reference and the tools used to check generated kernels.

## Quick start: generate and verify one kernel

Use Linux with Git, Python 3.10+, `jq`, Docker and a configured `opencode` CLI.
The host does not need CANN, PyASC or torch for this path: verification runs
inside the pinned development simulator image. The recorded fresh-generation
qualification used Linux ARM64 and OpenCode 1.19.0; see
[verification status](docs/verification-status.md) for exact coverage.

```bash
git clone https://github.com/aloschilov/pyasc-skill-stack.git
cd pyasc-skill-stack
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pyyaml
export PYASC_PYTHON=python

IMAGE=$(jq -r .image.ref reference/images/ENV.json)
docker pull "$IMAGE"
```

The generation check uses the tracked `cloud-glm-5.3` profile, independently of
your global OpenCode model settings. Supply `OPENCODE_BASE_URL` (the
OpenAI-compatible endpoint) and `OPENCODE_API_KEY` through your existing provider
configuration. Do not put credentials in this repository. Other tracked
profiles and their required variables are documented in the
[profile guide](integrations/opencode/profiles/README.md).

```bash
python tests/tools/opencode_profile.py --profile cloud-glm-5.3 \
  --require-credentials --out /dev/null
python tests/tools/generation_acceptance.py \
  --results-dir "$HOME/pyasc-qualification-runs/generation-$(date +%Y%m%d-%H%M%S)"
```

This makes a paid provider call and generates a vector-add kernel and design
document in a temporary project. The unchanged kernel must pass static checks,
JIT compilation and simulator numerical verification, including a
suppressed-launch negative control. Missing prerequisites block acceptance.
The results directory must be new and outside the checkout; it retains the
source, manifest, logs and timings, including failures.

This checks the kernel's observed comparisons, not an independent correctness
oracle. It proves neither untested shapes nor NPU behavior or performance.
See the [verification contract](docs/verification-contract.md).

For an official CANNBench case, follow the [OpenCode and CaModel walkthrough](integrations/cannbench/README.md#verify-an-official-case-in-camodel). It checks the generated file against the benchmark's original inputs and golden output.

## Develop your own kernel

Keep the checkout available: the skills use its references and verification
tools. To work in a separate project with the same tracked provider profile:

```bash
STACK=$(pwd)
PROJECT="$HOME/pyasc-projects/my kernel"
mkdir -p "$PROJECT"
./install.sh --target "$PROJECT/skills"
python tests/tools/opencode_profile.py --profile cloud-glm-5.3 \
  --skills on --require-credentials --out "$PROJECT/opencode.json"
cd "$PROJECT"
opencode
```

For example, ask:

> Write an asctile vector-add kernel for float32 inputs of sizes 1024 and 1025,
> with a design document and direct `assert torch.allclose(...)` checks against
> `x + y` at both sizes. Read the installed
> skills. PyASC is available in the pinned Docker image, not on the host; write
> the files without searching for a host runtime. I will run the repository's
> qualification command afterward.

After generation, qualify the actual output path:

```bash
python "$STACK/tests/tools/qualify_generated.py" "$PROJECT/kernel.py" \
  --results-dir "$HOME/pyasc-qualification-runs/kernel-$(date +%Y%m%d-%H%M%S)"
```

If the agent chose a subdirectory, use that kernel's path. A repair is a new
candidate; retain the failed original. The digest pin selects the runtime and
the verifier preserves its CANN entrypoint and sets the `Model` backend.

For an existing host CANN/PyASC environment, use the separate
[host setup guide](docs/cann-setup.md). This repository consumes upstream
runtime images; it does not build them.

## Skills

| Skill | Responsibility |
|---|---|
| `asctile-codegen-workflow` | Environment, design, implementation and verification workflow |
| `asctile-api-patterns` | asctile API, tiling and data movement |
| `asctile-syntax-constraints` | Supported kernel syntax |
| `asctile-docs-search` | Find API documentation and references |
| `asctile-build-run-verify` | Compilation, execution and numerical checks |
| `asctile-code-review` | Review kernel correctness and constraints |
| `asctile-env-check` | Diagnose runtime prerequisites |
| `asctile-task-focus` | Keep the development task focused |
| `asctile-target-operator` | Extend the upstream target suite |
| `asctile-cannbench-kernel` | Configure CANNBench operators; performance is a separate lane |

The target-suite and CANNBench skills apply to their respective operator contracts.
The design → implementation → review → validation → tuning workflow is in
`asctile-codegen-workflow`; these are caller-managed phases, not automatically
spawned agents. See [CI coverage](docs/ci-gates.md#agent-workflows).

`./install.sh` without `--target` links skills globally for Cursor and OpenCode;
`--claude` or `--agents` adds those destinations. Re-running is idempotent.
Existing files, directories and links to other locations are preserved with an
error. When moving a checkout, remove its old links explicitly before
reinstalling. Only dangling links into the current checkout are pruned.

## Maintenance and evidence

```bash
PYASC_PYTHON=python bash tests/ci-gate.sh --tier pr
python tests/tools/verification_contract.py \
  --results-dir "$HOME/pyasc-qualification-runs/contract-$(date +%Y%m%d-%H%M%S)"
```

The PR gate is offline. The contract needs Docker and the pre-pulled pinned
image. `tests/run-tests.sh --all` also includes provider-backed tests and may
incur charges; see [tests](tests/README.md).

- [Correctness and performance sources](docs/evidence-sources.md): the pinned
  Huawei target suite is the correctness reference; generated kernels are test
  subjects, never correctness or performance references.
- [Verification status](docs/verification-status.md): measured outcomes and
  remaining limitations. Simulator success does not close the hardware gate.
- [Public evidence dashboard](https://aloschilov.github.io/pyasc-skill-stack/):
  retained results with exact scope, freshness and archive links; [publication contract](dashboard/README.md).
- [Review order](docs/review-order.md): run the upstream target suite.
- [CANNBench smoke](integrations/cannbench/README.md): one-operator external NPU evaluation.
- [Review/tuning automation follow-up](https://github.com/aloschilov/pyasc-skill-stack/issues/20):
  acceptance criteria for the remaining agent handoffs.

Keep run artifacts outside checkout. Git holds checks, regression fixtures and
short evidence summaries. Preserve historical evidence; older campaign material
is available on `archive/campaign-2026`.

## Skill names and delivery

The ten skill IDs now use `asctile-*`, matching the experimental AscTile API.
Their directories and frontmatter names are identical; new kernel templates use
`from asc.experimental import asctile`. The repository name and Python package
remain `pyasc-skill-stack` and `pyasc-dev` respectively.

After updating an existing checkout, rerun `install.sh` for the same target.
It installs the new names and removes only dangling links into that checkout.
Foreign links and copied skill directories remain untouched; remove old copies
explicitly to avoid duplicate discovery. Historical receipts keep their original
`pyasc-*` names and hashes. Existing CI continuation validates that recorded
namespace instead of relabeling an old run.

The generation matrix and reviewer use the same `asctile-*` names. Retained
review plans select their original namespace from hash-bound inputs; mixed
namespaces or missing reviewer skills fail before execution. Historical
generation results retain their original skills and do not qualify the renamed
context for new generation.

No optional `asctile-u-*` namespace is introduced: a name would not make
checkout-dependent verification self-contained. `asctile-target-operator` is
maintainer-facing; the other skills guide operator development, with full
verification requiring the checkout described above. A wheel distributor can
copy the `skills/` tree to `experimental/asctile/skills`; this repository does
not modify or publish an upstream wheel as part of skill installation.

The rename at `0423963` was checked through installation from a skills-only
archive and native OpenCode 1.19.0 discovery (10/10 skills). CI
[42221103276](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/42221103276)
passed 72 Python tests, the 18-case verifier contract and six short-row simulator
regressions. [Retained delivery evidence](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-delivery-0423963):
`qualification-delivery-0423963.tar.gz`, SHA256
`ee32fb799cae5c5bc36f4eb0ad6c2a55e378b9983623f5a089f8250a75779b31`.
This does not establish new generation quality, performance or NPU coverage.

## Release archives

The current tag workflow publishes **skills-only** archives: `skills/`,
`install.sh`, this README and `CHANGELOG.md`. They can install skill links but
omit the runtime pins, references and verifier used above. Use the full checkout
for the documented end-to-end workflow. Archive installation alone is not
qualification. See [release workflow](.github/workflows/release.yml).

## License

This project is a port of the [CANN Skills](https://gitcode.com/cann/skills)
architecture. See the original project for license terms; vendored upstream
files retain their notices.
