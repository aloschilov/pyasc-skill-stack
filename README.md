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
> with a design document and direct numerical assertions. Read the installed
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
| `pyasc-codegen-workflow` | Environment, design, implementation and verification workflow |
| `pyasc-api-patterns` | asctile API, tiling and data movement |
| `pyasc-syntax-constraints` | Supported kernel syntax |
| `pyasc-docs-search` | Find API documentation and references |
| `pyasc-build-run-verify` | Compilation, execution and numerical checks |
| `pyasc-code-review` | Review kernel correctness and constraints |
| `pyasc-env-check` | Diagnose runtime prerequisites |
| `pyasc-task-focus` | Keep the development task focused |
| `pyasc-target-operator` | Extend the upstream target suite |
| `pyasc-cannbench-kernel` | Configure CANNBench operators; performance is a separate lane |

The first eight retain the core skill organization used in July. The two
specialized skills do not add requirements to the quick start.

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
- [Review order](docs/review-order.md): run the upstream target suite.
- [Roadmap](docs/roadmap.md): software work and deferred hardware work.

Keep run artifacts outside checkout. Git holds checks, regression fixtures and
short evidence summaries. Preserve historical evidence; older campaign material
is available on `archive/campaign-2026`.

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
