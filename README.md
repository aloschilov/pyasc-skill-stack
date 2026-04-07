# pyasc-skill-stack

Reusable Agent skill modules for pyasc kernel development on Huawei Ascend NPUs.

With skills installed, a short prompt like _"Develop an abs operator for float16, shapes [1,128], [4,2048], [32,4096]"_ is enough — the agent handles environment setup, design, implementation, review, and verification autonomously.

## Target users

- Ascend NPU application developers
- pyasc operator developers
- Contributors who wish to extend the skill set

## Quick start

### Prerequisites

These must be available before starting:

- `opencode` CLI installed
- Python 3.10.x with `pyasc >= 1.1.1` and `torch`
- CANN Toolkit (see [docs/cann-setup.md](docs/cann-setup.md))
- Simulator libraries for `Ascend910B1`

### Step 1. Clone the repository

```bash
git clone git@github.com:aloschilov/pyasc-skill-stack.git
cd pyasc-skill-stack
```

### Step 2. Set up the CANN environment

```bash
source $HOME/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
```

Quick check:

```bash
python3.10 -c "import asc; print('pyasc OK')"
python3.10 -c "import torch; print('torch OK')"
```

### Step 3. Install skills for OpenCode

```bash
TEAM_DIR=teams/pyasc-kernel-dev-team
INSTALL_DIR=$TEAM_DIR/.opencode

mkdir -p "$INSTALL_DIR/skills"
ln -sfn "$(pwd)/$TEAM_DIR/AGENTS.md" "$INSTALL_DIR/AGENTS.md"

for skill_dir in skills/*; do
    ln -sfn "$(pwd)/$skill_dir" "$INSTALL_DIR/skills/$(basename "$skill_dir")"
done
```

Quick check:

```bash
ls -la "$INSTALL_DIR/skills"
```

### Step 4. Start OpenCode

```bash
cd teams/pyasc-kernel-dev-team
opencode
```

Then give the agent a short prompt:

```text
Help me develop an abs operator that supports float16 data type.
The shape is mainly [1,128], [4,2048], [32,4096].
```

The agent will autonomously walk through environment check, design, implementation, review, and verification. Do not intervene unless the agent hits an external platform issue.

### Step 5. Verify the result

After the agent finishes, run the generated kernel manually:

```bash
source $HOME/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
python3.10 teams/pyasc-kernel-dev-team/kernels/abs_f16/kernel.py -r Model -v Ascend910B1
```

### Alternative: run from repo root without installing skills

If you prefer not to set up the `.opencode` directory, OpenCode can discover skills via the repo-level `opencode.json`:

```bash
cd ~/workspace/pyasc-skill-stack
opencode
```

This mode relies on `opencode.json` for skill discovery. The installed-skills approach (Step 3) is recommended because it gives OpenCode explicit skill context, allowing shorter prompts.

## What the agent does

When skills are installed correctly, the agent:

1. Loads `pyasc-codegen-workflow` and follows a 4-phase workflow (Phase 0 → 1 → 2 → 3)
2. Initializes the kernel project directory
3. Retrieves documentation from the golden set and API references
4. Writes a design document with API selection and syntax checks
5. Implements `kernel.py` using the kernel template and reviewed patterns
6. Conducts self-review and acceptance review against pyasc constraints
7. Runs verification (simulator or JIT fallback) and writes a verification record
8. Delivers a runnable kernel with all workflow artifacts

## Project structure

```
pyasc-skill-stack/
├── skills/                           # Skill modules (agent reads these)
│   ├── pyasc-codegen-workflow/       # Core 4-phase workflow
│   ├── pyasc-api-patterns/           # API usage patterns
│   ├── pyasc-syntax-constraints/     # Supported syntax inside @asc.jit
│   ├── pyasc-docs-search/            # Documentation index
│   ├── pyasc-build-run-verify/       # Build, run, verification
│   ├── pyasc-code-review/            # Code review checklist
│   ├── pyasc-env-check/              # Environment verification
│   └── pyasc-task-focus/             # Task tracking
├── teams/
│   └── pyasc-kernel-dev-team/
│       ├── AGENTS.md                 # Team agent definition
│       ├── quickstart.md             # Manual development guide
│       └── kernels/                  # Generated kernel workspace
├── golden/
│   ├── tutorials/                    # Golden reference kernels
│   ├── kernels/                      # Verified golden kernels (abs, sub, mul)
│   └── docs/                         # Local pyasc API documentation
├── tests/                            # Automated test pyramid
│   ├── run-tests.sh                  # Test runner
│   ├── ci-gate.sh                    # CI entry point (pr/merge/nightly)
│   ├── unit/                         # L1: structure/content checks
│   ├── behavior/                     # L2: trigger/action checks
│   └── integration/                  # L3: end-to-end workflow
└── opencode.json                     # Repo-level skill discovery
```

## Skills library

| Skill | Purpose |
|-------|---------|
| `pyasc-codegen-workflow` | 4-phase workflow: environment → design → implementation + review → verification |
| `pyasc-api-patterns` | API usage patterns, dynamic tiling, `ConstExpr` guidance |
| `pyasc-syntax-constraints` | Python syntax support/restrictions inside `@asc.jit` |
| `pyasc-docs-search` | Local-first documentation and tutorial search |
| `pyasc-build-run-verify` | JIT compilation, simulator execution, output verification |
| `pyasc-code-review` | Code review against pyasc constraints |
| `pyasc-env-check` | Python, pyasc, CANN, torch environment checks |
| `pyasc-task-focus` | Task focus and attention management |

## Testing

```bash
# Quick PR gate (L1 + JIT verification, < 60s)
bash tests/ci-gate.sh --tier pr

# Full test suite
bash tests/run-tests.sh --all
```

See [tests/README.md](tests/README.md) for details.

## License

This project is a port of the [CANN Skills](https://gitcode.com/cann/skills) architecture. See the original project for license terms.
