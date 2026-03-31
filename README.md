# pyasc-kernel-dev

AI-assisted workflow for developing custom Ascend operators using pyasc — a Python-surface language that compiles to Ascend C via JIT and runs on Ascend NPUs.

This project is a controlled port of the [CANN Skills](../skills/) Ascend C generator architecture. It preserves the reusable orchestration skeleton (phased workflow, checkpoints, skill modules, test pyramid) while replacing all Ascend C domain logic with pyasc equivalents.

## Quick start

```bash
# Run tests to verify the project
bash tests/run-tests.sh --all

# Initialize a new kernel project
bash skills/pyasc-codegen-workflow/scripts/init_kernel_project.sh my_kernel

# Check environment
bash skills/pyasc-env-check/scripts/check_env.sh
```

## Project architecture

```
pyasc-kernel-dev/
├── README.md                          # This file
├── teams/
│   └── pyasc-kernel-dev-team/
│       ├── AGENTS.md                  # Team definition (primary agent)
│       └── kernels/                   # Generated kernel workspace
│           └── add/                   # First scenario: vector add
│               ├── kernel.py
│               ├── docs/design.md
│               └── README.md
├── skills/
│   ├── pyasc-codegen-workflow/        # Core workflow (phases + checkpoints)
│   ├── pyasc-docs-search/            # Documentation and tutorial index
│   ├── pyasc-api-patterns/           # API usage best practices
│   ├── pyasc-syntax-constraints/     # Python syntax support/restrictions
│   ├── pyasc-build-run-verify/       # JIT build, run, verification
│   ├── pyasc-code-review/            # Code review against constraints
│   ├── pyasc-env-check/              # Environment verification
│   └── pyasc-task-focus/             # Task focus / attention management
├── golden/
│   ├── tutorials/                    # Golden reference kernels from pyasc
│   └── docs/                         # Key pyasc documentation snapshots
└── tests/
    ├── run-tests.sh                  # Test runner
    ├── lib/test-helpers.sh           # Shared test utilities
    ├── unit/                         # L1: structure/content validation
    ├── behavior/                     # L2: trigger/premature-action checks
    └── integration/                  # L3: end-to-end workflow tests
```

## Skills library

| Skill | Purpose |
|-------|---------|
| `pyasc-codegen-workflow` | Complete kernel development workflow with phased execution and checkpoints |
| `pyasc-docs-search` | Local-first documentation and tutorial search index |
| `pyasc-api-patterns` | API usage patterns, quick reference, and best practices |
| `pyasc-syntax-constraints` | Python syntax support/restrictions inside `@asc.jit` |
| `pyasc-build-run-verify` | JIT compilation, execution, and output verification |
| `pyasc-code-review` | Hypothesis-testing code review against pyasc constraints |
| `pyasc-env-check` | Development environment verification |
| `pyasc-task-focus` | Task focus and attention management via todo.md |

## Mapping table: Ascend C -> pyasc

| Ascend C Component | pyasc Analog | Action | Rationale |
|---|---|---|---|
| `ascendc-kernel-develop-workflow` | `pyasc-codegen-workflow` | Adapt | Preserve phased structure (Phase 0-3, CP-0 to CP-3). Replace C++/CMake/ACL steps with Python JIT model. |
| `ascendc-docs-search` | `pyasc-docs-search` | Replace | Point at pyasc docs/tutorials/tests instead of asc-devkit. Keep local-first priority. |
| `ascendc-api-best-practices` | `pyasc-api-patterns` | Replace | Replace Ascend C API categories with pyasc API surface (GlobalTensor, LocalTensor, data_copy, add, sync). |
| `ascendc-npu-arch` | (kept as reference) | Keep | Same NPU targets (Atlas A2/A3). Minimal adaptation needed. |
| `ascendc-tiling-design` | `pyasc-syntax-constraints` | Replace | pyasc complexity center is syntax restrictions, not host-side tiling. |
| `ascendc-precision-debug` | — | Defer | Out of scope for first vertical slice. |
| `ascendc-runtime-debug` | `pyasc-build-run-verify` | Replace | Replace ACL error codes with pyasc JIT diagnostics and verification. |
| `ascendc-env-check` | `pyasc-env-check` | Adapt | Check Python, pyasc, CANN, torch instead of npu-smi and CANN env vars only. |
| `ascendc-ut-develop` | — | Defer | Second-wave skill. |
| `ascendc-st-design` | — | Defer | Out of scope for first vertical slice. |
| `ascendc-code-review` | `pyasc-code-review` | Adapt | Keep hypothesis-testing method. Replace C++ specs with Python + pyasc syntax constraints. |
| `ascendc-task-focus` | `pyasc-task-focus` | Keep | Nearly domain-agnostic. Renamed prefix only. |
| `ascendc-whitebox-design` | — | Defer | Out of scope for first vertical slice. |
| `AGENTS.md` (team) | `AGENTS.md` (pyasc team) | Adapt | Same YAML structure. Replace Ascend C doctrine with pyasc doctrine. |
| `tests/` (L1/L2/L3) | `tests/` (L1/L2/L3) | Adapt | Same pyramid. Replace skill names and trigger words. |
| `asc-devkit/` | pyasc source tree | Replace | Reference `~/workspace/pyasc/` instead of bundled devkit. |
| `ops/{operator}/` | `kernels/{name}/` | Adapt | Simpler layout: no CMake, just Python scripts + docs. |

## What was preserved (orchestration skeleton)

These domain-independent patterns were extracted from the Ascend C project and kept:

- **Team definition**: YAML frontmatter with skill list, forced workflow enforcement, core principles
- **Phased workflow**: Phase 0 (env) -> Phase 1 (design) -> Phase 2 (implement + review) -> Phase 3 (verify)
- **Checkpoints**: CP-0 through CP-3 with quantitative exit criteria
- **Two-task implementation loop**: Task 1 (implement + self-review) -> Task 2 (acceptance review)
- **Skill module structure**: SKILL.md + references/ + scripts/ + templates/
- **Search priority**: Local docs first -> examples -> source code
- **Test pyramid**: L1 unit -> L2 behavior -> L3 integration
- **Task focus**: todo.md-based attention management

## What was replaced for pyasc

| Ascend C Concept | pyasc Replacement |
|---|---|
| C++ kernel code (.asc files) | Python kernel code (`@asc.jit`) |
| CMake build system | JIT compilation (automatic) |
| ACL runtime boilerplate | `kernel[cores, stream](...)` launch syntax |
| Host/kernel split | Single Python module (kernel + launch + verify) |
| asc-devkit (1022 API docs) | pyasc source tree (docs, tutorials, tests) |
| Tiling design (host-side) | Syntax constraints (supported Python subset) |
| npu-smi / CANN env check | Python/pyasc/CANN/torch check |
| NPU golden comparison | `torch.allclose` / numpy verification |
| C++ code style rules | Python (PEP 8) + pyasc syntax constraints |
| DataCopy/TBuf/Pipeline APIs | asc.data_copy, asc.add, set_flag/wait_flag |

## What remains out of scope

- `ascendc-precision-debug` — precision debugging (not yet ported)
- `ascendc-ut-develop` — unit test development workflow (not yet ported)
- `ascendc-st-design` — system test design (not yet ported)
- `ascendc-whitebox-design` — whitebox test design (not yet ported)
- `ascendc-npu-arch` — NPU architecture (kept as-is, minimal pyasc adaptation)
- Full runtime execution — pyasc and NPU hardware required for actual kernel runs
- Advanced tutorials (matmul, fused ops) — golden set includes them but no dedicated skills yet

## First scenario

The project includes one complete end-to-end scenario:

**"Generate a simple pyasc add kernel"**

1. Phase 0: `init_kernel_project.sh add` + `verify_environment.sh add`
2. Phase 1: Design document based on tutorial 01_add
3. Phase 2: `kernel.py` implementing `vadd_kernel` with manual sync
4. Phase 3: Verification via `torch.allclose(z, x + y)`

The generated kernel is at `teams/pyasc-kernel-dev-team/kernels/add/kernel.py`.

## Testing

```bash
# L1 unit tests only (no CLI required, < 30s)
bash tests/run-tests.sh --fast

# L2 behavior tests
bash tests/run-tests.sh --category behavior

# All tests including L3 integration
bash tests/run-tests.sh --all

# List available tests
bash tests/run-tests.sh --list

# Run a specific test
bash tests/run-tests.sh --test integration/test-simple-kernel.sh
```

## Requirements

- Bash 4.0+
- Python 3.9-3.12
- pyasc (`pip install pyasc`) — for kernel execution
- CANN Toolkit — for NPU/Model backend
- numpy < 2.0
- (Optional) torch, torch_npu — for tensor management and NPU support

## License

This project is a port of the CANN Skills architecture. See the original project for license terms.
