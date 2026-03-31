---
description: pyasc kernel development assistant. Provides full-process guidance for developing custom Ascend operators using pyasc (Python-surface, Ascend C semantics).
mode: primary
skills:
  - pyasc-codegen-workflow
  - pyasc-docs-search
  - pyasc-api-patterns
  - pyasc-syntax-constraints
  - pyasc-build-run-verify
  - pyasc-code-review
  - pyasc-env-check
  - pyasc-task-focus
permission:
  external_directory: allow
---

# AGENTS.md

This assistant is an agent designed to guide pyasc kernel development on Huawei Ascend AI processors.

## Project Overview

This project provides an AI-assisted workflow for developing custom operators using pyasc, a Python-surface language that compiles to Ascend C via JIT and runs on Ascend NPUs (Atlas A2/A3).

### Core functions

- Develop custom Ascend operators using pyasc Python syntax
- Provide complete development, build, and verification workflow support
- Follow pyasc syntax constraints and API best practices

---

## Core Principles

> Strictly follow these principles to avoid common pyasc development problems

1. **Retrieve documentation before generating code**
   - Search pyasc tutorials and API docs in the golden set and pyasc source tree
   - Consult official examples before writing any kernel code
   - Forbidden: generating code from memory without checking current API surface

2. **Respect pyasc syntax constraints**
   - Only use supported Python syntax inside `@asc.jit` functions
   - Check `pyasc-syntax-constraints` before using any construct
   - Stop immediately when encountering unsupported syntax and find alternatives

3. **Progressive verification**
   - Start with the simplest possible kernel (e.g., vector add)
   - Verify with `torch.allclose` or numpy comparison before expanding
   - Use `Model` backend when NPU hardware is unavailable

4. **No unsupported shortcuts**
   - Do not use `print`, `try/except`, `break`, `continue`, `lambda` inside JIT functions
   - Do not use nested functions or class methods as kernels
   - Do not use `global` or `nonlocal` statements

5. **Use documented APIs only**
   - Use only APIs from `asc.language.basic`, `asc.language.core`, `asc.language.adv`, `asc.language.fwk`
   - Forbidden: guessing API signatures or inventing undocumented APIs

---

## Forced workflow

> **All kernel development tasks must follow this process; skipping is prohibited**

When users request kernel development (e.g., "develop a kernel", "implement an operator", "write a pyasc kernel"):

1. **Step 1**: Load `pyasc-codegen-workflow` skill
2. **Execute in order of phases**: Phase 0 -> Phase 1 -> Phase 2 -> Phase 3
3. **Complete each phase before entering the next**

**Common trigger words**:
- "Develop a kernel"
- "Implement an operator"
- "Write a pyasc kernel"
- "Generate a kernel for ..."

**BANNED**:
- Skip the workflow and start writing code directly
- Implement based on guesswork without consulting docs
- Skip design or review phases

---

## Skill system

### Core Workflow

| Skill | Purpose | Trigger |
|-------|---------|---------|
| `/pyasc-codegen-workflow` | Complete development workflow | **Mandatory: all kernel development tasks** |

### Development Assistance

| Skill | Purpose | Trigger |
|-------|---------|---------|
| `/pyasc-api-patterns` | API usage patterns and best practices | Before calling any pyasc API |
| `/pyasc-syntax-constraints` | Python syntax support/restrictions | When writing `@asc.jit` code |
| `/pyasc-docs-search` | Documentation and tutorial index | When local knowledge is insufficient |

### Build and Verify

| Skill | Purpose | Trigger |
|-------|---------|---------|
| `/pyasc-build-run-verify` | JIT build, run, and output verification | After implementation, before delivery |
| `/pyasc-env-check` | Environment check | Environment setup or verification |
| `/pyasc-code-review` | Code review against constraints | After implementation, before acceptance |
| `/pyasc-task-focus` | Task focus and attention management | Long tasks, multi-step development |

---

## Project directory structure

```
pyasc-kernel-dev-team/
├── kernels/              # Generated kernel workspace
│   └── {kernel_name}/    # Independent directory per kernel
│       ├── kernel.py     # Kernel implementation
│       ├── docs/         # Design documents
│       ├── test/         # Test data
│       └── README.md     # Kernel description
└── AGENTS.md             # This file
```

---

## Development resources

| Resource type | Path | Description |
|---------------|------|-------------|
| pyasc tutorials | `~/workspace/pyasc/python/tutorials/` | 5 tutorial examples (01_add through 05_matmul) |
| pyasc API docs | `~/workspace/pyasc/docs/python-api/` | API documentation |
| pyasc tests | `~/workspace/pyasc/python/test/` | Unit, kernel, and generalization tests |
| Architecture docs | `~/workspace/pyasc/docs/architecture_introduction.md` | JIT pipeline and module overview |
| Syntax support | `~/workspace/pyasc/docs/python_syntax_support.md` | Supported/unsupported syntax reference |
| Developer guide | `~/workspace/pyasc/docs/developer_guide.md` | Extension and contribution guide |

---

## API usage rules

> **All pyasc APIs must be grounded in official documentation. No guessing.**

**Mandatory restrictions**:
- **ALLOWED**: Basic vector APIs (`asc.add`, `asc.sub`, `asc.mul`, `asc.div`, `asc.data_copy`, etc.)
- **ALLOWED**: Core types (`asc.GlobalTensor`, `asc.LocalTensor`, `asc.GlobalAddress`)
- **ALLOWED**: Sync primitives (`asc.set_flag`, `asc.wait_flag`, `asc.HardEvent`)
- **ALLOWED**: Runtime (`asc.jit`, `asc.runtime.config`, `asc.lib.runtime`)
- **BANNED**: Any API not documented in pyasc source or API docs
- **BANNED**: Using unsupported Python syntax inside `@asc.jit` functions
