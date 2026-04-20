---
description: pyasc kernel development assistant using the asc2 tile-based API. Provides full-process guidance for developing custom Ascend operators using pyasc (Python-surface, NumPy-like tile operations, JIT-compiled to Ascend C).
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
  read: allow
  edit: allow
  bash: allow
  glob: allow
  grep: allow
  list: allow
  skill: allow
  task: allow
---

# AGENTS.md

This assistant is an agent designed to guide pyasc kernel development on Huawei Ascend AI processors using the **asc2** high-level tile-based API.

## Project Overview

This project provides an AI-assisted workflow for developing custom operators using pyasc's asc2 API — a NumPy-like tile language that compiles to Ascend C via JIT and runs on Ascend NPUs (Atlas A2/A3).

### Core functions

- Develop custom Ascend operators using the asc2 tile-based API
- Provide complete development, build, and verification workflow support
- Follow pyasc syntax constraints and asc2 API best practices

---

## Core Principles

> Strictly follow these principles to avoid common pyasc development problems

1. **Retrieve documentation before generating code**
   - Search pyasc tutorials and API docs in the golden set and pyasc source tree
   - Consult official asc2 kernel examples before writing any kernel code
   - Forbidden: generating code from memory without checking current API surface

2. **Respect pyasc syntax constraints**
   - Only use supported Python syntax inside `@asc2.jit` functions
   - Check `pyasc-syntax-constraints` before using any construct
   - Stop immediately when encountering unsupported syntax and find alternatives

3. **Progressive verification**
   - Start with the simplest possible kernel (e.g., vector add)
   - Verify with `np.testing.assert_allclose` before expanding (numpy only; no torch/scipy)
   - Use `Model` backend when NPU hardware is unavailable

4. **No unsupported shortcuts**
   - Do not use `print`, `try/except`, `break`, `continue`, `lambda` inside JIT functions
   - Do not use nested functions or class methods as kernels
   - Do not use `global` or `nonlocal` statements

5. **Use documented asc2 APIs only**
   - Use only APIs from `asc2` (tensor, load, store, range, block_idx, etc.)
   - Use `asc.GlobalAddress`, `asc.ConstExpr[int]` for kernel parameter types
   - Use `asc.ceildiv()` for tiling math
   - Forbidden: guessing API signatures or inventing undocumented APIs
   - Forbidden: using asc v1 APIs (`asc.GlobalTensor`, `asc.LocalTensor`, `asc.data_copy`, `asc.set_flag`/`asc.wait_flag`)

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
| `/pyasc-api-patterns` | asc2 API usage patterns and best practices | Before calling any pyasc API |
| `/pyasc-syntax-constraints` | Python syntax support/restrictions | When writing `@asc2.jit` code |
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

> **Priority**: Always check the local golden set first. Use external pyasc source tree only as a fallback.

### Local golden set (always accessible, inside this project)

| Resource type | Path | Description |
|---------------|------|-------------|
| Golden kernels | `golden/kernels/` | asc2 kernels: abs_f16, gelu_f16, gelu_f32, leaky_relu_f16, reduce_sum_f16, reduce_sum_f32, softmax_f16 |
| Golden tutorials | `golden/tutorials/` | asc2 tutorials: 01_add, 02_add_framework, 03-05_matmul variants |
| API docs (language) | `golden/docs/python-api/language/` | basic, core, adv, fwk API indexes |
| API docs (generated) | `golden/docs/python-api/language/generated/` | Individual API reference pages |
| API docs (lib) | `golden/docs/python-api/lib/` | Runtime library API docs |
| Architecture docs | `golden/docs/architecture_introduction.md` | JIT pipeline and module overview |
| Syntax support | `golden/docs/python_syntax_support.md` | Supported/unsupported syntax reference |

### External pyasc source tree (optional, not available in CI)

> These paths exist only on local developer machines. If unavailable, rely on the local golden set above.

| Resource type | Path | Description |
|---------------|------|-------------|
| asc2 kernel tests | `~/workspace/pyasc/python/test/kernels/asc2/` | Canonical asc2 kernel examples |
| pyasc tutorials | `~/workspace/pyasc/python/tutorials/` | Tutorial examples |
| pyasc API docs | `~/workspace/pyasc/docs/python-api/` | API documentation |

---

## API usage rules

> **All pyasc APIs must be grounded in official documentation. No guessing.**

**Mandatory restrictions**:
- **ALLOWED (asc2)**: `asc2.tensor`, `asc2.load`, `asc2.store`, `asc2.range`, `asc2.block_idx`, `asc2.block_num`
- **ALLOWED (asc2 ops)**: `asc2.abs`, `asc2.exp`, `asc2.log`, `asc2.sqrt`, `asc2.relu`, `asc2.erf`, `asc2.sin`, `asc2.cos`, `asc2.where`, `asc2.reduce_sum`, `asc2.reduce_max`, `asc2.softmax`, `asc2.matmul`, `asc2.full`
- **ALLOWED (asc2 ops)**: Tile arithmetic via operators: `x + y`, `x - y`, `x * y`, `x / y`, `-x`
- **ALLOWED (asc2 ops)**: Tile method reductions: `x.sum()`, `x.max()`, `x.min()`
- **ALLOWED (shared)**: `asc.GlobalAddress`, `asc.ConstExpr[int]`, `asc.ceildiv`
- **ALLOWED (runtime)**: `asc2.jit`, `asc.runtime.config`
- **BANNED**: asc v1 APIs inside asc2 kernels (`asc.GlobalTensor`, `asc.LocalTensor`, `asc.data_copy`, `asc.set_flag`, `asc.wait_flag`, `asc.TPosition`)
- **BANNED**: Any API not documented in pyasc source or API docs
- **BANNED**: Using unsupported Python syntax inside `@asc2.jit` functions
