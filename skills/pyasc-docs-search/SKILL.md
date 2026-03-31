---
name: pyasc-docs-search
description: pyasc development resource index (local-first). Provides local API documentation, tutorial catalog, test examples, and architecture docs. Prioritize local pyasc source tree resources. Trigger — when searching for API usage, tutorial examples, or implementation references for pyasc kernel development.
---

# pyasc Development Resources

## Overview

This skill provides local-first documentation search for pyasc kernel development:
- **Local resources**: API docs, 5 tutorial examples, unit/kernel/generalization tests
- **pyasc source tree**: Complete Python API surface at `~/workspace/pyasc/python/asc/`

## Official resource paths

| Resource type | Path | Description |
|---------------|------|-------------|
| Tutorials | `~/workspace/pyasc/python/tutorials/` | 5 tutorial examples |
| API documentation | `~/workspace/pyasc/docs/python-api/` | Generated API docs |
| Architecture | `~/workspace/pyasc/docs/architecture_introduction.md` | JIT pipeline overview |
| Syntax support | `~/workspace/pyasc/docs/python_syntax_support.md` | Supported/unsupported syntax |
| Developer guide | `~/workspace/pyasc/docs/developer_guide.md` | Extension guide |
| Unit tests | `~/workspace/pyasc/python/test/unit/` | Per-API unit tests |
| Kernel tests | `~/workspace/pyasc/python/test/kernels/` | End-to-end kernel tests |
| Generalization tests | `~/workspace/pyasc/python/test/generalization/` | Cross-API generalization tests |
| Python API source | `~/workspace/pyasc/python/asc/language/` | API implementation (basic, core, adv, fwk) |

## Search priority

```
1. Golden set (golden/tutorials/, golden/docs/)
       | Not found
2. ~/workspace/pyasc/python/tutorials/ (5 tutorials)
       | Not found
3. ~/workspace/pyasc/docs/python-api/ (API documentation)
       | Not found
4. ~/workspace/pyasc/python/asc/language/ (API source code)
       | Not found
5. ~/workspace/pyasc/python/test/ (test examples)
```

## Tutorial catalog

| Tutorial | Path | Description |
|----------|------|-------------|
| 01_add | `~/workspace/pyasc/python/tutorials/01_add/` | Manual sync vector add |
| 02_add_framework | `~/workspace/pyasc/python/tutorials/02_add_framework/` | Framework-managed sync add |
| 03_matmul_mix | `~/workspace/pyasc/python/tutorials/03_matmul_mix/` | MIX mode matmul (cube + vector) |
| 04_matmul_cube_only | `~/workspace/pyasc/python/tutorials/04_matmul_cube_only/` | Pure cube mode matmul |
| 05_matmul_leakyrelu | `~/workspace/pyasc/python/tutorials/05_matmul_leakyrelu/` | Matmul + LeakyReLU fusion |

## API module index

| Module | Path | Content |
|--------|------|---------|
| `asc.language.basic` | `python/asc/language/basic/` | Basic vector APIs (add, sub, mul, div, data_copy, etc.) |
| `asc.language.core` | `python/asc/language/core/` | Core types (GlobalTensor, LocalTensor, enums, TPosition) |
| `asc.language.adv` | `python/asc/language/adv/` | Advanced APIs (matmul, etc.) |
| `asc.language.fwk` | `python/asc/language/fwk/` | Framework APIs (TPipe, TQue, etc.) |
| `asc.lib.host` | `python/asc/lib/host/` | Host-side helpers |
| `asc.lib.runtime` | `python/asc/lib/runtime/` | ACL runtime wrappers |
| `asc.runtime.config` | `python/asc/runtime/` | Backend/Platform configuration |

## Explore Agent usage

**When to use**:
1. Find API documentation or usage examples
2. Understand how a specific pyasc API works
3. Search for tutorial implementations similar to the target kernel
4. Find solutions when encountering JIT compilation errors

**How to use**:
```
Use Task tool, subagent_type=Explore
Provide clear search objectives and scope
```

**Example prompts**:
- "Search for usage examples of asc.data_copy in pyasc tutorials"
- "Find how manual sync is implemented with set_flag/wait_flag"
- "Search for LocalTensor initialization patterns in pyasc tests"

## References

- [API Index](references/api-index.md)
- [Tutorial Catalog](references/tutorial-catalog.md)
- [Example Catalog](references/example-catalog.md)
