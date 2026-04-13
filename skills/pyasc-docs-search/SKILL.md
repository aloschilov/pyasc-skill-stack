---
name: pyasc-docs-search
description: pyasc development resource index (local-first). Provides local API documentation, tutorial catalog, test examples, and architecture docs. Prioritize local pyasc source tree resources. Trigger — when searching for API usage, tutorial examples, or implementation references for pyasc kernel development.
---

# pyasc Development Resources

## Overview

This skill provides local-first documentation search for pyasc kernel development:
- **Local resources**: API docs, 5 tutorial examples, unit/kernel/generalization tests
- **pyasc source tree**: Complete Python API surface at `~/workspace/pyasc/python/asc/`

## Resource paths

### Local golden set (always accessible — use first)

| Resource type | Path | Description |
|---------------|------|-------------|
| Tutorials | `golden/tutorials/` | 5 golden tutorial kernels |
| API docs (basic) | `golden/docs/python-api/language/basic.md` | Basic API index (add, sub, mul, abs, data_copy, etc.) |
| API docs (core) | `golden/docs/python-api/language/core.md` | Core types (GlobalTensor, LocalTensor, etc.) |
| API docs (adv) | `golden/docs/python-api/language/adv.md` | Advanced API index (matmul, etc.) |
| API docs (fwk) | `golden/docs/python-api/language/fwk.md` | Framework API index (TPipe, TQue, etc.) |
| Generated API refs | `golden/docs/python-api/language/generated/` | 277 individual API reference pages |
| Lib API docs | `golden/docs/python-api/lib/` | Runtime library API docs |
| Architecture | `golden/docs/architecture_introduction.md` | JIT pipeline overview |
| Syntax support | `golden/docs/python_syntax_support.md` | Supported/unsupported syntax |
| Developer guide | `golden/docs/developer_guide.md` | Extension guide |

### External pyasc source (fallback — requires external_directory permission)

| Resource type | Path | Description |
|---------------|------|-------------|
| Tutorials source | `~/workspace/pyasc/python/tutorials/` | 5 tutorial examples |
| Unit tests | `~/workspace/pyasc/python/test/unit/` | Per-API unit tests |
| Kernel tests | `~/workspace/pyasc/python/test/kernels/` | End-to-end kernel tests |
| Generalization tests | `~/workspace/pyasc/python/test/generalization/` | Cross-API generalization tests |
| Python API source | `~/workspace/pyasc/python/asc/language/` | API implementation (basic, core, adv, fwk) |

## Search priority

```
1. Golden set (golden/tutorials/, golden/docs/) — ALWAYS CHECK FIRST
       | Not found
2. golden/docs/python-api/language/generated/ (individual API references)
       | Not found
3. ~/workspace/pyasc/python/tutorials/ (5 tutorials, external)
       | Not found
4. ~/workspace/pyasc/python/asc/language/ (API source code, external)
       | Not found
5. ~/workspace/pyasc/python/test/ (test examples, external)
```

## Tutorial catalog

| Tutorial | Golden path | External path | Description |
|----------|-------------|---------------|-------------|
| 01_add | `golden/tutorials/01_add.py` | `~/workspace/pyasc/python/tutorials/01_add/` | Manual sync vector add |
| 02_add_framework | `golden/tutorials/02_add_framework.py` | `~/workspace/pyasc/python/tutorials/02_add_framework/` | Framework-managed sync add |
| 03_matmul_mix | `golden/tutorials/03_matmul_mix.py` | `~/workspace/pyasc/python/tutorials/03_matmul_mix/` | MIX mode matmul (cube + vector) |
| 04_matmul_cube_only | `golden/tutorials/04_matmul_cube_only.py` | `~/workspace/pyasc/python/tutorials/04_matmul_cube_only/` | Pure cube mode matmul |
| 05_matmul_leakyrelu | `golden/tutorials/05_matmul_leakyrelu.py` | `~/workspace/pyasc/python/tutorials/05_matmul_leakyrelu/` | Matmul + LeakyReLU fusion |

### Quick API lookup

To find the API reference for a specific function (e.g., `asc.abs`):
```
Read: golden/docs/python-api/language/generated/asc.language.basic.abs.md
```

The naming pattern is: `golden/docs/python-api/language/generated/asc.language.{module}.{function}.md`

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
- "Search for usage examples of asc2.load/asc2.store in pyasc asc2 kernel tests"
- "Find how tiling is implemented with asc.ceildiv in asc2 kernels"
- "Search for asc2.tensor initialization patterns in pyasc test/kernels/asc2/"

## References

- [API Index](references/api-index.md)
- [Tutorial Catalog](references/tutorial-catalog.md)
- [Example Catalog](references/example-catalog.md)
