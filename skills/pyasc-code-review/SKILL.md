---
name: pyasc-code-review
description: pyasc asc2 kernel code review skill. Reviews kernel code against pyasc syntax constraints, asc2 API correctness, and Python secure coding guidelines using hypothesis testing methodology. Trigger — when reviewing kernel code for correctness, during acceptance review in Phase 2, or when the user provides code for review.
---

# pyasc Code Review (asc2)

## Core Principles

1. **Syntax compliance** — all code inside `@asc2.jit` must use only supported Python syntax
2. **asc2 API correctness** — all asc2 API calls must match documented signatures and patterns
3. **Hypothesis testing driven** — systematically collect evidence to determine risks
4. **Auditable** — the entire review is logged with an evidence chain

## Call Interface

### Required parameters

**Parameter 1: Code snippet**
- The kernel code to review (kernel.py or code block)

**Parameter 2: Review focus**
- Specify the type of issues to check
- Examples: "check syntax constraints", "verify asc2 API usage", "check tiling correctness"

**Parameter 3: Specification reference** (optional)
- Specific specification file path
- Default: pyasc syntax constraints + Python secure coding

### Parameter verification

If required parameters are missing:
1. Inform the user which parameters are missing
2. Do not execute review
3. Prompt for complete parameters

## Review Checklist for pyasc asc2 Kernels

### Mandatory checks

| Check | What to verify | Reference |
|-------|---------------|-----------|
| **Syntax compliance** | All constructs inside `@asc2.jit` are in the supported set | `pyasc-syntax-constraints` |
| **No unsupported constructs** | No `print`, `break`, `continue`, `lambda`, `try/except`, nested functions, `global`, `import` inside JIT | `pyasc-syntax-constraints` |
| **`@asc2.jit` correctness** | Kernel decorated with `@asc2.jit(always_compile=True)`; device functions separated; no class methods | `pyasc-api-patterns` |
| **Type correctness** | Kernel params use supported types; `ConstExpr` for tile_size/tile_per_block | `pyasc-syntax-constraints` |
| **asc2.load/asc2.store** | Tiles loaded via `asc2.load()` and written via `asc2.store()` — NOT manual `asc.data_copy` | `pyasc-api-patterns` |
| **asc2.tensor usage** | Global memory wrapped via `asc2.tensor(ptr, [shape])` | `pyasc-api-patterns` |
| **asc2.range usage** | Tile loops use `asc2.range()`, NOT `range()` | `pyasc-syntax-constraints` |
| **Launch syntax** | `kernel[core_num](...)` — no stream argument | `pyasc-api-patterns` |
| **Output verification** | `np.testing.assert_allclose` present (numpy only; no torch/scipy) | `pyasc-build-run-verify` |
| **Variable scoping** | No use of variables only defined inside one `if` branch | `pyasc-syntax-constraints` |

### Red flags — asc v1 API in asc2 kernel

| Red flag | What it means | Fix |
|----------|---------------|-----|
| `asc.GlobalTensor()` | Using v1 manual memory | Use `asc2.tensor(ptr, [shape])` |
| `asc.LocalTensor(...)` | Using v1 local buffers | Use `asc2.load()` / `asc2.store()` |
| `asc.data_copy(...)` | Using v1 DMA | Use `asc2.load()` / `asc2.store()` |
| `asc.set_flag(...)` / `asc.wait_flag(...)` | Using v1 sync | Remove — asc2 handles sync automatically |
| `asc.TPosition.*` | Using v1 buffer positions | Remove — not needed in asc2 |
| `BUFFER_NUM` | Using v1 double-buffering | Remove — asc2 handles this automatically |
| `kernel[n, stream](...)` | Using v1 launch syntax | Use `kernel[n](...)` |

### Python secure coding checks

| Check | What to verify |
|-------|---------------|
| Input validation | Kernel parameters validated before launch |
| No hardcoded secrets | No credentials or paths hardcoded |
| Resource cleanup | Devices properly managed |
| Numeric safety | No division by zero, overflow-safe arithmetic |

## Review Process (Hypothesis Testing)

### Phase 1: Preparation

1. Verify required parameters are complete
2. Read the pyasc syntax constraints and asc2 API patterns
3. Identify the code to review

### Phase 2: Hypothesis Testing

**Step 1: Segment code** — divide into: kernel function, device functions, launch function, verification

**Step 2: For each segment, establish hypothesis**
- H0: This code segment is correct and safe
- H1: This code segment has issues
- Confidence: 0%

**Step 3: Evidence collection**

| Evidence Type | Scoring |
|---------------|---------|
| Unsupported syntax used inside `@asc2.jit` | +50% |
| asc v1 API used instead of asc2 | +45% |
| Missing `asc2.load`/`asc2.store` (manual memory mgmt) | +35% |
| API signature mismatch | +40% |
| Missing output verification | +30% |
| Variable scoping violation | +25% |
| Python secure coding violation | +20% |
| Using `range()` instead of `asc2.range()` | +30% |

**Step 4: Verify evidence** — check if the issue is defended elsewhere

**Step 5: Decision** — confidence > 60% = report as issue

### Phase 3: Report

Generate report with:
1. Review category (syntax, API, memory, verification, security)
2. Risk points with line numbers and code snippets
3. Evidence chain
4. Suggested fixes
5. Overall score (10-point scale)

## References

- [Python Secure Coding](references/PythonSecureCoding.md)
- [pyasc Review Checklist](references/pyasc-review-checklist.md)
