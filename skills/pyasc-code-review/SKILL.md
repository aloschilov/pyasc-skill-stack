---
name: pyasc-code-review
description: pyasc kernel code review skill. Reviews kernel code against pyasc syntax constraints, API correctness, and Python secure coding guidelines using hypothesis testing methodology. Trigger — when reviewing kernel code for correctness, during acceptance review in Phase 2, or when the user provides code for review.
---

# pyasc Code Review

## Core Principles

1. **Syntax compliance** — all code inside `@asc.jit` must use only supported Python syntax
2. **API correctness** — all pyasc API calls must match documented signatures and constraints
3. **Hypothesis testing driven** — systematically collect evidence to determine risks
4. **Auditable** — the entire review is logged with an evidence chain

## Call Interface

### Required parameters

**Parameter 1: Code snippet**
- The kernel code to review (kernel.py or code block)

**Parameter 2: Review focus**
- Specify the type of issues to check
- Examples: "check syntax constraints", "verify API usage", "check sync correctness"

**Parameter 3: Specification reference** (optional)
- Specific specification file path
- Default: pyasc syntax constraints + Python secure coding

### Parameter verification

If required parameters are missing:
1. Inform the user which parameters are missing
2. Do not execute review
3. Prompt for complete parameters

## Review Checklist for pyasc Kernels

### Mandatory checks

| Check | What to verify | Reference |
|-------|---------------|-----------|
| **Syntax compliance** | All constructs inside `@asc.jit` are in the supported set | `pyasc-syntax-constraints` |
| **No unsupported constructs** | No `print`, `break`, `continue`, `lambda`, `try/except`, nested functions, `global`, `import` inside JIT | `pyasc-syntax-constraints` |
| **`@asc.jit` correctness** | Kernel decorated properly; device functions separated; no class methods | `pyasc-api-patterns` |
| **Type correctness** | Kernel params use supported types; `ConstExpr` for compile-time values | `pyasc-syntax-constraints` |
| **Sync flags** | Proper `set_flag`/`wait_flag` between pipeline stages (MTE2_V, V_MTE3, MTE3_MTE2) | `pyasc-api-patterns` |
| **Output verification** | `torch.allclose` or numpy comparison present | `pyasc-build-run-verify` |
| **Variable scoping** | No use of variables only defined inside one `if` branch | `pyasc-syntax-constraints` |

### Python secure coding checks

| Check | What to verify |
|-------|---------------|
| Input validation | Kernel parameters validated before launch |
| No hardcoded secrets | No credentials or paths hardcoded |
| Resource cleanup | Streams and devices properly managed |
| Numeric safety | No division by zero, overflow-safe arithmetic |

## Review Process (Hypothesis Testing)

### Phase 1: Preparation

1. Verify required parameters are complete
2. Read the pyasc syntax constraints and API patterns
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
| Unsupported syntax used inside `@asc.jit` | +50% |
| API signature mismatch | +40% |
| Missing sync flags between pipeline stages | +35% |
| Missing output verification | +30% |
| Variable scoping violation | +25% |
| Python secure coding violation | +20% |

**Step 4: Verify evidence** — check if the issue is defended elsewhere

**Step 5: Decision** — confidence > 60% = report as issue

### Phase 3: Report

Generate report with:
1. Review category (syntax, API, sync, verification, security)
2. Risk points with line numbers and code snippets
3. Evidence chain
4. Suggested fixes
5. Overall score (10-point scale)

## References

- [Python Secure Coding](references/PythonSecureCoding.md)
- [pyasc Review Checklist](references/pyasc-review-checklist.md)
