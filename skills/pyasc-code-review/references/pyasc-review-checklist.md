# pyasc Kernel Review Checklist

## Category 1: Syntax compliance

| # | Check | Severity |
|---|-------|----------|
| 1.1 | All code inside `@asc.jit` uses supported syntax only | Critical |
| 1.2 | No `print`, `break`, `continue`, `lambda`, `yield` | Critical |
| 1.3 | No `try/except`, `raise`, `with open` | Critical |
| 1.4 | No `import`/`from...import` inside JIT | Critical |
| 1.5 | No nested functions, `global`, `nonlocal` | Critical |
| 1.6 | No class methods as kernels | Critical |
| 1.7 | No `async def`/`async with` | Critical |
| 1.8 | Variables initialized before conditional use | High |

## Category 2: API correctness

| # | Check | Severity |
|---|-------|----------|
| 2.1 | All APIs exist in documented pyasc surface | Critical |
| 2.2 | API parameters match documented signatures | High |
| 2.3 | Tensor positions correct (VECIN for input, VECOUT for output) | High |
| 2.4 | Data copy direction correct (GM->UB or UB->GM) | High |
| 2.5 | Element counts and offsets are correct | High |

## Category 3: JIT decoration

| # | Check | Severity |
|---|-------|----------|
| 3.1 | Kernel function has `@asc.jit` decorator | Critical |
| 3.2 | Kernel function does not return a value | Critical |
| 3.3 | Device helper functions return top-level only | High |
| 3.4 | Compile options are valid if specified | Medium |

## Category 4: Synchronization

| # | Check | Severity |
|---|-------|----------|
| 4.1 | `set_flag`/`wait_flag` present between pipeline stages | Critical |
| 4.2 | Correct events used (MTE2_V, V_MTE3, MTE3_MTE2) | High |
| 4.3 | Buffer IDs match between set and wait | High |
| 4.4 | No missing sync between copy and compute | Critical |

## Category 5: Verification

| # | Check | Severity |
|---|-------|----------|
| 5.1 | Output verification present | High |
| 5.2 | Reference computation is correct | High |
| 5.3 | Tolerance is reasonable for dtype | Medium |
| 5.4 | Multiple test shapes if applicable | Low |

## Category 6: Code quality

| # | Check | Severity |
|---|-------|----------|
| 6.1 | Implementation matches design document | Medium |
| 6.2 | No unnecessary hardcoded magic numbers | Medium |
| 6.3 | Reasonable naming conventions | Low |
| 6.4 | Launch function properly configured | Medium |

## Scoring

| Score | Criteria |
|-------|----------|
| 9-10 | No critical/high issues; clean code |
| 8-8.9 | No critical issues; minor high issues addressed |
| 7-7.9 | No critical issues; some high issues remain |
| < 7 | Critical issues present; must fix before acceptance |
