# Phase 3: Verification and Delivery

## Purpose

Final verification of the kernel and preparation for delivery.

## Prerequisites

- Phase 2 completed with acceptance score >= 8.5

## Process

> **TIME BUDGET**: Phase 3 should take 2-3 tool calls maximum.
> If runtime fails, record the error and move on. Do NOT debug the runtime.

### Step 1: Attempt runtime verification

Set up the simulator environment and run:

```bash
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
cd kernels/{name}
python3.10 kernel.py -r Model -v Ascend910B1
```

**If runtime fails**: Record the exact error message and proceed immediately to Step 2. Do NOT attempt to fix the runtime environment, explore simulator directories, or try alternative platforms.

### Step 2: Static verification (always do this)

Use Python `ast` module to verify the kernel:
- Valid Python syntax (parses without errors)
- `@asc.jit` decorator present
- No banned constructs (print, try/except, break, continue, lambda, import inside JIT)
- `set_flag`/`wait_flag` sync pairs present
- `data_copy` usage present
- `allclose` or numpy verification present in host code

### Step 3: Write verification.md

Write `kernels/{name}/docs/verification.md` with:

```markdown
# Verification Record

## Runtime verification
- Backend: Model
- Platform: Ascend910B1
- Status: PASS / FAIL / SKIP
- Output: (paste output or error message)

## Static verification
- [x/] Valid Python syntax
- [x/] @asc.jit decorator present
- [x/] No banned constructs
- [x/] set_flag/wait_flag sync pairs
- [x/] data_copy usage
- [x/] allclose verification in host code

## Limitations
(State any limitations, e.g. "Runtime verification skipped: CANN simulator not fully configured")
```

### Step 4: Delivery

Provide:
- `kernel.py` — complete, verified kernel implementation
- `docs/design.md` — design document
- `docs/self_review.md` — self-review
- `docs/acceptance_review.md` — acceptance review
- `docs/environment.json` — environment snapshot
- `docs/verification.md` — verification record

## CP-3 Exit Conditions

- [ ] Runtime execution attempted (or documented skip)
- [ ] Static verification completed
- [ ] verification.md written
- [ ] All deliverables present
- [ ] Limitations stated explicitly if runtime unavailable
