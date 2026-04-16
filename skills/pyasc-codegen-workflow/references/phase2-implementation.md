# Phase 2: Kernel Implementation

## Purpose

Implement the kernel based on the Phase 1 design, review it, and verify output correctness.

## Prerequisites

- Phase 1 completed with score >= 8.5
- `docs/design.md` exists

## Process

### Task 1: Implementation + Self-Review

1. **Implement kernel.py** based on design.md
   - Use `templates/kernel-template.py` as starting point
   - Follow the API selections from the design
   - Implement proper sync flags
   - Add output verification

2. **Self-review checklist**:
   - [ ] All syntax inside `@asc.jit` is supported
   - [ ] No unsupported constructs (print, break, continue, lambda, etc.)
   - [ ] Kernel function does not return a value
   - [ ] Proper sync: set_flag/wait_flag between pipeline stages
   - [ ] Output verification with np.testing.assert_allclose (numpy only)
   - [ ] Code matches the design document

3. **Run verification**:
   ```bash
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
   python3.10 kernel.py -r Model -v Ascend910B1
   ```

4. **Return report** with "Review Code Result" field

### Task 2: Acceptance Review

1. **Verify self-review was done**: Check "Review Code Result" field exists
2. **Code review**: Use `pyasc-code-review` skill
   - Syntax compliance check
   - API correctness check
   - Sync flag correctness
   - Output verification present and correct
3. **Re-run verification**: Execute kernel.py again
4. **Rate** on 10-point scale

### Acceptance checklist

| Item | Check |
|------|-------|
| `@asc.jit` decoration correct | Kernel vs device function |
| Supported syntax only | No unsupported constructs inside JIT |
| Proper sync flags | MTE2_V, V_MTE3, MTE3_MTE2 as needed |
| Output verification | np.testing.assert_allclose (numpy only) |
| Design consistency | Implementation matches design.md |
| No hardcoded values | Parameterized where appropriate |

## CP-2 Exit Conditions

- [ ] 2 Task records (implementation + acceptance)
- [ ] Acceptance score >= 8.5
- [ ] kernel.py runs without error
- [ ] Output verification passes
- [ ] No syntax constraint violations
