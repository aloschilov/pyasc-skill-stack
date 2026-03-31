---
name: pyasc-codegen-workflow
description: pyasc kernel development standard workflow. Contains 4 phases with checkpoints — environment preparation, design, implementation with review, and verification. Trigger — user requests kernel development using pyasc. Applicable to pyasc JIT kernels (Python scripts), not Ascend C direct mode.
---

# pyasc kernel development workflow

## pyasc JIT kernel requirements

> **This skill is for pyasc JIT kernels**: the final product is a runnable Python script using `@asc.jit`, not a C++ executable or `.so` library.

**Phase 2 exit requirements**:
- There is a kernel function decorated with `@asc.jit`
- There is a launch function using `kernel[core_num, stream](...)`
- There is a host-side driver (`if __name__ == "__main__"`) or test harness
- There is output verification (`torch.allclose` or numpy comparison)
- The kernel runs successfully (on NPU or Model backend)

**Golden reference**: `~/workspace/pyasc/python/tutorials/01_add/add.py`

---

> **Forced Workflow**: Phase 0 -> Phase 1 -> Phase 2 -> Phase 3
>
> **Forbidden**: Write code directly, skip design, skip acceptance review

---

## Process Integrity Checklist

| Phase | Required Task | Count | Why |
|-------|---------------|-------|-----|
| Phase 0 | Initialize project + verify environment | 1 | Ensure environment is ready and directory structure is complete |
| Phase 1 | Design + evaluation | 2 | Evaluation finds design flaws early |
| Phase 2 | Implementation + review + acceptance | 2/branch | Acceptance catches syntax violations and API misuse |
| Phase 3 | Verification | On demand | Ensure kernel produces correct output |

**Common mistakes**:
- Phase 0 skips environment check -> pyasc not installed, CANN missing
- Phase 2 uses unsupported syntax inside `@asc.jit` -> JIT compilation fails
- Skipping acceptance review -> syntax constraint violations missed

---

## Quick Checklist

- [ ] Phase 0: Project directory + environment.json saved? (CP-0)
- [ ] Phase 1: Design document + rating >= 8.5? (CP-1)
- [ ] Phase 2: Acceptance review report + verification passed? (CP-2)
- [ ] Phase 3: Test pass record? (CP-3)

---

## Process Overview

```
Phase 0: Environment preparation         [1 Task]
    |
Phase 1: Design and API selection        [2 Tasks]
    |
Phase 2: Kernel implementation           [2 Tasks/branch]
    |
Phase 3: Verification and delivery
```

---

## Force checkpoints

| Checkpoint | Timing | Check content | Passing criteria |
|------------|--------|---------------|------------------|
| **CP-0** | After Phase 0 | Project directory + environment.json | Directory exists + file complete |
| CP-1 | After Phase 1 | design.md + rating | 2 Task records + rating >= 8.5 |
| CP-2 | After Phase 2 per branch | Acceptance report + verification | 2 Task records + score >= 8.5 |
| CP-3 | After Phase 3 | Verification record | Tests passed |

---

## Phase 0: Environment preparation

> **PROHIBITED**: Skip Phase 0 and go directly to design

### Step 1: Initialize kernel project

```bash
bash <skills_path>/pyasc-codegen-workflow/scripts/init_kernel_project.sh {kernel_name}
```

**Created directory structure**:
```
kernels/{kernel_name}/
├── docs/           # Design documents
├── test/           # Test data and scripts
├── kernel.py       # Kernel implementation (created in Phase 2)
└── README.md       # Kernel description
```

### Step 2: Verify environment and save results

```bash
bash <skills_path>/pyasc-codegen-workflow/scripts/verify_environment.sh {kernel_name}
```

**Output**: `kernels/{kernel_name}/docs/environment.json`

### CP-0 Exit Conditions

- [ ] Project directory created (`kernels/{kernel_name}/`)
- [ ] Subdirectories created (docs/, test/)
- [ ] **environment.json saved** (including Python version, pyasc version, CANN version, backend)

**Detailed guide**: [references/phase0-environment.md](references/phase0-environment.md)

---

## Phase 1: Design and API selection

> **Prerequisites**: Phase 0 completed, project directory and environment.json exist

```
Main Agent
 |-- Task: Design -> Steps 1-5 -> Design document
 |-- Task: Evaluation -> Score >= 8.5
```

### Design steps

1. **Understand the operation**: What mathematical/logical operation does this kernel perform?
2. **Retrieve documentation**: Use `pyasc-docs-search` to find relevant tutorials and API docs
3. **Select APIs**: Choose from `asc.language.basic`, `asc.language.core`, etc.
4. **Check syntax constraints**: Use `pyasc-syntax-constraints` to verify all constructs are supported
5. **Write design document**: Use [templates/design-template.md](templates/design-template.md)

**Detailed guide**: [references/phase1-design.md](references/phase1-design.md)

---

## Phase 2: Kernel implementation

### Execution process

```
for each branch:
    |-- Task 1: Implementation + self-review
    |   -> Implement kernel.py
    |   -> Self-check (syntax constraints, API correctness, sync flags)
    |   -> Run verification
    |   -> Return report
    |
    |-- Task 2: Acceptance review
    |   -> Code review (pyasc-code-review)
    |   -> Re-run verification
    |   -> Rating (10-point scale)

    Rating >= 8.5? -> Next phase / Fix and re-accept
```

### Phase 2 exit conditions (all must be met)

| Condition | Check method | When not met |
|-----------|-------------|--------------|
| 2 Tasks executed | Check Task records | Execute acceptance Task |
| Acceptance score returned | Check "Total Score" field | Re-execute acceptance |
| Score >= 8.5 | Check score value | Fix and re-accept |
| Kernel runs without error | Run kernel.py | Fix implementation |
| Output verified | Check torch.allclose / numpy | Fix computation |

**Key acceptance checklist items**:
1. All syntax inside `@asc.jit` is in the supported set
2. `@asc.jit` decorator used correctly (kernel vs device function)
3. Proper sync flags (`set_flag`/`wait_flag`) between pipelines
4. Output verification present and passing
5. No unsupported constructs (no `print`, `break`, `continue`, `lambda`, etc.)

**Detailed guide**: [references/phase2-implementation.md](references/phase2-implementation.md)

---

## Phase 3: Verification and delivery

| Step | Content |
|------|---------|
| Verification | Run kernel on Model backend and/or NPU; verify output correctness |
| Limitation statement | If NPU unavailable, state limitation explicitly |
| Delivery | Provide kernel.py + design.md + verification results |

**Detailed guide**: [references/phase3-testing.md](references/phase3-testing.md)

---

## Quick index

### Templates
- Design document: [templates/design-template.md](templates/design-template.md)
- Kernel template: [templates/kernel-template.py](templates/kernel-template.py)

### Scripts
- `init_kernel_project.sh` — initialize kernel directory
- `verify_environment.sh` — environment verification

### Related skills
- `pyasc-syntax-constraints` — Python syntax support reference
- `pyasc-api-patterns` — API best practices
- `pyasc-docs-search` — Documentation search
- `pyasc-build-run-verify` — Build and verify
- `pyasc-code-review` — Code review
