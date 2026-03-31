# Phase 1: Design and API Selection

## Purpose

Design the kernel implementation before writing code. Retrieve documentation, select APIs, and verify syntax compliance.

## Prerequisites

- Phase 0 completed
- `kernels/{kernel_name}/docs/environment.json` exists

## Process

### Task 1: Design

1. **Understand the operation**: Define the mathematical/logical operation
2. **Retrieve documentation**: Use `pyasc-docs-search` to find relevant tutorials and API docs
3. **Select APIs**: Choose from `asc.language.basic`, `asc.language.core`, etc.
4. **Plan multi-core strategy**: How to distribute work across cores
5. **Plan buffer strategy**: Single or double buffering, tile sizes
6. **Plan sync strategy**: Which pipeline events to use
7. **Check syntax constraints**: Verify all planned constructs are supported
8. **Write design document**: Fill in `templates/design-template.md`, save as `docs/design.md`

### Task 2: Evaluation

Review the design document and rate it on a 10-point scale:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Completeness | 20% | All sections filled, APIs identified |
| API correctness | 25% | All APIs exist and are used correctly |
| Syntax compliance | 25% | All constructs are in the supported set |
| Verification plan | 15% | Clear plan for output verification |
| Clarity | 15% | Design is clear and implementable |

**Passing score**: >= 8.5

## CP-1 Exit Conditions

- [ ] `docs/design.md` exists and is complete
- [ ] 2 Task records (design + evaluation)
- [ ] Evaluation score >= 8.5
- [ ] All APIs referenced are documented in pyasc
- [ ] All syntax constructs are in the supported set
