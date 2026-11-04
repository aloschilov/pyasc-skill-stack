---
name: pyasc-codegen-workflow
description: End-to-end current-v2 asctile kernel design, implementation, independent review, validation, and provenance workflow. Use for multi-step kernel generation or tuning.
---

# Current-v2 kernel generation workflow

This workflow produces a reviewable kernel and evidence chain. Load
`pyasc-api-patterns` and `pyasc-syntax-constraints` before implementation;
load `pyasc-code-review` for independent review and
`pyasc-build-run-verify` for validation. For CANNBench also load
`pyasc-cannbench-kernel`, whose task contract takes precedence.

## Phase 0 — pin and inspect

1. Read `reference/images/ENV.json` and `reference/asctile/PIN.json` from the
   skill-stack checkout. Reuse the pinned image for the repository workflow.
2. Record the selected runtime, backend and platform. If using another runtime,
   identify its exact source revision; do not silently mix versions.
3. Read the nearest shipped target test and the operator contract. Consult the
   matching upstream source only when the shipped references are insufficient.
4. If verification runs in Docker, do not search for or install a host runtime.
   Use the existing qualification command from the repository README.

## Phase 1 — design

Write `design.md` covering:

- semantics and exact public signature
- complete shape/dtype/attribute/special-value matrix
- host dispatch and JIT specializations
- `global_tensor` descriptors and `copy_in/copy_out` ranks
- tiling, tails, neutral padding, core count, and UB budget
- numerical precision and reduction strategy
- fallback routes that remain asctile numerical kernels
- static, compile, camodel, and real-NPU validation plan

Reject the design if any case relies on a host numerical fallback or an API
not present in the pinned source.

## Phase 2 — implementation

Start from `templates/kernel-template.py` or the nearest current target test.
Use `import asc` for shared types/runtime configuration and
`from asc.experimental import asctile` for the tile API (legacy source builds
expose it as a top-level `import asctile`; the template handles both). The
standard data path is:

```text
global_tensor -> copy_in -> asctile compute -> copy_out
```

Run `python3 -m py_compile` immediately. Then self-review every host route,
local shape, dtype cast, tail, and output shape. Perform acceptance review in a
fresh session/worker; the same configured model is allowed. Record actual model
and session IDs: a fresh session is not an independent-model review. Preserve
the reviewed candidate hash and findings. A repair creates a new candidate that
must be reviewed and qualified again. If only self-review ran, record that
limitation; do not silently count it as the separate review phase.

These phases require a caller to arrange the handoffs. A compile/numerical CI
job does not by itself execute the review or tuning phases.

For worker provenance record:

- model and OpenCode session for each phase
- exact loaded skill names/directories and hashes
- prompt, design, and candidate hashes
- source commit and runtime artifact hash
- each gate result and its limitations

## Phase 3 — validation

Run gates in this order:

1. syntax and static anti-cheat/contract check
2. exact-v2 dispatch + compile/lowering for every case and specialization
3. camodel numerical smoke including adversarial routes
4. full feasible camodel matrix where practical
5. private CANNBench submission on real NPU

Compile-only/QEMU is a valid fast gate, not an execution result. Repair from
measured failure signatures and rerun the entire affected operator matrix.
Never delete or shrink a failing benchmark case.

## Phase 4 — tuning

Tune only after correctness qualification. Change one measurable lever at a
time: tile size, core occupancy, buffering, layout, cast count, temporary
lifetime, or specialization threshold. CaModel can reject bad variants and
find compiler limitations; CANNBench is the final performance oracle. Keep the
incumbent unless a new candidate preserves all cases and improves the official
measurement beyond noise.

## Deliverables

- candidate module/kernel
- design and independent review
- source/runtime/model provenance
- per-case compile and numerical evidence
- CANNBench submission/job links and result JSON when available
- documented pyasc blockers with reproducer and affected cases

## Resources

- [Request workflow and gate taxonomy](references/workflow.md)
- [Phase 0](references/phase0-environment.md)
- [Phase 1](references/phase1-design.md)
- [Phase 2](references/phase2-implementation.md)
- [Phase 3](references/phase3-testing.md)
- [Review checklist](references/code-review-checklist.md)
- [Kernel template](templates/kernel-template.py)
- [Design template](templates/design-template.md)
