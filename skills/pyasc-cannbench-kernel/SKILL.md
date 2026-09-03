---
name: pyasc-cannbench-kernel
description: Generate, review, and locally qualify pyasc v2 operator modules for the vendored CANNBench contract. Use for CANNBench kernels, full benchmark coverage, exact-v2 compile checks, submission preparation, or investigation of CANNBench failures.
---

# pyasc v2 CANNBench kernel workflow

Use this skill only for the vendored benchmark under
`integrations/cannbench/`. Its compact module contract overrides the generic
standalone-kernel workflow.

## Current v2 snapshot

The current CANNBench campaign is pinned to
`compiler-team/pyasc@0a631f70968c3cb7c33ce45330a85768dd5a6f06`. In this
snapshot the tile API package was renamed from `asc2` to `asctile`. New
candidates must use `import asctile`, `@asctile.jit`, and the `asctile.*`
symbols. Treat `asc2` examples in historical artifacts as obsolete and do
not emit them for this campaign.

## Non-negotiable contract

- Treat `tasks/<op>/proto.yaml`, `golden.py`, and all 20 entries in
  `cases.yaml` as one indivisible specification.
- Produce one `candidate.py` with the exact top-level callable and defaults.
- Use `torch` only for allocation, metadata views, and launches. Numerical work
  belongs in JIT kernels. Prefer `@asctile.jit`; a low-level `@asc.jit` route
  must also prove that its C310 ABI contains no hidden FFTS argument.
- Import and call `ensure_npu_platform` from `._pyasc_runtime`.
- Target `Ascend950PR_9599`, at most 72 vector cores, and UB capacity 253952 B.
- For the pinned runtime use `asctile.global_tensor`, `asctile.copy_in`, and
  `asctile.copy_out`; use `real_shape` for tails. Do not substitute the older
  `tensor/load/store` spelling.
- Treat padded lanes as executed lanes: choose an explicit, operation-neutral
  `pad_value` for `real_shape` loads (especially `1` for divisors) so tail
  arithmetic does not create avoidable Inf/NaN exceptions.
- A production candidate normally uses bare `@asctile.jit` or an evidenced
  allocation option. At the pinned commit, direct AscTile option discovery is
  broken; use the integration's concrete-options adapter before relying on
  `reuse_alloc`, `static_alloc`, or `vf_fusion`. `always_compile=True` is a
  development aid, not a submission requirement.

## Required workflow

1. Read the three task files and summarize every dtype, rank, attribute, list
   arity, special value, and boundary shape. Do not optimize for one example.
   An upstream target test is a useful implementation reference, not evidence
   that its narrower dtype/shape/attribute domain implements this contract.
2. Read [operator patterns](references/operator-patterns.md), selecting only
   the target operator section.
3. Design host dispatch, kernel specializations, tails, dtype conversions, UB
   budget, and numerical behavior before writing code.
4. Implement the exact callable. Run `python3 -m py_compile candidate.py` and
   the worker static contract check.
5. Run the exact-v2 local compile gate for that operator as described in
   [local validation](references/local-validation.md). A compile failure is
   measured feedback; repair it and re-run all 20 cases. For low-level kernels,
   record `has_ffts_arg=false` on C310.
6. For numerically sensitive formulas, run the adversarial camodel smoke when
   the native runtime is available. It must exercise both ordinary and
   special-value routes using the task's dtype-specific tolerances. Treat a
   small all-finite smoke as insufficient for promotion.
7. Have a different available model review the task, design, candidate, and
   local report. Record both model IDs, OpenCode session IDs, loaded skill
   source hashes, prompt/design/candidate hashes, and gate results.

Never claim numerical correctness or performance from the local compile gate.
Only camodel execution can provide local numerical evidence; only CANNBench on
the real NPU is the acceptance/performance oracle. In particular, a VF/reuse
kernel that fits the static UB budget can still trigger a vector-core timeout
on a large case.

## Evidence labels

Use exactly one label for every conclusion:

- `verified-local-compile`: all task routes lower through pinned pyasc v2.
- `verified-camodel-smoke`: the stated representative routes executed and
  matched their golden; scope must list omitted dtypes/shapes/cases.
- `verified-camodel`: outputs were executed and compared to the task golden.
- `verified-cannbench`: official hardware result and case count are recorded.
- `suspected`: evidence is incomplete or the root cause is inferred.

Do not promote a locally qualified candidate into the canonical submission
package without numerical evidence. When a CANNBench run reports case-level
failures, feed those exact case IDs and failure classes into repair; a clean
compile matrix must never override measured numerical failures. Do not spend a
submission credit unless the user explicitly enables remote evaluation.

## Known limitations

Before using a workaround, consult
[pyasc v2 blockers](references/pyasc-v2-blockers.md). Preserve full case
coverage when a case exposes a compiler limit; never remove or shrink the case
to make a gate pass.
