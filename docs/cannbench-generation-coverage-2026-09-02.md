# CANNBench full-coverage generation status — 2026-09-02

## Outcome

OpenCode workers generated a complete nine-operator CANNBench L1 bundle using
skills from this checkout. The final exact-v2 matrix is **9/9 operators and
180/180 dispatch/lowering routes**. This result is labelled
`verified-local-compile`; it is not a numerical or performance result.

A second, narrower gate built the exact v2 source natively for AArch64. On the
local `Ascend950PR_9599` camodel, all **9/9 basic float32 smokes** and all
**11/11 additional dtype/control-flow smokes** passed against Torch references.
The critical routes include int8/uint8 MaskedScale, narrow BF16 SwiGLU,
ForeachNorm `p=inf`/`p=1.5`, and BF16 RMSNorm with odd `D=17`. This is labelled
`verified-camodel-smoke`; it is not equivalent to the official 180-case matrix.
The local native wheel is kept under
`integrations/cannbench/.camodel-runtime-dist/` (gitignored).

The generated bundle is under
`integrations/cannbench/workers/runs/20260902_full_skill_generated/`. The
canonical `integrations/cannbench/submission/` package was not overwritten and
no submission credit was consumed.

| Operator | Official cases | Unique compiled specializations | Local result |
|---|---:|---:|---|
| Sigmoid | 20 | 3 | 20/20 |
| Exp | 20 | 3 | 20/20 |
| Mish | 20 | 3 | 20/20 |
| Gelu | 20 | 6 | 20/20 |
| MaskedScale | 20 | 12 | 20/20 |
| SwiGLU | 20 | 4 | 20/20 |
| ForeachAddcdivScalar | 20 | 3 | 20/20 |
| ForeachNorm | 20 | 19 | 20/20 |
| RMSNorm | 20 | 3 | 20/20 |

Machine-readable evidence is in
`evidence/cannbench/generated_full_coverage_20260902/summary.json` (SHA-256
`42ed5c39f8ef51375b14e611e5d2ff907571c50de5eb752fdc2aff7c8056cacb`).
Camodel results and raw per-run records are in
`evidence/cannbench/generated_camodel_smoke_20260902/`.

## How the kernels were generated

The main campaign used `dashscope/qwen3.7-max` and
`dashscope/glm-5.2`. Each accepted phase had to invoke OpenCode's native
`skill` tool with a directory resolving to this repository's `skills/`; text
claims were not accepted as provenance. The driver also required a phase
completion marker, source artifact, static contract check, source hashes, and
all 20 exact-v2 routes.

The first matrix qualified six operators directly. SwiGLU produced a complete
candidate but missed its marker after a worker timeout. ForeachNorm produced a
candidate with an invalid host/JIT boundary, and RMSNorm produced first a UB
overflow and then a PlainValue/LocalTensor type error. Those three were passed
through the staged measured-repair workflow. Qwen repaired ForeachNorm and
RMSNorm; GLM independently reviewed the final ForeachNorm, RMSNorm and SwiGLU
artifacts. Session IDs and observed skill directories are stored in each
operator's `provenance.json`.

The full chain is preserved rather than flattened:

- main generation: `workers/runs/20260902_140812/`;
- measured repair attempts: `workers/runs/20260902_staged_completion/` and
  `workers/runs/20260902_staged_completion2/`;
- compact successful SwiGLU review:
  `workers/runs/20260902_staged_completion3/`;
- final immutable collection: `workers/runs/20260902_full_skill_generated/`.

## What the local gate proves

The QEMU image installs the self-contained CPython 3.12/x86_64 wheel built from
`compiler-team/pyasc`, branch `v2`, commit
`ac1222a48c8914d3f81297c7570d1a84f0f26778`. For every official YAML case it
imports the module, executes host dispatch with shape/dtype tensors, captures
the selected JIT launches, specializes the pointers and scalars, performs pyasc
codegen/passes/AscendC translation, and checks the 950PR UB budget.

The QEMU gate does **not** run the generated binary. Consequently, 180/180 does not prove
the formulas, output values, NaN positions, real-NPU launch behavior, or speed.
The representative camodel gate now checks basic formulas for all nine plus
selected high-risk dtype/control branches. Official large layouts, complete
special-value coverage, and every case still require broader camodel and
CANNBench evidence with separate labels.

The critical run also exposed avoidable vector diagnostics in padded lanes.
`real_shape` controls the loaded/stored extent, but later arithmetic still
executes over the full local tile. The generation skill and worker constraints
now require an explicit neutral `pad_value` (notably one for divisors). Some
diagnostics remain legitimate when the operator itself permits zero divisors
or computes a general norm through `log(0)`; output comparison remains the
numerical pass criterion.

## Current comparison point and next gate

The most recent official canonical run remains 178/180: eight operators passed
20/20, while SwiGLU failed case 9 at NPU runtime and case 12 on bf16 NaN
positions. The generated bundle's 180/180 compile result does not supersede
that hardware result.

Before a future submission:

1. Expand the current 20-route camodel smoke to every distinct
   dtype/control-flow specialization and then to feasible official shapes.
2. Add matched ACLNN/reference camodel runs before interpreting tick deltas as
   speedups; the current raw ticks are not comparable across operators.
3. Prioritise SwiGLU cases 9 and 12 on real NPU because QEMU cannot reproduce
   either failure class.
4. Run all nine operators privately; only a 180/180 hardware result with zero
   anti-cheat failures may replace the canonical package.
5. Use CANNBench profiler results as measured feedback for the next perf-tuning
   cycle; do not tune from compile success or model intuition alone.
