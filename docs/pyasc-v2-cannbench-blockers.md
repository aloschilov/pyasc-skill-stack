# pyasc v2 blockers exposed by CANNBench generation

Scope: `compiler-team/pyasc` branch `v2`. The original nine-operator campaign
used `ac1222a48c8914d3f81297c7570d1a84f0f26778`; the four-operator comparison
uses the then-current `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`, target
`Ascend950PR_9599`. “Confirmed” means reproduced by pinned-v2 lowering or an
official NPU report. “Suspected” means the observed symptom is real but its
root cause has not been isolated.

## Confirmed pyasc/compiler constraints

- **Target Softmax does not lower for BF16 on current v2.** The unchanged
  `python/test/asctile/target/test_softmax.py::softmax_fused` calls
  `asctile.softmax` directly on BF16 input; pinned `030e9b2c` rejects it with
  `input dtype must be one of float16, float32`. Six of the 20 official
  CANNBench Softmax cases exercise BF16. Evidence:
  `integrations/cannbench/comparisons/four-operator-20260903/handwritten/local_validation/softmax.json`.
- **Target RMSNorm requires explicit launch-time ConstExpr wrapping.** Its JIT
  signature leaves shape/tiling arguments unannotated, while the upstream test
  wraps `num_col_align`, `block_factor`, and `ub_factor` with
  `asctile.ConstExpr` at dispatch. A generic host adapter that passes plain
  integers fails every specialization. Matching the upstream launch contract
  restores 20/20 local compilation without changing the JIT body. Evidence:
  `integrations/cannbench/comparisons/four-operator-20260903/handwritten/local_validation/rms_norm.json`.
- **Target Transpose does not cover the full advertised dtype/rank contract.**
  On pinned `030e9b2c`, unchanged target JIT bodies lower for 17/20 official
  cases. `LocalTensor.transpose` rejects int64, and the lowering reports
  `Store tensors with dim > 4 not implemented` for a rank-5 route. Evidence:
  `integrations/cannbench/comparisons/four-operator-20260903/handwritten/local_validation/transpose.json`.

- **PlainValue versus LocalTensor unary API.** `asc2.rsqrt(acc * inv_D +
  epsilon)` fails because a reduction accumulator is PlainValue and `rsqrt`
  requires LocalTensor. RMSNorm failed 0/20 until the scalar was materialised
  with an aligned `asc2.full` tile. Evidence:
  `workers/runs/20260902_140812/rms_norm-generate/iter2/local_compile.json`.
- **PlainValue versus LocalTensor binary API.** ForeachNorm's infinity route
  failed on `asc2.maximum(acc, m)` because both reductions were PlainValue.
  The diagnostic is `BinaryOperandTypeError: At least one operand must be
  tensor`. Evidence:
  `workers/runs/20260902_staged_completion2/foreach_norm/preflight/local_compile.json`.
- **Scalar branches in `where`.** A generated sigmoid review changed one branch
  to Python `1.0`; pinned v2 attempted to read its `.dtype`. Both data branches
  must be LocalTensor. The exact gate rejected the candidate before packaging.
- **Static UB sensitivity.** A generated RMSNorm with `tile_d=2048` consumed
  roughly 279–329 KB depending on dtype, above the 253,952-byte 950PR budget.
  `tile_d=1024` with a single aligned scalar tile lowers successfully. Evidence:
  `workers/runs/20260902_140812/rms_norm-generate/iter1/local_compile.json`.
- **Mixed f16/bf16 generated identifiers.** A specialization containing both
  half types can emit conflicting `c0_f16` declarations. Current workaround is
  to isolate types or cast each input directly into an f32-only arithmetic
  path. The failure and workaround are recorded in
  `evidence/cannbench/comparison.md`.
- **int8 vector conversion path is incomplete.** Direct arithmetic and direct
  `.to(float32)` on int8 tiles fail; the measured route is int8 → f16 → f32.
  uint8 requires host metadata reinterpretation plus an f32 correction. The
  MaskedScale official 20/20 result confirms the workaround, not that the API
  limitation is resolved.
- **2-D copy last-dimension alignment.** Physical copy width must satisfy a
  32-byte final-dimension constraint even with `real_shape` tails. SwiGLU's
  `[1000003,2]` bf16 route therefore needs a separately valid narrow layout.
- **Tail padding is computationally active.** `copy_in(...,
  real_shape=[n])` fills the rest of the physical tile, and subsequent vector
  instructions evaluate those lanes. A zero-padded denominator generated
  divide-by-zero and Inf/NaN CAModel diagnostics while the logical
  ForeachAddcdivScalar output still matched Torch. The skill now requires an
  operation-neutral explicit `pad_value`; this is an API semantic constraint,
  not a failed logical-output case.

## Confirmed integration and skill-stack defects

- **Raw pointers at launch.** A worker used `Tensor.data_ptr()` and host-side
  slice assignment. The pinned JIT contract needs the tensor itself to retain
  pointer dtype specialization; final dtype conversion must occur in a JIT
  kernel. Static checks now reject `data_ptr()`.
- **Failure compaction used the wrong case field.** The local report records
  `dispatch` and `compile`, not `status`. The driver previously fed all 20
  stack traces back for one failing specialization. It now selects only failed
  routes and deduplicates first-line signatures.
- **Review context was oversized and contradictory.** The generic review skill
  still contains standalone `tensor/load/store`, numpy and `always_compile`
  guidance. CANNBench routing now stops at the compact benchmark skill, and
  repair/review loads only that profile. This changed repeated 360-second
  timeouts into completed cross-model reviews.
- **Review independence after failover.** The original campaign could select
  the same model for implementation and review after a phase retry. The driver
  now calculates review selection from the model that actually completed
  implementation.

## Official NPU symptoms with suspected roots

- **SwiGLU case 9:** confirmed runtime failure around
  `narrow(...).contiguous()` with `aclnnInplaceCopy`; the asynchronous trace is
  not sufficient to assign the root to torch guard, pyasc, layout, or runtime.
- **SwiGLU case 12:** confirmed bf16 NaN-position mismatch with zero reported
  finite-value error; the exact operation causing the NaN difference is not
  isolated.
- A later replacement SwiGLU package failed 0/20 with generic abnormal exits.
  That establishes a regression only; it does not establish platform
  auto-detection or any other proposed cause.

Evidence for these symptoms is in
`evidence/cannbench/full_official_9op_summary.json` and
`evidence/cannbench/swi_glu_fix_failed_summary.json`.

## Native build and camodel status

The machine has AArch64 CANN/camodel and the local LLVM/MLIR reference at
`/home/aloschilov/workspace/llvm`; the submitted self-contained exact-v2 wheel
is CPython 3.12/x86_64. A native CPython 3.10/AArch64 exact-v2 wheel was built
successfully (SHA-256
`f40f54fc7edd443a044c5db0c685b82e6feec0f2e5bc6b4dc0c84d819b314f57`).
It is retained locally at
`integrations/cannbench/.camodel-runtime-dist/pyasc-1.1.1-cp310-cp310-linux_aarch64.whl`
and ignored by git as a reproducible binary artifact.
The first parallel build exposed a missing generated-header dependency
(`AscendCDialect.h.inc` was consumed before generation); a serial pass generated
the prerequisites, after which parallel continuation succeeded. GCC `-O3`
compilation of `Translation.cpp` took roughly 15 minutes and peaked around
12 GB RSS, so the native build is viable but operationally expensive.

The exact native wheel then ran one basic float32 route for every generated
operator plus 11 selected dtype/control-flow routes; all 20/20 matched Torch
references. The remaining local gap is the full 180-case
shape/special-value matrix and matched reference tick normalisation, not basic
Model-runtime availability. Evidence:
`evidence/cannbench/generated_camodel_smoke_20260902/summary.json`.
