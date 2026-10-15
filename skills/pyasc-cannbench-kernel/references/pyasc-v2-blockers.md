# Pinned pyasc v2 limitations relevant to CANNBench

This is a routing summary. Detailed evidence and confidence labels live in
`docs/pyasc-v2-cannbench-blockers.md`.

- Generated AscendC can declare the same scalar identifier for f16 and bf16,
  producing a `c0_f16` redefinition. Isolate the half types or bridge through a
  separate f32 buffer.
- Direct arithmetic/cast paths from int8 tiles are incomplete. The observed
  conversion route is int8 -> f16 -> f32.
- Tile DMA requires a 32-byte-aligned final dimension even when `real_shape`
  is smaller. Choose an aligned physical tile and a valid layout/fallback.
  The verbatim compiler diagnostic is `error: Last dimension of a tensor must
  be aligned by 32 bytes, got 4 x 4 bytes` (measured on pinned-runtime nms
  with a `[256, 4]` f32 tile built from 4-component boxes). Narrow feature
  dims (4, 2, 1 elements) must be padded to an aligned physical extent or
  handled through a flat 1-D layout.
- `real_shape` tail padding is still evaluated by later vector operations.
  Use an explicit neutral `pad_value`; zero-padding a divisor produces CAModel
  divide-by-zero/Inf diagnostics even when the stored logical output is right.
- Static UB usage is sensitive to live SSA values and loop unrolling. Run the
  exact compile gate for every specialization; keep a smaller-tile fallback.
- Python-float loop accumulators cannot be reassigned to pyasc PlainValue/Tile
  results. Seed with a reduction result of an aligned zero tile.
- In the historical GeLU snapshot `adadd7d6`, `asctile.where` supports numeric
  scalar branches; the older tensor-only branch restriction must not be carried
  forward as a blanket rule. Confirm supported operand kinds against the
  CANNBench runtime pin `0a631f70` before relying on scalar branches.
  Tensor-only unary ops such as `asctile.rsqrt` still reject PlainValue
  reductions; use supported scalar algebra or materialize an aligned tensor.
- Binary tensor APIs have the same distinction: `asctile.maximum` rejects two
  PlainValue reduction results. This surfaced in the ForeachNorm infinity
  route; preserve a LocalTensor accumulator or use supported scalar algebra.
- Passing `Tensor.data_ptr()` is an integration error, not a kernel workaround:
  pass the tensor to the JIT so pointer dtype specialization remains available.
- Local QEMU compilation cannot establish NPU runtime behavior, numerical
  correctness, or performance.
- The base low-level `asc.Compiler` unconditionally legalizes an FFTS argument,
  including for C310. On CANNBench 950PR this fails in `c2c_ctrl_addr()` before
  the kernel launches. This is historical low-level diagnostic evidence, not
  permission to introduce a custom compiler adapter into new kernels. Use the
  supported public AscTile path for the high-level campaign.
- In the historical GeLU snapshot `adadd7d6`, public decorators and invocation
  overrides work in a cold AscTile-first process. If base `asc.runtime.jit.JITFunction` discovers its
  options first, inherited `_config_keywords` hides AscTile-only options and
  `@asctile.jit(reuse_alloc=1)` raises unknown option. Extraction itself uses
  `self.compiler.options_cls`. Record this import-order defect; do not load a
  concrete-options adapter or patch the compiler in the high-level campaign.
  [Reproducer](../../../../integrations/cannbench/comparisons/gelu-asctile-jit-20260907/jit_probe.py).
- Public-erf FP32 exact GeLU fails dense negative-tail qualification locally
  in the historical GeLU snapshot at `adadd7d6` / CANN 9.0 for all 18 reuse/static/fusion combinations. A
  64-element constant-input reproducer isolates backend Erf error amplified
  by cancellation. This is not a measured NPU result, and compiling all 20
  shapes does not override it. Keep submission blocked until repaired through
  a permitted public API/backend change.
  [Evidence](../../../../integrations/cannbench/comparisons/gelu-asctile-jit-20260907/evidence/numeric-reproducer.json).
- Multi-tile/repeated FP16 Model execution at this pin requires an explicit
  runtime gate even after `verify_sync=True` passes. Fresh-cache exact GeLU
  with reuse1/VF=true exceeds a 120-second whole-child budget with repeat1
  incomplete, also with aligned input and `always_compile=True`. Longer
  traces do show progress through repeats and eight-core launches. Controls
  disabling either reuse or VF, and FP32 controls, complete. Do not infer
  repeat-specific/NPU deadlock or a cache-only cause from deadline exhaustion.
  [Bounded controls](../../../../integrations/cannbench/comparisons/gelu-asctile-jit-20260907/evidence/tail-controls/summary.json).
- Static UB success does not prove a large VF-fused loop is runnable. A
  72-core, tile-13,824 GeLU tanh route with VF+reuse fit 221,184 bytes but timed
  out on the vector core in official case 4. Treat the combination as unsafe
  until separately reproduced and bounded.
- Native low-level GeLU arithmetic supports FP16/FP32 in the current compile
  probe, but basic `asc.mul` rejects BF16. Keep BF16 computation promoted to
  FP32 and cast once on output.
- CANN exposes a fused `AscendC::Gelu` helper, but pyasc v2 at `0a631f70` has no
  dialect/Python binding for it. Composed `erfc`/`tanh`/`exp` routes cannot be
  assumed to match the fused baseline; record this as an upstream API and
  performance blocker.
- Target-derived kernels may rely on launch-time `asctile.ConstExpr` wrappers
  even when their JIT signatures are unannotated. Preserve the upstream test's
  dispatch contract; this is required by current-v2 RMSNorm.
- Current-v2 `asctile.softmax` accepts f16/f32 local tensors but rejects BF16.
  A complete CANNBench Softmax implementation needs an explicit supported-dtype
  compute path rather than copying the target kernel unchanged.
- Current-v2 `asctile.rms_norm` likewise accepts f16/f32, not BF16. Converting
  BF16 through f16 changes the reference accumulation semantics and may expose
  mixed-half generated declarations; route BF16 through an explicit f32
  reduction/normalization path.
- Current-v2 local transpose rejects int64 and the store lowering does not
  support tensors with rank greater than four. Use a word-preserving supported
  dtype view for int64 movement and collapse/tile high-rank layouts before the
  JIT store when implementing the complete benchmark contract.
- Current-v2 transpose axes are variadic. Use `tile.transpose(*permute)` (or
  explicit integer axes), never `asctile.transpose(tile, [1, 0])` with a list
  supplied as one positional axis.
