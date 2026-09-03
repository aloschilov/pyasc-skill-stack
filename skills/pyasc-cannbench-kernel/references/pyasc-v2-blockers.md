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
- `real_shape` tail padding is still evaluated by later vector operations.
  Use an explicit neutral `pad_value`; zero-padding a divisor produces CAModel
  divide-by-zero/Inf diagnostics even when the stored logical output is right.
- Static UB usage is sensitive to live SSA values and loop unrolling. Run the
  exact compile gate for every specialization; keep a smaller-tile fallback.
- Python-float loop accumulators cannot be reassigned to pyasc PlainValue/Tile
  results. Seed with a reduction result of an aligned zero tile.
- `asctile.where` requires tensor-valued branches, and tensor-only unary ops such
  as `asctile.rsqrt` reject PlainValue reductions. Select scalar-compatible
  algebra or explicitly materialize one aligned local tensor.
- Binary tensor APIs have the same distinction: `asctile.maximum` rejects two
  PlainValue reduction results. This surfaced in the ForeachNorm infinity
  route; preserve a LocalTensor accumulator or use supported scalar algebra.
- Passing `Tensor.data_ptr()` is an integration error, not a kernel workaround:
  pass the tensor to the JIT so pointer dtype specialization remains available.
- Local QEMU compilation cannot establish NPU runtime behavior, numerical
  correctness, or performance.
- The base low-level `asc.Compiler` unconditionally legalizes an FFTS argument,
  including for C310. On CANNBench 950PR this fails in `c2c_ctrl_addr()` before
  the kernel launches. Use a narrowly evidenced C310 compiler repair and verify
  `has_ffts_arg=false`, or stay on AscTile.
- AscTile defines `reuse_alloc`, `static_alloc`, and `vf_fusion` in its compiler
  options, but inherited JIT option discovery/extraction uses the base option
  type at commit `0a631f70`. Direct decorators with those options need the
  integration's concrete-options adapter.
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
