---
name: pyasc-code-review
description: Review current-v2 asctile kernels for contract coverage, API validity, correctness, memory safety, numerical behavior, performance, and evidence quality.
---

# Current-v2 pyasc kernel review

Review against the exact selected source and task contract. For the current
CANNBench campaign use `pyasc v2@0a631f70968c3cb7c33ce45330a85768dd5a6f06`, `import asctile`,
`@asctile.jit`, `global_tensor`, `copy_in`, and `copy_out`.

## Review order

1. **Public contract:** exact callable name/signature/defaults; every shape,
   dtype, attribute, list arity, tail, and special-value route is covered.
2. **Implementation boundary:** torch is limited to allocation, metadata, and
   launches. No torch numerical fallback, native operator, CPU round-trip, or
   `Tensor.data_ptr()`.
3. **Current API:** no `asc2`, `asctile.tensor/load/store`, removed compile
   options, invalid range kwargs, or wrong transpose argument form.
4. **Memory:** descriptor/local/offset/real-shape ranks match; physical final
   dimensions are transfer-aligned; tail padding is operation-neutral; all
   live buffers fit UB/L1/L0 after unrolling.
5. **Numerics:** promotion and final cast match the golden; NaN/Inf/signed-zero
   behavior is deliberate; reduction identities and cross-core combines are
   correct.
6. **Performance:** useful core occupancy, contiguous DMA, shape-aware tiles,
   no unnecessary materialization or launch explosion, and no tiny fixed tile
   across all shapes.
7. **Evidence:** static, exact-v2 compile, camodel, and CANNBench claims are
   clearly separated.

## Critical rejection conditions

- a benchmark case is omitted, narrowed, or silently routed to torch math
- candidate source does not contain a real `@asctile.jit` numerical path
- host output aliases an input contrary to the contract
- runtime source/wheel commit differs from the declared commit
- compile-only evidence is represented as numerical or performance evidence
- CANNBench score is quoted without case count and immutable run/job identity

## Known current-v2 checks

- int8 arithmetic may need int8 -> f16 -> f32
- f16/bf16 temporaries in one specialization can cause declaration conflicts
- compare/select destination sizing has a measured 256-byte repeat hazard
- scalar reductions may be `PlainValue`, not `LocalTensor`
- BF16 builtin softmax, int64 local transpose, and rank>4 stores require
  explicit supported routes rather than hopeful dispatch
- target-derived RMSNorm launches may require explicit `ConstExpr` wrappers
- bare `asctile.block_idx`/`asctile.block_num` (missing `()`) fail lowering
  with `TypeError: Unsupported value type for materialization: function`;
  they are zero-arg calls
- `asctile.global_tensor`/`copy_in`/`copy_out` at host level or passed across
  the host/JIT boundary fail with `RuntimeError: '<name>' cannot be called
  without initialization of global builder`; construction belongs inside
  `@asctile.jit` code with `GlobalAddress` parameters, and the launch
  receives raw host tensors
- `asc.GlobalAddress`/`asc.ConstExpr` and the `asctile.`-prefixed spellings
  both resolve on the pinned runtime (`asctile` re-exports them); do not
  reject a candidate on spelling alone — judge by compile evidence

## Output format

Report findings first, ordered by severity. For each finding include the exact
file/line or route, affected cases, expected behavior, and a concrete repair.
Then state which gates were actually run and list residual risks. “No findings”
must still state evidence scope.

## References

- [Review checklist](references/pyasc-review-checklist.md)
- [Secure host-code notes](references/PythonSecureCoding.md)
- `pyasc-cannbench-kernel/references/pyasc-v2-blockers.md`
