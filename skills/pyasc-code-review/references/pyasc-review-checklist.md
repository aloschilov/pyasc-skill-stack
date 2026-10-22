# Current-v2 AscTile review checklist

## Contract

- [ ] Exact callable name, parameters, defaults, output shape/dtype/list arity
- [ ] Every task case, dtype, attribute, boundary, and special value covered
- [ ] No case removal, narrowing, or host numerical fallback

## API and syntax

- [ ] `import asctile` and at least one real `@asctile.jit` numerical kernel
- [ ] `global_tensor` + `copy_in/copy_out`; no legacy tile API
- [ ] Supported JIT options, range kwargs, value kinds, and transpose axes
- [ ] No unsupported Python constructs inside JIT
- [ ] `asctile.concat(*tiles)` variadic, never a single list argument
- [ ] String-typed numeric attrs (`epsilon`/`norm_eps`/`eps` values like
  `'1e-05'`) coerced with `float()`/`int()` before reaching JIT arguments
- [ ] List-valued attrs (`group_list`, etc.) stay host metadata: no tensor
  methods on the list; materialize with whitelisted `torch.tensor` (never
  `torch.as_tensor`) when the kernel needs them on-device
- [ ] Host dispatch survives the shape-only gate replay: no unguarded
  `.item()`/tensor-value reads; genuinely data-dependent reads use a
  fallback that catches only the replay signature, is unreachable on the
  real evaluator, and supplies shape metadata only — never numerical values

## Memory and numerics

- [ ] Ranks/offsets/real shapes match and final physical dimensions align
- [ ] Tail pad values are operation-neutral
- [ ] Live local buffers fit memory after unrolling
- [ ] Dtype promotion/casts and NaN/Inf/signed-zero behavior match the golden
- [ ] Reductions use correct identities and cross-core combination

## Performance and evidence

- [ ] Shape-aware core count and tile size; contiguous DMA where possible
- [ ] No unnecessary materialization, launch explosion, or repeated cast path
- [ ] Exact-v2 compile matrix recorded
- [ ] Camodel and CANNBench claims are scoped and linked independently
