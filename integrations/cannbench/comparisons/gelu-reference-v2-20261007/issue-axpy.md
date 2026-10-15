# [Feature/API] Explicit fused multiply-add/Axpy semantics for AscTile reference-DAG reproduction

## Scope

Current compiler-team/pyasc v2: `9069108e323746187c48d78fec4929c4afa2efc4`, experimental AscTile enabled. This is an API/lowering question, not a claim that separate Mul+Add violates Python expression semantics or that a measured hardware regression is caused by it.

## Reference

[Public ops-nn GeLU tanh implementation](https://gitcode.com/cann/ops-nn/blob/f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307/activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h#L103) computes `square=x*x`, `cube=square*x`, then `MicroAPI::Axpy(cube, x, TANH_APPROX_FACTOR, mask)`, followed by multiplication, Exp, Adds, Div. Its FP32 constants are `22.363859176635742` and `-0.07135481387376785`.

## Observed AscTile lowering

In a compiled GeLU kernel, the authored high-level expression is:

```python
x = raw.to(asctile.float32)
square = x * x
cube = square * x
argument = (cube + x * 22.363859176635742) * -0.07135481387376785
result = x / (asctile.exp(argument) + 1.0)
```

With `reuse_alloc=2`, `vf_fusion=True`, `insert_sync=True`, `opt_level=3`, C310, the emitted AscendC is (identifiers preserved from the actual dump):

```cpp
AscendC::Reg::Mul(v51, v50, v50, v62);
AscendC::Reg::Mul(v52, v51, v50, v62);
// An intermediate UB store is also emitted here; tracked separately in #5.
AscendC::Reg::Duplicate(v53, c22_3638592_f32, v62);
AscendC::Reg::Mul(v54, v50, v53, v62);
AscendC::Reg::Add(v55, v52, v54, v62);
```

This is an observation of **emitted AscendC**, not a final ISA assertion. The installed objdump reports instructions as `<not available>`, so backend contraction has not been established either way. The sampled compiled FP32 tanh route passes the official numerical checker; numerical failure is not the motivation for this request.

## Requested guidance

The public AscTile exports currently expose `mul`/`add`, but no explicit elementwise `fma`/`axpy`. Is there a supported high-level way to require the same fused operation/rounding semantics as the reference? If not, please consider a documented explicit fused operation or opt-in contraction policy, with dtype/cast behavior and platform restrictions.

For a high-level reference reproduction, silently rewriting `a*b+c` is not necessarily correct because contraction changes rounding. Handwritten register code through `inline_vf` would express the reference instruction, but would abandon the high-level implementation style (our only allowed inline exception is Erf/Erfc).

Please distinguish mathematical equivalence, rounding equivalence, emitted API calls and final backend instructions. A fused API could reduce instruction/dependency count, but no NPU speedup is asserted here. No compiler patches or custom pass pipeline were used.
