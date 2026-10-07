Rechecked on compiler-team/pyasc v2 **9069108e323746187c48d78fec4929c4afa2efc4** (2026-10-07), built with `PYASC_SETUP_EXPERIMENTAL=1`. The API is now `from asc.experimental import asctile`.

The limitation remains: [unary_ops.py](https://gitcode.com/compiler-team/pyasc/blob/9069108e323746187c48d78fec4929c4afa2efc4/experimental/asctile/python/asctile/language/unary_ops.py#L208) defines `erf(input)` only and emits `math.erf`; it does not expose `ErfConfig`.

The precise reference dependency is now pinned publicly: [opbase Vec::Erf](https://gitcode.com/cann/opbase/blob/f59a9cb9f414c4b80cf4eccc58006798c1af15de/pkg_inc/op_common/atvoss/util/vec.h#L278) uses `SUBSECTION_POLYNOMIAL_APPROXIMATION`; [ops-nn GeLU DAG](https://gitcode.com/cann/ops-nn/blob/f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307/activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h#L134) selects that wrapper for FP32 exact. FP16/BF16 use FP32 intermediates but default Erf.

I compiled and executed the following escape hatch on CANN 9.0 / CaModel Ascend950PR_9599, using the **unmodified** current compiler:

```python
erf_value = asctile.inline_vf("""
    {
        static constexpr AscendC::ErfConfig config = {
            AscendC::ErfAlgo::SUBSECTION_POLYNOMIAL_APPROXIMATION};
        AscendC::Erf<float, false, config>($0, $1, $1.GetSize());
    }
""", [tile], asctile.float32, [x * 0.7071067690849304])
result = (erf_value + 1.0) * (x * 0.5)
```

With tile 15872, unroll 1, `reuse_alloc=2`, `vf_fusion=True`, synchronization enabled, compiler-reported UB is 190464 bytes. This is not an NPU performance result.

Important qualification: exposing the selector is useful for reproducing the reference choice, but **is not by itself a guarantee of CANNBench accuracy**. The compiled FP32 route passed sampled official input ranges, yet a separate 2049-point `linspace(-8, -2)` stress segment failed the official checker: maximum absolute error about 1.204e-7; cancellation error count 6 versus CPU count 2. Output guards remained intact. This is a sampled composition/SDK/checker result, not proof of an AscTile pass bug or a claim that the actual remote reference uses this precise public SDK revision. Local CANN is 9.0; historical runners reported 9.1.

A supported selector (or a numerically stable high-level GeLU/Erfc API) would let users investigate this without raw AscendC. No compiler changes were made.
