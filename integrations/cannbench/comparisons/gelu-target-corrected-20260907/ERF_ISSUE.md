# [Feature/API] Expose Erf algorithm selection in AscTile (ops-nn FP32 GeLU uses a different accuracy mode)

Related to #6, but this request is specifically about exposing an existing AscendC algorithm choice, not asserting a compiler-pass bug or a violation of the default Erf accuracy contract.

## Source evidence

Tested pyasc v2: `adadd7d66ed0ee16d33d79487bf584899a26ef1e`, local CANN 9.0, CaModel Ascend950PR_9599.

Public ops-nn revision `f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307`:

- [gelu_v2_dag.h, FP32 route (lines 134–148)](https://gitcode.com/cann/ops-nn/blob/f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307/activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h#L134): `GeluV2Erf32BDag` uses `Vec::Erf<T>`.
- [Same file, low-precision route and ErfFast (lines 30–37, 115–132)](https://gitcode.com/cann/ops-nn/blob/f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307/activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h#L30): FP16/BF16 promote to float, but use `ErfFast<T>`, which calls default `AscendC::Erf`.

The matching installed CANN 9.0 dependency `pkg_inc/op_common/atvoss/util/vec.h` defines `Vec::Erf` with `ErfConfig{ErfAlgo::SUBSECTION_POLYNOMIAL_APPROXIMATION}` and invokes `AscendC::Erf<T, false, config>`. This configuration is in the dependency, not directly in the linked ops-nn DAG.

[Official AscendC Erf documentation](https://asc.gitcode.com/api/SIMD-API/adv_api/math_compute/Erf_interface/Erf.html) documents default `PADE_APPROXIMATION` as the performance-oriented algorithm and `SUBSECTION_POLYNOMIAL_APPROXIMATION` as the high-accuracy algorithm. The config parameter is supported on 950PR/950DT.

At the tested pyasc pin, public `asctile.erf(input)` has no algorithm/config argument. Its `math.erf` lowering emits the default AscendC Erf call, without an explicit config. Reuse/static allocation/VF flags do not expose this choice.

## Why this matters

For exact GeLU, `y = (x * 0.5) * (1 + erf(x / sqrt(2)))`, a small absolute Erf error is amplified in relative terms when Erf is close to -1. In our existing #6 reproduction, the default high-level route fails FP32 negative-tail checks; the full CANNBench diagnostic passed 18/20, with exact FP32 cases 11 and 20 failing. This is evidence about this composition and checker, not proof of an incorrect Erf primitive or compiler pass.

Without a public selector, users cannot reproduce the algorithm choice already made by ops-nn while keeping kernels in target-style AscTile. They must either accept insufficient composition accuracy or add a more expensive high-level tail treatment. Our sampled passing workaround adds a six-level normal-tail continued fraction; on a matched local FP32 probe it costs 4966 Model ticks versus 3521 for the failing direct-Erf route. These are local Model measurements, not NPU latency, and NOT a measurement of the subsection algorithm.

## Requested API and validation

Please expose a documented, compile-time algorithm selection through public AscTile Erf, preserving the current default. Reject unsupported platform choices clearly, include the choice in specialization/cache identity, and account for its temporary UB requirements. A dedicated accurate GeLU/erfc API could be an alternative if Erf output rounding still prevents the required composition accuracy.

Validation should compare both algorithms on identical inputs, dtype, tile/core geometry and JIT settings; record primitive absolute/ULP error separately from GeLU error, NaN/Inf/subnormal behavior, peak UB, synchronization and warmed execution time.

We have NOT executed or benchmarked ops-nn's subsection-configured Erf in this experiment. Exposing it would enable a meaningful accuracy/performance comparison; it is not yet proven to fix all GeLU cases or to be faster than the workaround. A more accurate FP32 Erf can still lose very small tails when rounded near -1.

## Russian comment

Это отдельный запрос на публичный выбор алгоритма Erf, связанный с #6, а не утверждение об ошибке pass'ов. В ops-nn FP32 exact GeLU использует Vec::Erf; соответствующий заголовок установленного CANN 9.0 выбирает SUBSECTION_POLYNOMIAL_APPROXIMATION. FP16/BF16 идут через FP32-промежуточные, но используют ErfFast с алгоритмом по умолчанию. Ссылки на публичный DAG и официальную документацию приведены выше.

Настройка влияет на точность Erf, стоимость вычисления и требования к временной UB. Для GeLU особенно важен отрицательный хвост: в 1 + erf(...) ошибка усиливается из-за вычитания близких величин. Сейчас публичный asctile.erf не позволяет выбрать этот режим; reuse_alloc/static_alloc/vf_fusion не заменяют выбор математического алгоритма.

Высокоточный режим ops-nn пока не измерялся нами: нельзя обещать, что он исправит все случаи или ускорит GeLU. Нужен контролируемый эксперимент с обоими алгоритмами. Уже проверенный высокоуровневый обход через устойчивый хвост проходит локальные выборочные проверки, но добавляет вычисления; это практическая мотивация предоставить пользователю выбор без inline AscendC и ручных регистров.
