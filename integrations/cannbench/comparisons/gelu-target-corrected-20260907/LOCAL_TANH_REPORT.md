# Corrected target GeLU: local accuracy and performance

Date: 2026-09-07. **Local CaModel experiment, not a CANNBench submission.**

Latest follow-up: the [high-level exact FP32 remedy](EXACT_REMEDY.md) now
passes local dense/random/boundary and repeated-tail accuracy checks. Its
selected six-level variant takes 4966 Model ticks in the matched small probe.
The [combined run plan](RUN_PLAN.md) can resume final-package qualification;
no new credit has been spent. The [earlier direct-Erf failure](EXACT_CHECK.md)
and original tanh results below remain historical evidence, not overwritten.

## Outcome

`verified-camodel-smoke`: correcting the two scalar coefficients makes the
unmodified FP32 target kernel pass the representative tanh-GeLU checks. The
original coefficients fail the same independent oracle. The correction has no
observed median timing penalty in the matched local probe: both take 9384 ticks.

Coefficient correction alone is **not** a general FP16/BF16 solution:

- FP16 fails the dense negative-tail checks. For example, at `x=-4.1015625`
  it returns `-0.0`, versus double-reference `-4.287338957873659e-5`.
- BF16 is rejected before execution: `asctile.exp` accepts only FP16/FP32
  in the tested runtime.
- A separate [FP32-intermediate extension](promoted_kernel.py), with public
  casts before arithmetic and on output, passes the same sampled checks for
  FP16 and BF16 at tile 8192, unroll 2, using 131072 B of UB.
  This is **not** the coefficient-only variant; its performance is unmeasured.

`suspected`: FP16 exponent/denominator overflow explains the negative-tail zero;
the FP32-cast control supports this explanation, but individual intermediate
values were not instrumented. For the example above, real-valued arithmetic
gives an exponent argument about 11.46862 and `exp` about 95665.86, exceeding
FP16's largest finite value 65504. This calculation is supporting CPU evidence,
not a captured compiled intermediate. These findings do not establish a
compiler-pass bug.

## Source, mathematics and runtime

[target_kernel.py](target_kernel.py) contains the upstream function unchanged;
the harness asserts AST equality against the pinned target. Only host-supplied
constexpr coefficients differ:

```python
TANH_APPROX_FACTOR = 0.044715
NEG_SQRT_EIGHT_OVER_PI = -math.sqrt(8 / math.pi)
# x / (1 + exp(-sqrt(8/pi) * (x + 0.044715*x**3)))
```

The original pair was `1/0.044715`, `-1.595769121*0.044715`, which produces
`x/(1+exp(-1.595769121*(0.044715*x+x**3)))` instead. The tiny extra precision
in the corrected scale is recorded separately from this coefficient-placement error.

Source/runtime pin: `adadd7d66ed0ee16d33d79487bf584899a26ef1e`.
Native AArch64/CPython 3.11, CANN 9.0, CaModel `Ascend950PR_9599` (C310).
The upstream target file has identical Git blob
`840ce5a6e97b7caebfeb3921e88dc1faee030e49` at the local checkout
`0a631f70968c3cb7c33ce45330a85768dd5a6f06`, runtime pin, and fetched v2
`62ebad490d3c88d49e5e1eee49581dabd0bafde8`. The runtime was **not** relabeled
as that newer revision. Neither compiler nor original target checkout was edited.

## Accuracy scope

[check.py](check.py) executes compiled kernels, rather than just CPU formulas.
Independent oracle: `torch.nn.functional.gelu(..., approximate="tanh")` in
double precision, with same-dtype CPU output for the official CANNBench
precision checker. Checker SHA256:
`51f603402614f1abbc6f34122d28a1b90e4e8fe1e88cac63bc6bb6196b9b201f`.
Default MERE thresholds are `2**-13` for FP32, `2**-10` for FP16 and `2**-7`
for BF16; the normal-range MARE bound is ten times the corresponding threshold.
The same checker also handles special values, small ranges and cancellation;
these results are not reduced to a generic `allclose` tolerance.

Inputs include 4096 seeded normal samples, 8193 points across [-10,10], 513-point
samples from each applicable official finite range, NaN/Inf, signed zero,
very small values, finite dtype limits, and ±0.5/±2. Inputs are converted to
the tested dtype before constructing the reference. Numerical probes are padded
to complete tiles; they are **not full official shapes**.

FP32 passes with reuse 1/VF off, reuse 1/VF on, and reuse 2/VF on.
For the coefficient-corrected baseline, dense-grid MERE is about `6.64e-8`
and MARE `1.74e-6`. Original-coefficient normal-input MERE is about `0.2674`.
See the complete per-segment checker results, including small-value and
cancellation checks, in [evidence](evidence/summary.json).

Only tanh routes are in scope: official cases 4, 5, 6, 8, 10, 12, 14, 17, 19.
The 11 exact/Erf cases are **not implemented or validated** by this target.
Passing sampled ranges does not mean 9/9 or 20/20 official cases passed.

An additional FP32 tail probe (`N=2049`, tile 1024, 8 cores, unroll 1) passes
accuracy and leaves all guarded output outside the logical tensor untouched.
The generated copy lowering clips transfers to the global tensor's remaining
length even though this target does not pass explicit `real_shape`.
This verifies that probe, not every possible tail/partition.

## Local performance

`verified-camodel-smoke`: same FP32 input, N=31744, range [-2,2], one core,
reuse 1, default static allocation. Three cached timed launches after diagnostic
execution and warmup; compilation, synchronization diagnostics, input/output
preparation and comparison are outside timing. Device debugging is off.
All corrected variants pass each repetition and produce identical repeated outputs.

- Original coefficients, tile 15872/unroll 2/VF off: **9384 ticks** median
  (9384, 9425, 9356); accuracy fails, so this is a diagnostic comparison only.
- Corrected coefficients, same geometry: **9384 ticks** (9384, 9420, 9356),
  UB 253952 B.
- Corrected, tile 1024/unroll 2/VF off: **13555 ticks**
  (13493, 13555, 13609), UB 16384 B. Wide tile is about 1.44× faster here.
- Corrected, tile 15872/unroll 2/VF on: **9438 ticks**
  (9438, 9470, 9378), UB 253952 B. No convincing gain in this small sample.
- Earlier [stable-tanh implementation](../gelu-asctile-jit-20260907/candidate/gelu.py),
  tile 1024/**unroll 1**/VF off: **31341 ticks** (31270, 31341, 31418),
  UB 20480 B; all repeats pass and outputs repeat identically. Same input,
  dtype and core count, but different mathematics implementation and unrolling.
  The corrected wide-target configuration is about 3.34× faster than this
  configuration locally; this does **not** isolate an expression-only gain.

These are **CaModel ticks**, not NPU microseconds and not speedup over the
CANNBench reference. The original target's 150994944 elements / 72 cores
were compile-checked only. There is no demonstrated ≥1× CANNBench result.

## JIT/UB checks

All 18 combinations of reuse 0/1/2, static None/False/True, VF off/on compile
with `verify_sync=True`, `insert_sync=True`, `opt_level=3`, and diagnostic
`always_compile=True`, at target tile 15872/unroll 2. This is a compile matrix,
not 18 numerical passes. Requested/effective options and source/binary hashes
are saved in the individual result files.

- Default and `static_alloc=True` generate identical C++ in all six paired
  comparisons: this confirms the effective static default for this experiment.
- Static reuse 0 reports 1714176 B UB and is pruned before simulation.
- Static reuse 1/2 reports 253952 B, exactly the platform budget; VF does not
  reduce that allocation here. Fitting UB is not itself evidence of speed.
  This is not an inferred or rounded budget: four FP32 buffers of 15872
  elements give `4 * 15872 * 4 = 253952` B, corroborated by four 63488 B
  reservations in the corresponding non-static emitted code.
- Static False has no UB metadata. Read-only TPipe accounting finds 253952 B
  for reuse 1/2. Reuse 0 has conditional reservations the conservative parser
  cannot fully resolve. No static-False configurations were simulated.
- The older stable-tanh candidate at tile 15872/unroll 2 needs 317440 B and is
  pruned, not timed. The smaller valid baseline is recorded in Local performance.

No custom compiler pipeline, monkeypatch, handwritten AscendC, or low-level
compute was introduced. The shared source gate passes both new source files.
The `pyasc-cannbench-kernel` workflow drove independent-oracle validation,
high-level source checks and separation of local evidence from hardware claims.

## Reproduction and evidence

From the repository root:

```bash
source integrations/cannbench/comparisons/gelu-asctile-jit-20260907/env.sh
python integrations/cannbench/comparisons/gelu-target-corrected-20260907/run_suite.py numerical
python integrations/cannbench/comparisons/gelu-target-corrected-20260907/run_suite.py matrix
python integrations/cannbench/comparisons/gelu-target-corrected-20260907/run_suite.py perf
python integrations/cannbench/comparisons/gelu-target-corrected-20260907/run_suite.py extensions
python integrations/cannbench/comparisons/gelu-target-corrected-20260907/audit.py
```

The runner skips existing result files. Use a new `check.py --output` directory
for a fresh rerun. [CSV index](evidence/results.csv), [JSON index](evidence/summary.json),
[suite runner](run_suite.py), [audit](audit.py).
The initial `corrected-f32` attempt failed on a harness path assertion;
`corrected-f32-v2` is the successful corrected-path run. Do not count that setup
failure as a kernel defect. No submission credits were spent.

## Independent review and adjudication

OpenCode `dashscope/qwen3.7-max` reviewed full inline skill, kernel, harness,
summary and draft-report text without tools or execution. See
[review](workers/dashscope_qwen3.7-max/review.txt) and
[prompt/input hashes and session provenance](workers/dashscope_qwen3.7-max/provenance.json).
The review occurred while the optional old-stable timing probe was still pending;
it does not independently validate that later result.

The main agent independently checked its objections:

- Accepted the need to keep FP32, FP16, unsupported BF16, and promoted variants
  separate, and to avoid extrapolating samples to full official shapes. These
  boundaries were already explicit and remain enforced.
- Rejected the assertion that a PyTorch tanh-GeLU oracle is circular: tanh is
  the specified approximation route, and PyTorch is an independent
  implementation, not the target's own expression or coefficients. It does
  not establish exact-GeLU correctness, which is not claimed.
- Rejected the suggestion that matching UB capacity implies coarse accounting:
  explicit emitted buffer reservations account for the exact byte count above.
- Rejected the claim that a same-input, same-core local tile comparison is
  invalid merely because it is not NPU timing. It is valid within its stated
  CaModel scope; it cannot establish hardware speedup or multicore scaling.
- Retained the FP16 mechanism as an inference. Promoting intermediates is an
  empirical remedy for the tested samples, not an instrumented pass diagnosis.

No changes were made solely on the reviewer's authority.
