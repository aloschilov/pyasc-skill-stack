# High-level FP32 exact GeLU remedy

2026-09-07. This supersedes the *local accuracy* blocker for the old direct-Erf
candidate, not the pending whole-package/CANNBench qualification.

## Selected FP32 refinement

[exact_fast_kernel.py](exact_fast_kernel.py) uses **six recurrence levels and
the switch x < -3**, retaining the same high-level template. Its source SHA256
is `a9779265f28440e138d19c2db40bc5465fe5d5d4b9c89e33195733005eae6f33`.
It passes the original dense/normal/range/special-value FP32 probes, independent
random negative tails, the switch grid and adjacent representable values, and
the repeated eight-core guarded-tail test. Tile 1024, unroll 1, reuse 1,
static default, VF on; UB is **49152 B**, versus 73728 B for the 12-level variant.

The dense-grid MERE is `8.042e-7`, MARE `8.456e-5`; both meet official FP32
bounds. The fine switch grid has maximum absolute error about `3.06e-7`.
CPU double [tail-only convergence](evidence/tail-math-fast.json) on 100001 points of [3,14] has maximum
relative error `2.4303e-5`. Extending the ordinary Erf route down to -3 trades
some numerical margin for fewer divisions, while retaining measured tolerance.
The 12-level implementation below remains the validated diagnostic baseline.
The selected refinement has not yet received whole-package official-shape
qualification. A subsequent [selected-dispatch compile sweep](evidence/selected-exact-official-compile.json)
passes all 11 exact-case requests: FP32 uses the six-level kernel (49152 B UB),
FP16/BF16 use the simple FP32-intermediate Erf kernel (98304 B UB).
The earlier compile sweep below belongs to the 12-level variant. Neither sweep
executes full official shapes or verifies the final combined host wrapper.

The six-level refinement takes **4966 CaModel ticks** median on the matched
timing probe described below, with all three timed outputs correct and
repeatable. This is about **28.2% fewer ticks** than the 12-level fix (6917),
but about **41.0% more** than the inaccurate direct-Erf baseline (3521).
Select the refinement for the next FP32 exact candidate; do not claim an
improvement over the CANNBench hardware reference.

## Finding in ops-nn

Local ops-nn revision `f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307`:
[gelu_v2_dag.h](/home/aloschilov/workspace/ops-nn/activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h)
uses `Vec::Erf<float>` for FP32 exact and a separate default `ErfFast` for
low-precision inputs promoted to FP32. The installed CANN 9.0
[atvoss/vec.h](/usr/local/Ascend/cann-9.0.0/aarch64-linux/pkg_inc/op_common/atvoss/util/vec.h)
implements `Vec::Erf` with
`ErfConfig{ErfAlgo::SUBSECTION_POLYNOMIAL_APPROXIMATION}`.
The backend default is instead `PADE_APPROXIMATION`.

Pinned pyasc v2 `adadd7d66ed0ee16d33d79487bf584899a26ef1e` exposes
`asctile.erf(input)` without that algorithm selector; its emitted call uses the
default Erf configuration. This is a concrete API/configuration difference,
not proof of an allocation-pass bug. The ops-nn FP32 postprocessing also scales
x by 0.5 separately before multiplying by `1+erf`, as our exact template now does.
The specialized ops-nn algorithm was **not executed** in this experiment, so no
accuracy or speed claim for it is inferred solely from its name/source.

No ops-nn, installed CANN, or pyasc compiler files were changed. Source hashes
and CPU convergence results are in [tail-math.json](evidence/tail-math.json).

## Implemented remedy

[exact_stable_kernel.py](exact_stable_kernel.py) keeps public AscTile types,
copies, tiled loops, casts and expressions. It retains public `erf` in the
central/positive range and uses a cancellation-free normal tail for x < -2:

```python
central = (x * 0.5) * (asctile.erf(x * 0.7071067811865476) + 1.0)
a = asctile.maximum(-x, 2.0)
remainder = asctile.full([tile_length], 0.0, asctile.float32)
for order in asctile.static_range(12, 0, -1):
    remainder = order / (a + remainder)
tail = (x / (a + remainder)) * asctile.exp((-0.5 * a) * a) * 0.3989422804014327
out = asctile.where(x < -2.0, tail, central)
```

This is the normal-tail Mills-ratio continued fraction, obtained from the
standard [DLMF 7.9.1 erfc continued fraction](https://dlmf.nist.gov/7.9.E1)
by substituting `z=a/sqrt(2)`. It is **not** a tanh substitution, benchmark-case
dispatch, copied ops-nn special-function polynomial, handwritten register
implementation, or a compiler patch. It is an independent high-level numerical
remedy motivated by the unavailable ops-nn accuracy configuration.

Twelve recurrence levels control numerical convergence, not compulsory tile
unrolling. CPU double-formula checks over 200001 points on a in [2,14] give
maximum relative normal-tail errors of approximately 0.00277 (4 levels),
0.000599 (6), 0.000156 (8), 0.00001515 (12), and 0.00000202 (16).
Twelve provides margin below the checker tolerance without 16 divisions.
These CPU results are separate from compiled-kernel evidence, and are sampled
convergence checks, not a global floating-point error proof.

## Compiled accuracy

`verified-camodel-smoke`: native AArch64/CPython 3.11, pinned pyasc, CANN 9.0,
CaModel Ascend950PR_9599, synchronization ON. Independent oracle is PyTorch
`gelu(approximate="none")` in double and same dtype, using the official checker.

- The initial FP32 probe (tile 2048, unroll 1, reuse 1, VF on) passes all
  segments: 4096 normal values, 8193 dense [-10,10] values, applicable official
  range samples, NaN/Inf, tiny values and finite dtype limits. UB: 147456 B.
- Dense-grid MERE is `2.044e-7`; MARE is `1.512e-5`. The prior direct-Erf
  kernel fails that grid. Do not interpret early-success default-zero
  cancellation counters as separately measured error-count reductions.
- Additional FP32, FP16 and BF16 probes pass seeded random negative tails,
  a fine transition grid around -2, adjacent representable values at the
  switch, dense/normal/range and exceptional-value samples. Tile 1024,
  unroll 1, reuse 1, VF on; UB: 73728 B each.
- FP32 tail/repeat probe: N=2049, tile 1024, eight cores, same JIT settings;
  valid outputs pass and the cached second execution agrees including NaNs.
  Output storage outside the logical tensor is untouched, including its guard.
- All 11 exact official-case compile requests pass for their input lengths
  and dtypes, tile 1024/unroll 1/VF on, 72 requested cores. Input length is a
  runtime argument; these are not 11 distinct math specializations, nor full
  multicore numerical executions. Every request reports UB 73728 B.

These are sampled numerical checks, not full official-shape executions. The
source's compare/select tiles have byte sizes divisible by 256, avoiding the
known partial-vector destination-size hazard. The full tile is allocated even
when its global valid length is shorter.

## Performance and qualification boundaries

The continued fraction adds divisions; an accuracy fix is **not automatically
a speedup**. Hardware latency and >=1x CANNBench performance remain unmeasured.
Local warmed matched-input timing is recorded separately when complete.
The old failed kernel remains intact in [exact_kernel.py](exact_kernel.py).

Matched local FP32 timing: N=2048, range [-6,2], tile 1024, unroll 1, one core,
reuse 1, static default, VF on. Diagnostic compilation/execution and warmup
are outside the three cached timed launches; output preparation and checker
work are also outside the interval. The direct-Erf baseline takes **3521 ticks**
median but fails accuracy on that input. The 12-level remedy takes **6917 ticks**
and passes each timed repetition. Thus this initial correctness remedy costs
about 1.96x as many Model ticks, not a performance improvement. These are not
NPU microseconds or CANNBench reference speedups.

Only the exact diagnostic kernel is repaired here. The final exact+tanh
submission wrapper, evaluator-compatible wheel/ABI, all 20 host routes and
final artifact hashes still require validation before the one authorized
submission. Do not resubmit the old failed source or claim 20/20 hardware
success from these probes. No credit was spent during this repair.

Recommended dtype dispatch: use the six-level remedy for **FP32 exact**.
For FP16/BF16 exact, retain this campaign's simple [exact_kernel.py](exact_kernel.py)
with FP32 intermediate arithmetic, which already passed its wider local checks.
Do not impose the extra divisions on those dtypes without a measured need.
The remedy's FP16/BF16 checks remain diagnostic evidence, not a reason to force
one mathematical path for every dtype.

The `pyasc-cannbench-kernel` workflow drove source-gate validation, independent
oracle tests and conservative qualification labels. No skill instruction was
changed to make a failed test count as a pass.

## Independent review and adjudication

OpenCode `dashscope/qwen3.7-max` independently reviewed full inline source,
skill text, mathematical convergence, numerical/guard evidence and report draft;
it did not execute tools. [Review](workers-remedy/dashscope_qwen3.7-max/review.txt)
and [model/session/prompt/input provenance](workers-remedy/dashscope_qwen3.7-max/provenance.json)
are preserved. It confirmed the recurrence and oracle and identified no concrete
sampled-correctness blocker. The main agent checked, rather than blindly adopted,
that review:

- Its phrase "No discontinuity" is too strong: a finite continued fraction
  can cause a small switch mismatch. The measured FP32 boundary absolute errors
  are about 7e-7 and pass tolerance; strict continuity was not established.
- Equal tensor shapes alone do not address the known `where` lowering hazard.
  The tested full-tile allocation sizes are multiples of 256 bytes; that is the
  relevant alignment condition, including when a transfer has a shorter tail.
- BF16 exact has three official range samples (3/9/16), not two as the review
  said. None is a full official-shape numerical execution.
- The finite truncated recurrence approximates the exact-mode mathematical
  function within measured error; it is not an identity at finite depth.
- Performance and complete submission qualification remain separate gates.

The six-level refinement received a [second independent review](workers-fast/dashscope_qwen3.7-max/review.txt),
with separate [provenance](workers-fast/dashscope_qwen3.7-max/provenance.json).
No numerical blocker was identified; the main agent adjudicated its details:

- Persisted the fast CPU convergence sweep separately; it is not the initial
  12-level-domain sweep and is not a Model error measurement.
- The proposed bound on the mathematical switch jump was not adopted: sampled
  pointwise absolute errors are not a global continuity proof, and an absolute
  error must not be compared directly with a relative tolerance.
- The selected-dispatch 11-case compile sweep completed after review input
  capture. Whole-package/hardware qualification remains pending.
- Rejected the suggested performance bias from fewer negative-tail lanes:
  these are eager tile expressions followed by `where`. The emitted VF code
  computes every continued-fraction division with the active tile mask before
  the final Select. Both formulas execute on all active lanes; branch fractions
  do not skip the divisions. The matched performance comparison remains valid
  within its small-probe Model scope.
- The eight-core tail test has one cached replay, not a repeated fresh-process
  stability campaign. This limitation remains explicit.
