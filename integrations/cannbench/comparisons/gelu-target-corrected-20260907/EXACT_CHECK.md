# Exact GeLU: local-first submission decision

Follow-up: [EXACT_REMEDY.md](EXACT_REMEDY.md) documents a locally passing
high-level FP32 remedy. The failure and hold decision below describe the
earlier unchanged direct-Erf candidate, not that new implementation.

2026-09-07. Decision: **check locally before diagnostic submission**. The
new target-style exact kernel reproduces an existing FP32 accuracy limitation;
the combined 20-case candidate must not be submitted unchanged under the current
local-qualification gate. No new submission or credit expenditure occurred.

## What was actually tested

[exact_kernel.py](exact_kernel.py) uses public AscTile tiled copies, a contiguous
target-style multicore partition, explicit valid lengths, FP32 intermediate
arithmetic and cast-back:

```python
x = raw.to(asctile.float32)
out = (x * 0.5) * (asctile.erf(x * 0.7071067811865476) + 1.0)
```

Source/runtime: pyasc v2 `adadd7d66ed0ee16d33d79487bf584899a26ef1e`, native
AArch64, CANN 9.0, CaModel Ascend950PR_9599. The original upstream target and
compiler were not modified. The shared high-level source gate passes.

Bounded numerical probes: tile 8192, unroll 2, one core, reuse 1, static default,
VF off; a separate FP32 VF-on control. Synchronization remains on and diagnostic
compilation uses `verify_sync=True`. Result files retain requested/effective
options, runtime path, source/harness/checker/binary hashes and UB metadata.

Independent oracle in [check.py](check.py): PyTorch `gelu(approximate="none")`
in double precision and same dtype, evaluated with the official CANNBench
precision checker. Inputs are converted to the target dtype before constructing
references. This is compiled-kernel execution, not merely a CPU formula check.

For each dtype: 4096 seeded normal values, 8193 points in [-10,10], 513-point
samples from applicable official finite ranges, official NaN range, extra
NaN/Inf/signed-zero/tiny values, finite dtype limits and ±0.5/±2. Each input is
zero-padded to 16384 elements. These are sampled probes, **not full official
shapes, full multicore qualification, or 20 hardware cases**.

## Observations

`verified-camodel-smoke`:

- FP32, VF off: compiles (131072 B UB), but the dense negative-tail segment
  fails. The normal segment, all sparse official-range samples, special values
  and finite-limit samples pass.
- FP16, FP32 intermediates: all sampled checks pass, 98304 B UB.
- BF16, FP32 intermediates: all sampled checks pass, 98304 B UB.
- FP32, VF on: compiles (131072 B UB), but the dense negative-tail segment
  still fails. This control does not rescue the exact route.

FP32 dense-tail official checker reports cancellation-error counts **51 for
compiled output versus 5 for CPU**, among 484 cancellation-range elements.
The failing grid has 8193 elements; its normal error metrics alone are not a
substitute for the checker's cancellation/small-value criteria.

A concrete point, already present in that grid:

```text
x                         = -4.111328125
compiled output           = -8.111295755952597e-05
double exact-GeLU reference= -8.086769955458579e-05
CPU FP32 exact-GeLU        = -8.086790330708027e-05
```

The compiled relative error at this point is about 0.303%. Sparse 513-point
samples over cases 11/20 ranges pass while the dense probe fails; sparse-range
success is not evidence of full-case success.

Scaling x by 0.5 first passes the tested large positive finite limits, but does
not repair cancellation in `1 + erf(...)`. This is consistent with the earlier
isolated [public-Erf accuracy investigation](../gelu-upstream-diagnostics-20260907/issues/accuracy.md).
The new full kernel was not instrumented to isolate primitive ULPs, so this
retest does **not** newly prove a particular pass defect, a primitive-contract
violation, or identical behavior on CANN 9.1 hardware.
The VF-off emitted C++ retains separate `Muls(..., 0.5, ...)` before the
final `Mul`; thus early scaling is not merely an unverified source spelling.

## Strategy and remaining work

Stop before a full performance sweep or remote upload of this unchanged exact
candidate. FP16/BF16 sampled accuracy is encouraging but does not remove the
FP32 exact gate for the combined job. Local Model wall time and diagnostic
ticks are not reported as warmed performance or hardware speedup here.

The historical public-Erf implementation already had hardware exact-FP32
failures in [job_cdee9a024da2](https://cannbench.com/workspace/jobs/job_cdee9a024da2).
That was a different kernel/environment; it supports caution, not a claim that
the new kernel has been evaluated remotely. Repeating a credit-consuming
diagnostic has low expected value without a new accuracy remedy or a concrete
hardware-versus-Model question.

Retain the single planned exact+tanh run, blocked on exact FP32 qualification.
Do not silently remove exact cases, substitute tanh, or introduce a low-level
fallback. Proceed after a supported high-level accuracy change is locally
validated, or after a separately authorized targeted diagnostic exception.
No compiler patch is made in this investigation.

The `pyasc-cannbench-kernel` workflow required the independent oracle and
adversarial local gate; the measured failure stops promotion, not compilation
success or an unverified worker opinion.

Reproduce with the campaign environment and
`python integrations/cannbench/comparisons/gelu-target-corrected-20260907/run_suite.py exact`.
The runner skips existing result files: use fresh `check.py --output` paths for
fresh execution. See [raw results](evidence/exact-summary.json).

## Independent review

OpenCode `dashscope/qwen3.7-max` reviewed the full inline skill, source, harness,
four-probe results and this draft report, without tool access or execution.
[Review](workers-exact/dashscope_qwen3.7-max/review.txt) and
[session/input provenance](workers-exact/dashscope_qwen3.7-max/provenance.json)
are retained. Main-agent adjudication:

- The reviewer confirmed correct exact-mode oracle selection and separation
  of compiled/sampled accuracy from full-shape and hardware claims.
- Its suggestion to consider a direct `erfc` formulation is mathematically
  relevant, but availability must not be assumed. Source inspection and a
  fresh import of this pinned runtime confirm public `asctile.erf` exists,
  while `asctile.erfc` and `LocalTensor.erfc` do not. A genuinely accurate
  direct erfc/CDF API could be future work; spelling it as `1-erf` merely
  reintroduces the cancellation problem.
- Early half scaling was intended only to address finite-positive overflow,
  not cancellation. The report already states this, and emitted ordering was
  additionally inspected. No claim of a cancellation remedy is warranted.
- Other JIT cells, tails/multicore and full-shape qualification remain untested
  for this new exact kernel. One reproducible current-configuration accuracy
  failure is enough to withhold this candidate; it is not proof that every
  high-level implementation or configuration must fail.
- The review's stronger primitive-defect attribution is not adopted. The
  pointwise new observation is an end-to-end kernel result; the earlier
  isolated primitive investigation is separate evidence.

The hold applies to the current candidate, not a permanent prohibition on
diagnostic submissions or a claim that no remedy can exist.
