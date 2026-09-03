# Initial diagnosis: pyasc target intersection on CANNBench

Date: 2026-09-02

## Executive finding

The first arm was not copied verbatim from the current working-tree directory
`/home/aloschilov/workspace/pyasc-fork/python/test/asc2/target`. It was built
from the same repository's Git objects at commit
`4d1db41d61cabf565bca1cfb0b11ef5ec4f84c7f`, then adapted to the CANNBench
interfaces. That commit contained both
`python/test/asc2/target/test_gelu.py` and `test_addcdiv.py` and was visible in
the then-fetched v2 history. The local checkout itself is currently the
unrelated `apply-adam-target` branch at `38a0770f`; it contains the GELU file
but not the Addcdiv file.

The submission runtime was built from v2 commit
`ac1222a48c8914d3f81297c7570d1a84f0f26778`. The arithmetic in the two target
tests is unchanged between `ac1222a4` and `4d1db41d`; their differences are
test-harness naming only. Nevertheless, using different source and runtime
commits made the provenance needlessly ambiguous. The corrected run will pin
both to `ac1222a4`.

After a fresh fetch on 2 September, `origin/v2` is
`030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d` and has force-updated history. The
API has been renamed from `asc2` to `asctile`, so the current paths are
`python/test/asctile/target/test_gelu.py` and `test_addcdiv.py`. Their operator
coverage remains narrow in the same material ways described below.

## What was source-derived and what was added

The submitted files under `handwritten/candidates/` retained the target-test
arithmetic and tiling intent. The adapter added CANNBench signatures, arbitrary
shapes and tails, a TensorList host loop, a runtime Addcdiv scalar, and all
CANNBench dtypes and attributes. Those are semantic extensions, not transport
only. The original target tests do not claim that wider contract:

- target GELU tests one FP32 shape and one sigmoid-equivalent tanh
  approximation; it has no `approximate="none"` implementation, BF16 route,
  special-value matrix, or wide-range accuracy requirement;
- target Addcdiv tests FP32/FP16, bounded nonzero denominators and fixed
  `value=0.5`; it has no BF16 route or arbitrary scalar/special-value matrix.

Therefore the upstream target kernels can pass their own tests while the
adapted CANNBench arm fails. The 14/40 result is primarily an incomplete
contract adapter, not evidence that the original target tests are broken.

## First-run results and root causes

Private job `job_79c05b96ed9d` on 950PR passed 14/40 cases, scored 69.794947,
and reported zero anti-cheat failures.

- ForeachAddcdivScalar: 13/20. Six BF16 routes fail compilation because direct
  pyasc v2 division accepts FP16/FP32 but not BF16. Case 17 fails NaN-position
  correctness under FP16 arithmetic with scalar zero. Promoting inputs to FP32
  inside the kernel, as the generated candidates already do, addresses both
  classes and has independently passed 20/20 on CANNBench.
- GELU: 1/20. Six BF16 routes fail compilation because direct `exp` does not
  accept BF16. The remaining failures are semantic/numerical: the adapter
  ignores `approximate`, applying the target tanh formula to exact-erf cases,
  and the sigmoid/exp form is unstable or insufficiently close to Torch over
  CANNBench's adversarial ranges. The sole pass is the all-zero tanh case.

The generated arms expose a separate skill-stack gap. They compile all 40
routes and ForeachAddcdivScalar passes 20/20, but GELU still fails adversarial
accuracy:

- without skills, `job_75a7fee4ae6f`: GELU 15/20, total 35/40;
- with skills, `job_ae3bfdefd087`: GELU 16/20, total 36/40.

The skill-guided candidate's direct erf/tanh formulas cancel for sufficiently
negative FP32 values. A later stable-tail candidate avoids cancellation but
still differs enough from Torch in cases 5, 8, 11 and 20, and mishandles the
BF16 Inf/NaN route in case 12. This proves that the current compile-only
20-route gate plus small, tolerant camodel smoke is not a sufficient
qualification gate.

## Corrective plan

1. Rebuild the target-derived arm from the exact `ac1222a4` target files and
   record every semantic extension explicitly.
2. Add FP32 internal arithmetic for Addcdiv half/BF16 inputs.
3. Complete GELU's two-mode CANNBench contract with a hybrid formulation:
   direct pyasc erf/tanh where it is accurate, stable negative-tail evaluation
   where cancellation would occur, and exact preservation of Inf/NaN behavior.
4. Extend the CANNBench skill and numerical harness so adversarial ranges,
   strict dtype-specific thresholds, signed zero, and NaN/Inf positions are
   checked before promotion.
5. Run all 20 compile routes per operator, targeted camodel numerical probes,
   then private 950PR reruns for the corrected target-derived and skill-guided
   arms. Preserve both failed and replacement job evidence.

## Evidence locations

- first-run payloads: `remote_runs/`
- immutable first-run archives: `submissions/*.zip`
- target-derived candidates and provenance: `handwritten/`
- generated candidates and provenance: `no_skills/`, `with_skills/`
- exact benchmark contracts: `../../tasks/gelu/` and
  `../../tasks/foreach_addcdiv_scalar/`

