# CANNBench corrective reruns: provenance, diagnosis, and results

Date: 2026-09-03 (Europe/Moscow). Benchmark: `official-tasks` v1.1.1.
Hardware: 950PR. All submissions were private. Public benchmark entry point:
[CANNBench leaderboard](https://cannbench.com/leaderboard).

## Executive result

Both corrective submissions passed all selected cases with no anti-cheat
failures:

- manual target-corrected: [`job_a3e9153a2930`](https://cannbench.com/workspace/jobs/job_a3e9153a2930), **40/40**, score
  **135.084504**, geometric-mean speedup **0.525847×**;
- skill-driven corrected: [`job_028f05fd8ec8`](https://cannbench.com/workspace/jobs/job_028f05fd8ec8), **40/40**, score
  **128.725482**, geometric-mean speedup **0.457136×**.

The first run labelled “handwritten” was not a literal copy from the current
working directory. It was reconstructed from Git objects in the same
`compiler-team/pyasc` repository at commit
`4d1db41d61cabf565bca1cfb0b11ef5ec4f84c7f`, using
`python/test/asc2/target/test_gelu.py`, `test_addcdiv.py`, and `helpers.py`, and
then adapted to the CANNBench callable interfaces. The current checkout is at
`38a0770fba74b1cb328fc196b3cb2d2006dad0d8`; current `origin/v2` is
`030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d` and has renamed the directory to
`python/test/asctile/target`.

Accordingly, “handwritten” is corrected in this report to **target-derived
adapter**. The successful manual rerun is a **target-derived contract
completion**, not a verbatim target test.

## Result matrix

| Variant | Job | Cases | Score | GMean speedup | Anti-cheat |
|---|---|---:|---:|---:|---:|
| Target-derived adapter, first run | [`job_79c05b96ed9d`](https://cannbench.com/workspace/jobs/job_79c05b96ed9d) | 14/40 | 69.794947 | 1.023126× | 0 |
| Generated without skills | [`job_75a7fee4ae6f`](https://cannbench.com/workspace/jobs/job_75a7fee4ae6f) | 35/40 | 125.452428 | 0.616455× | 0 |
| Generated with skills, first run | [`job_ae3bfdefd087`](https://cannbench.com/workspace/jobs/job_ae3bfdefd087) | 36/40 | 124.607094 | 0.595558× | 0 |
| Manual target-corrected | [`job_a3e9153a2930`](https://cannbench.com/workspace/jobs/job_a3e9153a2930) | **40/40** | **135.084504** | **0.525847×** | 0 |
| With-skills corrected | [`job_028f05fd8ec8`](https://cannbench.com/workspace/jobs/job_028f05fd8ec8) | **40/40** | **128.725482** | **0.457136×** | 0 |

## Per-operator results

| Variant | Operator | Cases | Average speedup | Operator score |
|---|---|---:|---:|---:|
| Target-derived, first | ForeachAddcdivScalar | 13/20 | 1.189408× | 53.191518 |
| Target-derived, first | GELU | 1/20 | 0.880090× | 16.603429 |
| Without skills | ForeachAddcdivScalar | 20/20 | 1.082186× | 76.274837 |
| Without skills | GELU | 15/20 | 0.351157× | 49.177591 |
| With skills, first | ForeachAddcdivScalar | 20/20 | 0.920270× | 72.679093 |
| With skills, first | GELU | 16/20 | 0.385419× | 51.928002 |
| Manual corrected | ForeachAddcdivScalar | **20/20** | **1.147209×** | **78.626238** |
| Manual corrected | GELU | **20/20** | **0.241033×** | **56.458266** |
| Skills corrected | ForeachAddcdivScalar | **20/20** | **0.917337×** | **72.599692** |
| Skills corrected | GELU | **20/20** | **0.227804×** | **56.125790** |

The first target-derived run’s high geometric-mean speedup is not evidence of
a better complete solution: only 14 cases contributed valid performance data.
Correctness must be treated as a hard gate before comparing speed.

## Why the target kernels failed the broader contract

The upstream target tests are valid development tests but substantially
narrower than the selected CANNBench contract:

- target GELU covers one FP32 shape and the sigmoid/tanh-equivalent formula;
  it does not implement the CANNBench `approximate="none"` path or exercise
  BF16, wide ranges, tails, and special values;
- target Addcdiv fixes `scalar=0.5`, uses bounded nonzero denominators, and does
  not cover the BF16 and special-scalar matrix;
- the initial adapter retained direct target-dtype arithmetic where CANNBench
  required a wider internal compute contract.

Observed failures in `job_79c05b96ed9d`:

- Addcdiv: six BF16 compile/runtime failures from unsupported direct BF16
  division, plus one FP16 scalar-zero NaN-position mismatch;
- GELU: six BF16 compile/runtime failures and thirteen precision/cancellation
  failures. Only the all-zero tanh route passed.

The generated first attempts narrowed the problem to GELU. The no-skills arm
failed cases 5, 8, 11, 12, and 20; the skills arm failed cases 5, 8, 11, and
20. These were FP32 negative-tail cancellation failures, except the additional
BF16 NaN-position mismatch in no-skills case 12.

## Corrective changes

### Manual target-derived contract completion

- Addcdiv promotes FP16/BF16 inputs to FP32, performs division and scaling in
  FP32, and casts once at output.
- GELU exact uses a stable Numerical Recipes `erfc(abs(x)/sqrt(2))` Horner
  form and selects positive/negative output forms without `1 + erf`
  cancellation.
- GELU tanh uses the bounded-exponential sigmoid identity
  `x * exp(min(s,0)) / (1 + exp(-abs(s)))`.
- The manual implementation uses performance-aware wide/safe dispatch. It
  produced the best corrected score, 135.084504.

### Skill and worker corrections

The repository skill now states that an upstream target test is a reference,
not complete CANNBench contract evidence. It also requires adversarial camodel
checks for numerically sensitive formulas and records the measured GELU UB
budget.

The first measured repair had the right stable formula but chose exact tile
2048 with unroll factor 2. Pinned-v2 lowering measured 795392–819968 bytes of
UB use against a 253952-byte limit. The accepted repair uses exact tile 512,
measured at 198848–204992 bytes; tanh remains tile 1024 at 172032–184320 bytes.

Accepted provenance:

- repair model/session: Qwen, `ses_f9c952882ffeAnIEP58bPaIy7l`;
- independent accepted review: Qwen, `ses_f9c8e48c5ffeFcOrr6Qvqf72Tj`;
- both accepted phases loaded the repository-local
  `pyasc-cannbench-kernel` skill;
- a GLM review session, `ses_f9c93e480ffeq5Hu4yhp750SLO`, loaded the skill
  but timed out and was not accepted;
- accepted GELU candidate SHA-256:
  `794e4ad018ba82487b36247eff5c3af5e1ee4925f384a09bb8d4a194a77fc1de`.

The global skill-provenance gate was also corrected to accept a required skill
set as a subset of loaded skills, so legitimate additional skill calls no
longer cause false rejection.

## Validation ladder

```text
pyasc target references
        │
        ├── manual CANNBench contract completion
        └── OpenCode generation/repair + native skill-call provenance
                         │
                  Python/static checks
                         │
              pinned-v2 QEMU compile gate
                   40/40 per bundle
                         │
            Ascend950PR_9599 adversarial camodel
                 both operators passed
                         │
                  CANNBench 950PR
                   40/40 per job
```

Local evidence before submission:

- target-corrected QEMU: GELU 20/20 and Addcdiv 20/20;
- skills-corrected QEMU: GELU 20/20 and Addcdiv 20/20;
- all unique pinned-v2 specializations compiled;
- both bundles passed the adversarial AArch64 camodel suite with
  dtype-specific tolerances and exact NaN/Inf-position checks;
- cumulative camodel suite ticks: manual GELU 40386, manual Addcdiv 16201,
  skill GELU 38344, skill Addcdiv 15277.

Camodel ticks are simulator totals for small adversarial suites and are not
silicon speedups. CANNBench case-level 950PR measurements are the performance
oracle.

## What CANNBench evaluates—and what it does not know

CANNBench installs the self-contained submission wheel and calls the exported
operator functions on real 950PR hardware. For each selected operator it
checks compilation/runtime, numerical correctness, anti-cheat conditions, and
case-level latency/speedup against benchmark baselines.

The evaluator does not automatically discover `pyasc-skill-stack`, prompts,
or model sessions. It evaluates the submitted program, not the generation
history. The distinction between “without skills” and “with skills” is made
auditable by local provenance: immutable candidate hashes, OpenCode event
logs, session IDs, required native skill calls, and validation artifacts.

## Interpretation and next iteration

The integration defect is fixed for this two-operator intersection: both the
manual and skill-driven corrected bundles are 40/40. The remaining gap is
performance, especially GELU:

- evaluate exact GELU tile 512/unroll2 against 1024/unroll1 using the same
  stable formula and full 20-case correctness gate;
- evaluate Addcdiv tile 1024 against size-dispatched 2048; the manual corrected
  arm’s 1.147209× average versus 0.917337× for the skill arm shows a concrete
  tuning opportunity;
- preserve correctness as a hard prerequisite, then compare case-level
  silicon latency rather than aggregate scores from partial runs;
- keep the source/runtime distinction explicit while upstream v2 transitions
  from `asc2` to `asctile`.

## Artifacts

- Initial diagnosis: `reports/initial-diagnosis.md`
- Final report: `reports/final-cannbench-reruns.md`, `.html`, and `.pdf`
- Corrective manifest: `correction/MANIFEST.json`
- QEMU reports: `correction/local_validation/`
- Camodel reports: `correction/camodel/`
- Remote payloads and logs: `correction/remote_runs/`
- Immutable submission archives: `correction/submissions/*.zip`
- Worker traces: `correction/with_skills/*.log`
