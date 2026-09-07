# High-level AscTile GeLU: JIT flags, correctness and lowering

[Уточнение о лишних VF-store](VF_STORE_REPORT.md) разделяет backend Erf correctness,
VF materialization, JIT compile/sync evidence и границы pass-анализа.
Объём опубликованных evidence описан в конце уточнения; остальные исходные
ссылки ниже могут вести на локальные, ещё не опубликованные artifacts.

## Outcome

Implemented a compact [target-style GeLU](candidate/gelu.py) using only public
AscTile compute APIs. **It is not eligible for submission:** the public-erf
FP32 exact path fails dense negative-tail checks for **all 18 JIT settings**.
No CANNBench submission was created, no credit was spent, and no new hardware
speedup is claimed. The ≥1× objective has not been established.

- All **20/20 official host dispatches** compile on both pinned runtimes:
  native AArch64/CPython 3.11 and evaluator-compatible x86_64/CPython 3.12.
- All **108/108** dtype/mode/JIT combinations compile with synchronization
  verification: 3 dtypes × 2 modes × 3 reuse levels × 3 static settings × 2 VF settings.
- Reduced numerical samples covering all 20 case ranges pass across **6/6 routes**.
  These are not executions of every full benchmark shape.
- A denser 4097-point check over [-10,10] exposes the FP32 exact failure;
  **0/18** JIT configurations pass it. A separate 64-lane reproducer isolates
  the numerical cause, without a tiled loop, allocation reuse or VF fusion.

The geometric tuning search and final NPU submission were stopped at this
mandatory numerical gate. There is no qualified full-GeLU candidate from
which to select the two fastest configurations. Finite-input performance
controls below are diagnostic measurements, not promotion evidence.

An additional repeated-execution/tail check also encountered bounded Model
timeouts. It did not complete multicore qualification; details below keep this
separate from the already established numerical blocker.

## Revisions and implementation

Source/runtime pin: `adadd7d66ed0ee16d33d79487bf584899a26ef1e` from
`compiler-team/pyasc` v2. The [runtime audit](evidence/audit.json) compares all
1194 archived source files in each isolated build tree with the pinned Git
archive; both trees match. Native AArch64/CPython 3.11 was built with the local
LLVM reference. The evaluator-compatible CPython 3.12/x86_64 wheel was built
successfully with QEMU and the existing x86 LLVM installation.
[ABI/import verification](evidence/runtime-x86.json) confirms that the loaded
library exactly matches the new wheel; [x86 dispatch/lowering replay](evidence/official-x86.json)
passes 20/20. The upstream archive build retains package version `1.1.1`, so
version text is not revision proof: use the archive and wheel SHA256 audit.
The wheel remains a local artifact under `runtime/wheels/`, not a new submission.
The original pyasc checkout and previously measured kernels were not replaced.

Candidate SHA256: `82da6f0be5cb38f0cbc4a748250a9d7c2e5b007d876c323a77d16a0f30ead463`.
The unchanged callable is `gelu(x, approximate="none")`. The source uses
`@asctile.jit(reuse_alloc=1)`, FP32 tile arithmetic, `real_shape` tails,
`min(size-offset,tile)`, and a bounded grid-stride launch. Exact mode calls
`asctile.erf`; tanh mode uses a stable exponential identity. There is no
authored C++, inline VF, legacy `asc`/`asc2` compute, polynomial replacement,
custom JIT, or compiler monkeypatch. The [host helper](candidate/_pyasc_runtime.py)
only selects the NPU platform. Local numerical probes call the same JIT
kernel on Model; host dispatch is checked separately with metadata tensors.

Local execution uses **CANN 9.0 / Ascend950PR_9599**, whose installed platform
configuration specifies 72 vector cores and 253952 bytes UB. Historical
CANNBench jobs used CANN 9.1 and another 950PR variant. Local failure is not
a claim that the untested remote runtime produces identical outputs.

## Why exact GeLU is blocked

The [minimal source](numeric_repro.py) executes 64 identical FP32 values with
`reuse_alloc=0`, `static_alloc=False`, `vf_fusion=False`, and sync verification.
[Measured result](evidence/numeric-reproducer.json):

| Quantity | Value |
| --- | ---: |
| Input x | -4.111328125 |
| Scaled z | -2.9071478843688965 |
| Backend Erf(z) | -0.9999605417251587 |
| Correctly rounded FP32 Erf(z) | -0.9999606609344482 |
| Actual GeLU | -0.00008111295755952597 |
| Float64 golden GeLU | -0.00008086769955458579 |
| GeLU using correctly rounded Erf | -0.00008086790330708027 |
| Checker MARE | 0.0030290842 (about 0.303%) |
| Cancellation failures: candidate / CPU | 64 / 0 |

The backend Erf error is two float32 ULPs. Subtracting almost equal values in
`1 + erf(z)` magnifies it. The correctly rounded Erf path matches the CPU
result at this point, so this is **not** proof that every FP32 implementation
must fail. It is a measured limitation of this public API/backend composition.

The dense matrix also records failures outside this one point. Its saved
arrays and checker details are in [the complete 18-setting summary](evidence/numerical-dense-exact/summary.json).
Changing allocation, fusion, tile geometry or core count cannot be accepted
as a repair without rerunning this gate. We did not substitute tanh for exact,
hide a polynomial in a helper, or relax the checker.

Required upstream direction: a more accurate supported Erf implementation,
or a high-level cancellation-resistant exact GeLU/CDF/erfc operation with an
appropriate lowering. No pyasc compiler change was made in this task.

## JIT options and lowering

The matrix tests `reuse_alloc={0,1,2}`, `static_alloc={None,False,True}` and
`vf_fusion={False,True}` at identical tile1024/unroll1 geometry. It retains
requested options, actual compiler options, pass calls, generated AscIR/C++,
binary hashes and memory metadata in [108 compile results](evidence/compile-matrix/summary.json).
Synchronization stays enabled; `opt_level=3`, no device debugging, and the
C310 default vector-register width of 256 bytes are used.

[Public JIT tests](evidence/jit-cold/result.json) prove default/decorator options,
invocation overrides, a cache hit without compiler invocation, and agreement
with forced compilation. The separate [base-first test](evidence/jit-base-first/result.json)
reproduces an inherited option-discovery cache defect: calling the base JIT's
`get_config_keywords()` first makes `reuse_alloc` unknown. Extraction itself
uses the concrete compiler option class. No adapter was loaded to hide it.

The [lowering report](LOWERING.md) provides source locations and before/after
IR for two independent performance limitations:

1. **False VF outputs:** `FindVFGroup` marks intermediate arithmetic results
   as group outputs. The existing nested `EliminateDataTransfer` forwards
   loads but keeps final stores to those declared outputs, retaining UB and
   synchronization overhead. The pass is present, not missing.
2. **Erf library boundary:** `LowerMath` chooses the LocalTensor library ABI;
   Erf has no register conversion and splits the surrounding VF groups.
   Transfers required by this ABI are distinct from redundant intermediate stores.

`static_alloc=None` and `True` produce identical C++ for each corresponding
route/settings pair. `False` uses TPipe; empty compiler memory metadata means
**unknown**, not zero. The lowering analyzer derives explicit TPipe buffer
reservations where possible. C310 Erf's scratch descriptor was investigated:
the inspected implementation does not consume that scratch storage, so no
scratch-overlap corruption is claimed.

Static IR/C++ counts do not prove every transfer survives Bisheng or dominates
runtime. No compiler patch or machine-instruction-level optimization claim is made.

## Finite-input warmed performance controls

Controls use FP32, 1024 finite values in [-2,2], tile1024, unroll1, one core,
static allocation default, one warm-up and three cached timing samples.
Compilation and diagnostic instrumentation are outside timed samples.
[Individual measurements](evidence/performance-controls/summary.json) record ticks and
the repeated-output hash comparison. These small controls do not rank all six routes or
replace the blocked full geometry search; simulator ticks are not NPU microseconds.

| reuse_alloc | vf_fusion | Exact median ticks | Exact UB bytes | Tanh median ticks | Tanh UB bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 (framework baseline) | False | 2446 | 24576 | 2440 | 57472 |
| 0 | True | 2500 | 24576 | 2387 | 57472 |
| 1 (target-style baseline) | False | 2381 | 12288 | 2374 | 20480 |
| 1 | True | 2376 | 12288 | 2380 | 20480 |
| 2 | False | 2375 | 12288 | 2388 | 16512 |
| 2 | True | 2376 | 12288 | 2342 | 16512 |

UB comes from matched compile configurations in the lowering analysis, not
from timing instrumentation. Reuse substantially reduces allocated UB, but
these tiny controls show only modest and variable timing changes. VF fusion
is not uniformly faster, and reuse2 is not automatically better than reuse1.
Three samples at one size are insufficient to select a production winner.
The readable candidate therefore retains the target-style `reuse_alloc=1`
baseline; no performance-selected dispatch has been promoted.

## Repeated execution and tail safety

The [structural harness](structural_probe.py) starts with 2049 finite values,
tile1024, one core, then plans 9217 values on eight cores. It changes inputs,
poisons outputs, and repeats calls with synchronization verification enabled.
The [recorded outcome](evidence/structural/summary.json) is **not a pass**:
FP16 exact reuse1/reuse2 each show five completed launches (three single-core,
then two eight-core) and an incomplete final eight-core repeat in the final
logs. FP16 tanh reuse1 has no completed launch in its trace. Each whole child,
including compilation and initialization, had a 480-second limit. VF fusion
was enabled and static allocation was default. No final numerical summary
was saved; block completion must not be mistaken for numerical qualification.

After these three failures, three already-started children were stopped and
six were not started. The exact routes reached, but did not finish, the
planned eight-core validation sequence.
Compile-time synchronization verification and one-tile warmed results do not
establish multi-tile or repeat-launch safety. These are observed Model
execution timeouts, not yet proof of a hardware deadlock or its compiler cause.
Fresh-cache, aligned-tail, forced-compilation and baseline controls are in
[tail_controls.py](tail_controls.py), with [eight recorded results](evidence/tail-controls/summary.json).

| Finite FP16 exact control (unless indicated) | Outcome within 120-second whole-child budget |
| --- | --- |
| reuse1 / VF=true, size2049 | First finite check passes; repeat1 incomplete |
| reuse1 / VF=true, aligned size2048 | First finite check passes; repeat1 incomplete |
| reuse1 / VF=true, forced compilation | First finite check passes; repeat1 incomplete |
| reuse0 / VF=false | Both finite checks complete |
| reuse1 / VF=false | Both finite checks complete |
| reuse0 / VF=true | Both finite checks complete |
| FP16 tanh, reuse1 / VF=false | Both finite checks complete |
| FP32 exact, reuse1 / VF=true | Both finite checks complete |

The shorter watchdog also includes compilation, initialization and first
execution; three children ran concurrently. Consequently these observations
do **not** prove a repeat-specific hang. The longer structural traces show
that repeated execution can progress. The VF/reuse/dtype combination is an
evidenced investigation target, while slow simulation versus synchronization
or runtime failure remains unresolved. Input mutation plus NaN output
initialization is not sufficient to cause the timeout: passing controls use
the same procedure. Any follow-up should measure serial, timestamped launches
with equal per-launch budgets before assigning a compiler cause.

## Case navigation

All rows refer to the [official case manifest](../../tasks/gelu/cases.yaml).
[Native dispatch/compile evidence](evidence/official-native.json) covers all 20.
[Evaluator-compatible compile evidence](evidence/official-x86.json) covers the
same cases without executing them on an NPU.
[Reduced numerical evidence](evidence/numerical-smoke/summary.json) samples each
case's value range; it does **not** establish full-shape hardware correctness.
The shared dense exact-mode blocker still prevents submission of this candidate.

| Case | Input shape | dtype | Mode | Implementation |
| --- | --- | --- | --- | --- |
| 1 | 1024 × 1024 | float16 | none | [exact](candidate/gelu.py#L33) |
| 2 | 2048 × 2048 | float32 | none | [exact](candidate/gelu.py#L33) |
| 3 | 4096 × 4096 | bfloat16 | none | [exact](candidate/gelu.py#L33) |
| 4 | 8192 × 8192 | float16 | tanh | [tanh](candidate/gelu.py#L27) |
| 5 | 8192 × 8192 | float32 | tanh | [tanh](candidate/gelu.py#L27) |
| 6 | 1023 × 1023 | bfloat16 | tanh | [tanh](candidate/gelu.py#L27) |
| 7 | 1009 × 1021 | float16 | none | [exact](candidate/gelu.py#L33) |
| 8 | 1537 × 769 | float32 | tanh | [tanh](candidate/gelu.py#L27) |
| 9 | 363 × 367 × 373 | bfloat16 | none | [exact](candidate/gelu.py#L33) |
| 10 | 2049 × 513 | float16 | tanh | [tanh](candidate/gelu.py#L27) |
| 11 | 3 × 7 × 13 × 4001 | float32 | none | [exact](candidate/gelu.py#L33) |
| 12 | 1000003 | bfloat16 | tanh | [tanh](candidate/gelu.py#L27) |
| 13 | 11 × 13 × 17 × 67 × 67 | float32 | none | [exact](candidate/gelu.py#L33) |
| 14 | 3 × 7 × 11 × 13 × 1009 | float16 | tanh | [tanh](candidate/gelu.py#L27) |
| 15 | 512 × 2049 | float32 | none | [exact](candidate/gelu.py#L33) |
| 16 | 255 × 8193 | bfloat16 | none | [exact](candidate/gelu.py#L33) |
| 17 | 4097 × 511 | float16 | tanh | [tanh](candidate/gelu.py#L27) |
| 18 | 2 × 511 × 2049 | float32 | none | [exact](candidate/gelu.py#L33) |
| 19 | 4 × 255 × 2049 | bfloat16 | tanh | [tanh](candidate/gelu.py#L27) |
| 20 | 2 × 3 × 17 × 1024 × 101 | float32 | none | [exact](candidate/gelu.py#L33) |

## Workers, skills and integration changes

The submitted-to-local-gates source is **handwritten, worker-reviewed**, not a
claim that raw worker output passed. OpenCode GLM-5.2 was attempted; its first
working tool-enabled session read the skills but did not produce a usable
candidate within the bounded run. `opencode/mimo-v2.5-free` produced a draft;
independent Qwen3.7-max review and main-agent inspection rejected it for wrong
tanh algebra, overlapping core partitions and a host dtype cast. That draft
was never promoted.

Qwen3.7-max also reviewed the main candidate and its evidence. Model findings
are not measurements: for example its assertion that Torch GeLU(-Inf) returns
zero is false here; executed golden and Model results both return NaN.
The main evidence takes precedence over reviewer prose.
[Worker summary](workers/run-20260907T080302Z/review-main-summary.json) includes
model/session IDs, source hashes and successful full-file skill-read events.
[Reconciled findings](workers/run-20260907T080302Z/findings.md) distinguish
measured facts from mistakes in raw model reviews.
Raw logs remain local; immutable worker drafts are retained as rejected evidence.

Existing skills and prompts now require the public AscTile style, teach the
JIT matrix and measured gates, and do not seed new work from legacy inline
sources. The shared [source contract](../../submission/source_contract.py)
checks aliases and helpers during generation, local qualification and package
preparation. It is a conservative AST contract, not a security sandbox or a
numerical proof.
All [36 focused contract tests](evidence/source-gate-tests.log) pass, including
aliased low-level imports and malicious runtime-helper fixtures. Updated skill
metadata validates; generation guidance now separates requested flags, actual
lowering, runtime qualification and performance evidence.

The upstream target's tanh coefficients were also checked independently: its
expression swaps the linear/cubic weighting and repeats the same formula in
its reference. At x=0.5 it gives about 0.279259 rather than standard tanh GeLU
0.345714. Target coding style was followed, not that mathematical defect.

## Reproduction and next action

Run from this directory, using an isolated simulator working directory:

```bash
bash build_native.sh
bash build_x86.sh
bash verify_x86.sh
source env.sh
python compile_official.py --output evidence/official-native.json
python jit_probe.py --output evidence/jit-cold/result.json
python jit_probe.py --base-first --output evidence/jit-base-first/result.json
python run_matrix.py --phase compile-matrix --jobs 4
python run_matrix.py --phase numerical-smoke --smoke --jobs 3
python run_matrix.py --phase numerical-dense-exact --dense-exact --jobs 3
python numeric_repro.py --output evidence/numeric-reproducer.json
python perf_control.py
python structural_probe.py
python audit_evidence.py --require-x86 --output evidence/audit.json
```

The compile and dense numerical matrix runners preserve same-source results;
use a new phase directory for a changed candidate. The expected dense result
at this pin is a failure, not a reason to weaken the gate. The audit separates
evidence integrity from `submission_eligible=false`.

Next: resolve the public API/backend numerical issue, rerun the dense checks,
then select valid JIT configurations, run the planned tile/unroll/core search,
and only then consider the single permitted GeLU-only CANNBench submission.
No earlier lower-level candidate is substituted automatically.

Historical NPU links: [i03](https://cannbench.com/workspace/jobs/job_a375a6e244ca)
and [i04](https://cannbench.com/workspace/jobs/job_7b4caccdc21f), with the
[original measured report](../gelu-perf-20260907/README.md). Their 20/20 results
and 0.474825× / 0.582646× speedups belong to lower-level implementations at
`0a631f70`, not to this high-level candidate.
