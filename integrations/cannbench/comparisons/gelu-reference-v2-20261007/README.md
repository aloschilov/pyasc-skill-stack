# GeLU reference-DAG reproduction on current pyasc v2

Experiment started 2026-10-07. Isolated from the concurrently modified skill repository.

## Published artifact and offline verification

The **complete successful submission** is [repair/diagnostic.zip](repair/diagnostic.zip),
including the evaluator build entrypoint, metadata bridge sources, kernel/launcher,
and the self-contained CPython 3.12/x86_64 runtime wheel. Its bytes are unchanged
from the upload. The [first archive](diagnostic.zip) is also preserved for the
manifest-only failure comparison; use the replacement, not the failed original.
The readable [kernel](kernel.py) and [launcher](bundle/cann_bench/gelu.py) match both
archives. The wheel contains upstream license notices; no CANN SDK is bundled.

From this report directory, run these CPU-only, network-free checks:

```sh
python3 verify_publication.py
python3 test_contract.py
```

The first command verifies both complete archive hashes, all member hashes,
nested wheel RECORDs, source identity, the one-member repair, 20-case timing
accounting and report links. It does not rerun CaModel or spend credits.
The second checks host geometry and the source contract, not device accuracy.

The other build/model/submission scripts are **historical experiment recipes**,
not a portable one-command runner: they require the recorded CANN/LLVM/Docker
installations and isolated checkouts. Absolute paths in evidence are provenance.
Only readable sources are duplicated under `bundle/`; extract the full archive
into a fresh directory to obtain the wheel and complete build payload. Do not run
the submission controller merely to inspect this report: its account/controller
dependencies and private raw responses are intentionally not published.

This publication includes selected local results and IR, both offline qualification
records, and credential-checked stage logs/states. It excludes raw job/response
objects, private scheme metadata, signed download URLs, build caches and local
installations. CANNBench job links may still require account access. The PR was
prepared in a separate worktree; it neither changes nor pauses the active skill
modification loop.

## Status and interpretation

**Hardware result: 20/20 official cases passed, zero anti-cheat failures. Conventional all-case GM speedup is 0.509582×; none of the 20 cases beats the stored CANNBench reference.** Successful diagnostic: [job_ce3bf7738fb3](https://cannbench.com/workspace/jobs/job_ce3bf7738fb3). Score: 63.503927. CANNBench's field named `geometric_mean_speedup` reports 0.538669×, which numerically equals the arithmetic mean of its 20 returned speedups—not their conventional geometric mean. This report uses `exp(mean(log(reference_us / candidate_us)))` without filtering cases.

The high-level [kernel](kernel.py) reproduces the inspected public ops-nn mathematical routes as closely as the current AscTile API permits. It is **not an instruction-identical or binary-identical reproduction of CANNBench's reference**. No compiler passes were changed.

Native compilation passed for all six dtype/mode routes at each of three tile sizes (18 configurations). At the selected tile, compiled CaModel checks passed five routes completely. FP32 exact passed the sampled official ranges but failed an additional dense negative-tail stress segment. That limitation is retained, not hidden with a relaxed tolerance or input-dependent workaround.

Evaluator-compatible x86 build and offline qualification passed: six stock lowerings, 80 mocked host dispatches (20 cases × caps 1/8/64/72), real SDK metadata-bridge loading and loaded-library identity verification. Both original and metadata-augmented wheel RECORD hashes verified. This is not local x86 device execution.

First private diagnostic: [job_b822d9190143](https://cannbench.com/workspace/jobs/job_b822d9190143), tag `gelu-reference-v2-9069108-20261007`, ended `compile_failed` before any kernel compilation or execution. Only the outer `build.sh` failed the in-build checksum; every other payload hash passed. All 20 case records are cascading build failures, **not 20 measured accuracy failures**. No hardware timing or valid GM exists for this attempt. [Summary](hardware-summary.json), [complete stage log](remote/logs.json).

A manifest-only replacement diagnostic [job_ce3bf7738fb3](https://cannbench.com/workspace/jobs/job_ce3bf7738fb3), tag `gelu-reference-v2-9069108-20261007-buildfix`, **succeeded**. It passes the official workload, not every possible numerical stress distribution, and is not a performance win. The initial attempt is preserved. States: [original](remote/state.json), [replacement](repair/remote/state.json). Offline evidence: [original](x86-qualification/x86-results.json), [replacement](repair/x86-qualification/x86-results.json).

Submitted archive SHA256: `a7819eede1ac0a1867364db81c97fe9283193219fa871f7d5a112c8ac8b62cd0`. Kernel SHA256: `5e3a0b09773e18965c3a57776b09dc40171e08134bec7b8c170a05c6f1b226cc`. [Package provenance](package.json), [archive](diagnostic.zip).

Replacement archive SHA256: `ef5269b46a56437463f5dcef53fd408dcf173c3a502e0306726a797a6f840ceb`. [Replacement provenance](repair/package.json), [replacement archive](repair/diagnostic.zip). The sole changed archive member is `source.sha256.txt`; authored kernel, runtime, wheel, helpers, inner build payload and even the authored outer launcher are byte-identical.

### Build-entry integration repair

The first manifest reintroduced a previously diagnosed integration mistake: hashing the outer staging entrypoint during its own execution. The public evaluator path executes `bash build.sh`; the outer runner's exact transformation is not available, so its precise mutation is **not** established. The first job proves only that the executed outer file differs while all inner payload hashes match.

The replacement excludes only that outer entrypoint from the in-build manifest. The complete archive/pre-upload manifest still binds its authored bytes. All compute/runtime/helper files, the wheel and `metadata/build_payload.sh` remain strictly checked; no accuracy or anti-cheat checks were relaxed. This is our packaging/integration correction, not a pyasc compiler issue.

[Seven local entrypoint/integrity tests](repair/entrypoint-tests.json) show that the old manifest rejects a synthetic launcher comment, the replacement permits that synthetic outer edit, and independent mutations of kernel, resource helper, inner payload or wheel fail. The comment is not claimed to reproduce the unknown real runner transformation. The corrected package then passed a fresh offline build/install/load, six lowerings and 80 host-dispatch checks. Native numerical evidence is reused only because every compute byte is unchanged. At most one replacement POST is allowed; this is a distinct archive after a confirmed terminal failure, not an ambiguous-upload retry.

## Frozen sources

- pyasc: [compiler-team/pyasc v2, 9069108e323746187c48d78fec4929c4afa2efc4](https://gitcode.com/compiler-team/pyasc/tree/9069108e323746187c48d78fec4929c4afa2efc4). Latest v2 resolved at experiment start; not a moving branch reference during testing. Native checkout: `/home/aloschilov/workspace/pyasc-gelu-reference-20261007`.
- Experimental API enabled with `PYASC_SETUP_EXPERIMENTAL=1`; import is now `from asc.experimental import asctile`.
- [Public ops-nn DAG](https://gitcode.com/cann/ops-nn/blob/f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307/activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h), commit `f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307`.
- [opbase Erf wrapper](https://gitcode.com/cann/opbase/blob/f59a9cb9f414c4b80cf4eccc58006798c1af15de/pkg_inc/op_common/atvoss/util/vec.h), commit `f59a9cb9f414c4b80cf4eccc58006798c1af15de`.
- [Full source dispatch trace](inputs/dispatch-trace/README.md), [per-case trace](inputs/dispatch-trace/cases.csv), [frozen cases](inputs/cases.yaml), [golden](inputs/golden.py), [official checker](inputs/precision.py).

The live CANNBench task's golden source matched the frozen source byte-for-byte. All 20 IDs, shapes, dtypes and modes matched. The live JSON serializes nonfinite range endpoints as null; the frozen YAML retains inf/nan. This is not evidence of changed shapes or a replacement nonfinite test.

Local compilation/execution uses CANN 9.0. Historical runners reported CANN 9.1 and Torch-NPU 2.10 without exact library source hashes. Current pyasc documentation recommends a newer CANN release. These environment differences limit numerical/performance extrapolation.

## Algorithm and API limits

All routes convert the tile to FP32 for computation. Tanh follows the reference operation order: square, cube, Axpy-equivalent expression, scaled exponent, add one, divide, cast back. The constants are rounded exactly as the inspected C++ float initializers: `22.363859176635742` and `-0.07135481387376785`.

Low-precision exact uses public `asctile.erf`, matching the default Erf selection in the inspected reference. FP32 exact uses the permitted small inline-Erf escape hatch to select `SUBSECTION_POLYNOMIAL_APPROXIMATION`, then the same high-level postprocessing `(erf_value + 1) * (x * 0.5)`. No hand-authored register kernel or replacement polynomial was added.

Two limitations were reported through the authorized Chrome session:

1. [Issue #9: explicit fused multiply-add/Axpy semantics](https://gitcode.com/compiler-team/pyasc/issues/9). AscTile's generated AscendC contains separate Mul and Add calls for the Axpy-equivalent expression. This does not prove the final backend cannot contract them: the installed disassembler cannot decode the instructions. There is no public API to explicitly request the reference's fused operation/rounding behavior. See [posted text](issue-axpy.md) and [publication evidence](issue-9.jpg).
2. [Issue #8: Erf algorithm selector, updated for current v2](https://gitcode.com/compiler-team/pyasc/issues/8). Public `erf(input)` still does not expose `ErfConfig`; the inline workaround compiles and executes. The update explicitly describes the remaining FP32 negative-tail failure rather than claiming that the selector fixes all accuracy. See [posted comment](issue-erf-update.md) and [publication evidence](issue-8-update.jpg).

After the hardware job completed, a [second clarification](issue-erf-hardware-followup.md) was posted to #8: all official hardware cases pass, while the distinct local stress failure remains. [Visible publication evidence](issue-8-hardware-followup.jpg). Neither issue claims a proven pass defect or a measured FMA-related speedup.

The source trace identifies SIMD for all 20 standard eager Ascend950 reference routes, not SIMT. Matching the reference's mathematics does not reproduce its scheduler, ping/pong buffers, tiling or instruction schedule. Our host launch uses balanced contiguous tile ranges and resource queries, not benchmark IDs.

## Geometry, JIT and UB

[Launcher](bundle/cann_bench/gelu.py), [compile inventory](inventory.json), [per-case geometry](case-geometries.json).

- Selected `tile_shape=[15872]`, unroll 1. This is the largest of the three compiled candidates, **not a proven throughput optimum or exhaustive UB maximum**.
- Each route's current generated allocation is 190464 bytes = 186 KiB. All compute in FP32; equal measured UB across these routes is not an assumption that input dtype never matters.
- Inventory: tile 2048 → 24576 bytes; tile 8192 → 98304 bytes; tile 15872 → 190464 bytes, for each of six routes. Compiler capacity check: 253952 bytes = 248 KiB.
- Authored flags: `reuse_alloc=2`, `vf_fusion=True`. Effective stock compiler defaults include `insert_sync=True`, `auto_sync=True`, `opt_level=3`, `vf_vec_len=256`, `static_alloc=None`. Generated allocation is static on this target. Diagnostics use `verify_sync=True` and a forced preflight compile; timings are not taken with device debugging enabled.
- Runtime launch count is `min(queried launch cap, ceil(N / tile))`. Reported AIV, device/thread/stream limits and launched blocks are distinct. Unknown limits fall back conservatively to one block. The metadata helper validates SIMD UB against the installed evaluator SDK; raw ACL's different UB quantity is not substituted for that capacity.
- The frozen helper supports active NPU device 0 only and fails explicitly for another device; no broader multi-device portability claim is made.

All rows below belong to [successful job_ce3bf7738fb3](https://cannbench.com/workspace/jobs/job_ce3bf7738fb3). They use unroll 1, `reuse_alloc=2`, `vf_fusion=True`, the remaining JIT configuration above and locally compiled UB 186 KiB. Kernel links identify the shared implementation and route, not 20 different kernels. Timings are the API's `elapsed_us` and stored `baseline_perf_us`, not a new controlled reference A/B.

| Case | Shape | Dtype | Mode | Tile shape | Reference µs | Candidate µs | Speedup | Accuracy | Implementation |
|---|---|---|---|---|---:|---:|---:|---|---|
| 1 | 1024 × 1024 | FP16 | none | [15872] | 4.490 | 11.66 | 0.3851× | pass | [public Erf](kernel.py) |
| 2 | 2048 × 2048 | FP32 | none | [15872] | 15.370 | 37.96 | 0.4049× | pass | [configured Erf](kernel.py) |
| 3 | 4096 × 4096 | BF16 | none | [15872] | 30.140 | 93.08 | 0.3238× | pass | [public Erf](kernel.py) |
| 4 | 8192 × 8192 | FP16 | tanh | [15872] | 172.930 | 199.79 | 0.8656× | pass | [tanh expression](kernel.py) |
| 5 | 8192 × 8192 | FP32 | tanh | [15872] | 387.265 | 401.78 | 0.9639× | pass | [tanh expression](kernel.py) |
| 6 | 1023 × 1023 | BF16 | tanh | [15872] | 4.460 | 6.86 | 0.6501× | pass | [tanh expression](kernel.py) |
| 7 | 1009 × 1021 | FP16 | none | [15872] | 4.450 | 11.61 | 0.3833× | pass | [public Erf](kernel.py) |
| 8 | 1537 × 769 | FP32 | tanh | [15872] | 6.110 | 8.00 | 0.7638× | pass | [tanh expression](kernel.py) |
| 9 | 363 × 367 × 373 | BF16 | none | [15872] | 117.530 | 269.05 | 0.4368× | pass | [public Erf](kernel.py) |
| 10 | 2049 × 513 | FP16 | tanh | [15872] | 4.560 | 6.95 | 0.6561× | pass | [tanh expression](kernel.py) |
| 11 | 3 × 7 × 13 × 4001 | FP32 | none | [15872] | 6.050 | 16.63 | 0.3638× | pass | [configured Erf](kernel.py) |
| 12 | 1000003 | BF16 | tanh | [15872] | 4.380 | 6.69 | 0.6547× | pass | [tanh expression](kernel.py) |
| 13 | 11 × 13 × 17 × 67 × 67 | FP32 | none | [15872] | 37.455 | 89.83 | 0.4170× | pass | [configured Erf](kernel.py) |
| 14 | 3 × 7 × 11 × 13 × 1009 | FP16 | tanh | [15872] | 7.780 | 11.93 | 0.6521× | pass | [tanh expression](kernel.py) |
| 15 | 512 × 2049 | FP32 | none | [15872] | 5.980 | 16.13 | 0.3707× | pass | [configured Erf](kernel.py) |
| 16 | 255 × 8193 | BF16 | none | [15872] | 6.230 | 16.92 | 0.3682× | pass | [public Erf](kernel.py) |
| 17 | 4097 × 511 | FP16 | tanh | [15872] | 6.310 | 9.33 | 0.6763× | pass | [tanh expression](kernel.py) |
| 18 | 2 × 511 × 2049 | FP32 | none | [15872] | 8.860 | 23.82 | 0.3720× | pass | [configured Erf](kernel.py) |
| 19 | 4 × 255 × 2049 | BF16 | tanh | [15872] | 6.260 | 9.83 | 0.6368× | pass | [tanh expression](kernel.py) |
| 20 | 2 × 3 × 17 × 1024 × 101 | FP32 | none | [15872] | 36.195 | 84.49 | 0.4284× | pass | [configured Erf](kernel.py) |

[Machine-readable cases](repair/hardware-cases.csv), [independent summary](repair/hardware-summary.json), [GET-only collector](collect_results.py).

Actual per-case AIV reported/limited/launched counts and remote post-lowering artifacts were **not returned by the successful job's stage-log API**. The launcher emits them, but this API snapshot contains only stage events. Do not replace those missing observations with 64/72 or infer physical core count from the SoC suffix. The cap64/cap72 fields in the local geometry inventory are counterfactual host calculations, not measured launches. Likewise, local UB/compiler-option evidence is not a captured remote IR dump.

Runner: `runner-950pr-multi #3`, `Ascend950PR_957c`, CANN 9.1.0, Torch-NPU 2.10.0.post4, PyTorch 2.10.0+cpu, CPython 3.12.13. The runner reports evaluator revision `e8609f6b5891361fb6f0afb141eb8909c1b4a59b` and benchsite revision `605a23e3e3b857353689282cc7de1b0bf66e8c99`. These are not source hashes for the installed CANN operator library. The first build-failed attempt was on runner #5; neither attempt establishes cross-runner performance reproducibility.

## Local validation: what actually ran

[Harness](probe.py), [serial model controller](run_local.py), [host contract tests](test_contract.py). The serial controller uses the existing shared simulator lock; it does not start another agent's queue.

Host tests validate rounded constants, exact nonoverlapping tile coverage at 1/8/32/64/72 logical blocks, tails, and the single allowed authored inline operation. These are not numerical execution tests.

Native stock-pipeline binary compilation passed 18 configurations. Since shape/size is a runtime argument, the 20 selected geometries map to six compiled dtype/mode specializations. This does **not** mean all full official tensors were simulated.

Compiled CaModel execution used C310/Ascend950PR_9599, tile 15872, unroll 1 and 8 logical blocks for the sampled-route qualification. Samples combine seeded random values, 513-point grids over each route's official range, NaN/infinity probes where relevant, and a separate 2049-point `[-8,-2]` tail. Samples do not duplicate the exact evaluator random generator or full case sizes. Each launch was repeated and followed by an output guard check.

| Route | Sampled official ranges | Additional negative-tail stress | Repeats/guards | Evidence |
|---|---|---|---|---|
| FP16 none | pass | pass | pass | [result](evidence/qualification-float16-none/result.json) |
| FP16 tanh | pass | pass | pass | [result](evidence/qualification-float16-tanh/result.json) |
| BF16 none | pass | pass | pass | [result](evidence/qualification-bfloat16-none/result.json) |
| BF16 tanh | pass | pass | pass | [result](evidence/qualification-bfloat16-tanh/result.json) |
| FP32 none | pass | **fail** | pass | [corrected-harness result](evidence/verified-float32-none/result.json) |
| FP32 tanh | pass | pass | pass | [corrected-harness result](evidence/verified-float32-tanh/result.json) |

FP32 exact stress: maximum absolute error about `1.2037e-7`; cancellation-error count 6, versus CPU reference count 2. The official relative-to-CPU allowance permits 3 here. Small absolute error alone is insufficient to pass this checker. The same expression evaluated with CPU float Erf passes the checker; therefore the evidence does not establish that every Erf-based GeLU must fail, or that an AscTile pass is responsible. SDK algorithm and composition rounding remain relevant.

The first harness incorrectly treated repeated NaNs as unequal; the definitive FP32 evidence above uses NaN-aware equality. An early fragmented FP32 run timed out and early BF16 runs hit a harness runtime-import error; their logs remain in the original isolated experiment but are omitted from this curated publication. They are not presented as kernel failures. The fixed combined-sample reruns completed.

Separate low-precision tanh tests execute 31745 elements (two full tiles plus one tail element) on one block to exercise repeated tile reuse, rather than merely idle blocks. Both [FP16](evidence/multitile-float16-tanh/result.json) and [BF16](evidence/multitile-bfloat16-tanh/result.json) passed. A separate [nine-tile/eight-block FP16 exact test](evidence/multicore-float16-none/result.json), 126977 elements, also passed accuracy, repeated execution and output guards. Model tick counts are simulator values for these specific samples, not hardware latency predictions; no GM speedup can be computed from them.

## Build and submission reproducibility

- Native: CPython 3.11/aarch64; existing ARM LLVM; stock `setup.py bdist_wheel`, experimental enabled, isolated build/install.
- Evaluator: CPython 3.12/x86_64 under QEMU, separate clean source clone at the same pin, LLVM 20 artifact SHA256 `b596c493d8f16e1c40026d16024fdcb3225dc09c5c7d8493833dff774677f67b`. Build script: [build_x86.sh](build_x86.sh).
- Self-contained wheel includes current `asc`, experimental AscTile, compiled extension, pybind11 headers and license notices; no dependency on an unpublished PyPI release. [Packaging](package.py), [offline qualification](qualify_x86.sh), [x86 lowering/host-dispatch tests](qualify_x86.py).
- The evaluator compiles only the small host metadata bridge against its own CANN SDK. No CANN SDK binaries are shipped. Compile/lowering tests are distinct from unavailable local x86 NPU execution.
- [Submission controller](submit.py) allows one private GeLU-only diagnostic, all 20 official cases, after offline package checks, fresh credits/active-job checks and the shared submission lock. It durably records intent before POST; ambiguous outcomes permit only GET reconciliation, never automatic duplicate POST.
- A first preflight stopped before intent/POST because the historical integration expected an independent runner pool. The live API now identifies `system-shared` as a platform-managed pool (`is_system=true`, `is_independent_environment=false`), with four online 950PR runners at the check. The published MCP 0.8 client was inspected, and the obsolete consent field was omitted. This was not a failed/retried upload and consumed no credit; the candidate archive was unchanged.
- No timers were created during the experiment. The experiment did not modify the active skill checkout or compiler source. This subsequent publication is confined to an isolated branch/worktree and a pull request.

## Conclusions so far

The public reference's algorithm is representable with a compact high-level AscTile kernel plus the explicitly allowed Erf configuration escape hatch. Exact instruction/rounding equivalence is not established, chiefly because explicit Axpy semantics are unavailable and the installed CANN differs from the historical runner.

The newer compiler successfully compiles and runs the sampled six routes with reuse mode 2 and VF fusion, including low-precision tanh tile reuse. The replacement diagnostic also passes **20/20 full official hardware cases**. It does not resolve or invalidate the separate dense FP32 exact stress failure on the local CANN 9.0 simulator. Official-case success and broader numerical robustness are different claims.

Performance remains below reference: conventional GM is **0.509582×** overall, **0.385435×** across the 11 exact-mode cases, and **0.716843×** across the nine tanh cases. The closest result is FP32 tanh `[8192,8192]` at **0.963873×** (401.78 versus 387.265 µs). No case exceeds 1×. Matching the public algorithm alone is therefore insufficient for the desired overall performance.

The next useful controlled experiment should target the exact-mode path: match geometry and SDK while measuring transfers around Erf, cast stages, live UB buffers and overlap against the reference scheduler. An explicit Axpy API remains relevant to strict tanh instruction reproduction, but this run does not establish it as the main performance bottleneck. Do not attribute the exact-route gap to a pass without matched traces and a minimal before/after reproducer.

Two credits were used (one packaging failure, one successful full diagnostic); eight remained at the final query. No further jobs were submitted and no timers were created. Raw private API response directories have restricted permissions and may contain private scheme metadata; do not publish them wholesale. The README, sanitized summary/CSV and authored kernel do not require those credentials.

Any comparison with older jobs is observational: both the compiler revision and expression changed, and hardware/SDK/load may differ. One diagnostic cannot isolate an algorithm effect from a compiler-pass effect.
