# Verification status

Use `passed | failed | blocked | not-run` per acceptance scope. Missing
prerequisites or resource blocks do not count as successful coverage.

| Scope | Evidence and limit |
|---|---|
| Offline checks and shipped simulator fixtures | Required on each PR head by `pr-gate` and `examples-gate`; inspect that head's Actions results. These do not invoke a model or NPU. |
| Optional source-built runtime | **Passed** for one pinned native CPython 3.12/x86_64 build: two clean imports, exact submission-runtime bytes, fixed add[128] compile/Model/numerics and suppressed-launch rejection. [Evidence and narrow scope](source-runtime.md#recorded-qualification); default runtime unchanged, no NPU qualification. |
| Fresh multi-operator generation in simulator | **Passed** for three GLM-5.3 candidates (add/reduce/broadcast), three deterministic patterns each, unchanged bytes and suppressed-launch controls. [Manual pilot, limits and archive checksum](../tests/generation/README.md#manual-pilot-2026-11-06). No NPU, performance or nightly claim. |
| Fresh generation and hardware measurement | **Passed** for one GLM-5.3 SwiGlu candidate: 20/20 official cases, zero reported anti-cheat failures, device timings. [Durable result, revisions and checksum](hardware-lane.md). |
| Performance improvement | **Not established**: the generated candidate's baseline/candidate geometric timing ratio is 0.017450287921005128x (slower). Service score is a separate metric. |
| Other profile and recipe combinations | Offline regression scope only. The shared CANNBench recipe driver has no new hardware qualification; the manual simulator pilot above qualifies only its recorded executor/configuration. Historical results retain their original configurations. |
| Separate reviewer automation | **Passed** for one fresh add candidate: two GLM-5.3 sessions, accepted static review, three Model launches, nine output/input checks and a rejecting suppressed-launch control. [Run, scope and durable archive](../tests/generation/review.md#manual-qualification-2026-11-06). Same model; no live repair was needed. |
| Optional bounded tuner | Offline transition/selection tests only; no live tuning improvement or new device measurement. [Caller and measurement boundary](../tests/generation/review.md#optional-bounded-tuner). |
| Full upstream target suite | [Exact census and bounded numerical shards](../tests/target-suite/README.md): 1,155 cases in 24 modules; six selected cases qualified with six rejecting controls, 1,149 remain unevaluated. No complete numerical pass claimed. `merge-gate` remains compile-only; resource blocks fail coverage. |
| Nightly generation | **Not-run**; opt-in controller/health observer implemented but inactive, with no cron triggers. Policy is disabled and has no alert destination. Actual scheduled generation and live alert/recovery delivery are unqualified. [Controls and activation criteria](../tests/generation/README.md#inactive-nightly-controller-and-health-observer). |
| Manual generation CI | **Passed** on integrated `e2f4c31`: 3/3 fresh GLM-5.3 candidates under `asctile-*`, nine Model launches, 21 numerical checks and three rejecting controls. [Reviewed durable baseline, usage and timing](../tests/generation/README.md#inactive-nightly-controller-and-health-observer). No new baseline comparison, NPU or scheduled claim. |
| Generation regression comparison | **Passed** for one historical independent same-skills pair: explicit durable baseline, compatible inputs, 3/3 fresh candidates and no lost PASS or coverage gaps. [Historical baseline, run and checksums](../tests/generation/README.md#selected-baseline-and-compatible-comparison). This validates the comparison path; scheduled alert delivery remains unqualified. |
| Paired context experiment | **Passed** for complete manual measurement: 24 auditable GLM-5.3 trials, six pairs per contrast; P6 qualified 5/6, P2/P3/P4 each 0/6. The P6-P3 conditional interval includes zero, so no general skills benefit is established. [Result, limits and durable archive](../tests/generation/paired.md#manual-qualification-2026-11-06). Local GET-only observer verified; hosted/scheduled observation remains unqualified. |
| Multiple models | **Passed** for one complete manual comparison: 24 auditable DeepSeek/GLM trials; P3 0/6 for each model, P6 6/6 and 5/6 respectively. All 13 failures remain in the denominator. All four conditional intervals include zero; no general benefit or model ranking. [Run, scope and durable archive](../tests/generation/models.md#manual-qualification-2026-11-06). Periodic coverage is disabled. |
| Public evidence dashboard | **Passed** for manual publication of five retained records, with public bytes matching the Pages artifact. [Publication identity, archive and limitations](../dashboard/README.md#public-publication-qualification-2026-11-06). This is a view of existing qualifications, not new numerical coverage or continuous monitoring. |

The checkout and installed-skill workflows also have earlier simulator-only
qualifications. Their source hashes, results, archive locations and checksums
remain in the [immutable historical index](https://github.com/aloschilov/pyasc-skill-stack/blob/654a05556107c538dca11fc71b02fd1378543ded/docs/verification-status.md).
Those records do not qualify later revisions or expand hardware coverage.
Full run artifacts remain outside checkout; see the [storage and attribution
contract](verification-contract.md).

The optional [simulator timing comparison](simulator-comparison.md) has one
qualified fixed Add experiment: canonical Ascend C versus PyASC with fusion
off/on, matching runtime bytes, per-launch ticks, numerics and negative controls.
Its descriptive ratios are not NPU performance claims; fusion emitted an
identical binary in this sample.
