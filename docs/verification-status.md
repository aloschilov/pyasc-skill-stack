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
| Nightly generation | **Enabled; manual controller acceptance passed** under the user-approved substitute for cron. Real failure/recovery issue delivery is confirmed for all three lanes. Actual cron delivery remains unverified. [Evidence and limits](../tests/generation/README.md#manual-controller-acceptance-2026-11-06). |
| Manual generation CI | **Passed** on `ab28634`: 3/3 fresh GLM-5.3 candidates, nine Model launches, 21 numerical checks, three rejecting controls, 253.91 s executor. [Current run and archive](../tests/generation/README.md#manual-controller-acceptance-2026-11-06). No NPU or scheduled claim. |
| Generation regression comparison | **Passed** for current manual controller against the explicitly selected durable baseline: compatible inputs, 3/3 candidates, no lost PASS or coverage gaps. Hosted observer and recovery verified. [Result](../tests/generation/README.md#manual-controller-acceptance-2026-11-06); actual scheduled coverage remains unverified. |
| Paired context experiment | **Passed** for complete manual measurement: 24 auditable trials; P6 4/6, P2/P3/P4 each 0/6, all failures retained. Conditional intervals include zero; no general skills benefit. Hosted recovery confirmed. [Result and archive](../tests/generation/README.md#manual-controller-acceptance-2026-11-06). |
| Multiple models | **Passed** for current 24-trial manual comparison with two actual response identities on one gateway; 7 qualified candidates and all failures retained. Hosted recovery confirmed; no general benefit, model ranking or scheduled claim. [Results and archive](../tests/generation/README.md#manual-controller-acceptance-2026-11-06). |
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
