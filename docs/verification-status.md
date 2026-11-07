# Verification status

Use `passed | failed | blocked | not-run` per acceptance scope. Missing
prerequisites or resource blocks do not count as successful coverage.

| Scope | Evidence and limit |
|---|---|
| Offline checks and shipped simulator fixtures | Required on each PR head by `pr-gate` and `examples-gate`; inspect that head's Actions results. These do not invoke a model or NPU. |
| Optional source-built runtime | **Passed** for one pinned native CPython 3.12/x86_64 build: two clean imports, exact submission-runtime bytes, fixed add[128] compile/Model/numerics and suppressed-launch rejection. [Evidence and narrow scope](source-runtime.md#recorded-qualification); default runtime unchanged, no NPU qualification. |
| Fresh multi-operator generation in simulator | **Passed** for three GLM-5.3 candidates (add/reduce/broadcast), three deterministic patterns each, unchanged bytes and suppressed-launch controls. [Manual pilot, limits and archive checksum](../tests/generation/README.md#manual-pilot-2026-11-06). No NPU, performance or nightly claim. |
| Generated official CANNBench cases in CaModel | Each accepted case passed the official comparison at seeds 19 and 23 and rejected both suppressed launches. The first cohort passed 1/3 on the first attempt and 3/3 in five attempts. [Case coverage by catalog revision and archive checksums](#cannbench-camodel-pilot) |
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

## CANNBench CaModel pilot

The November 7 pilot covered 3 of 1,060 cases from 3 of 53 operators at CANNBench
revision `597a915e`. The same accepted bytes also passed the evaluator at
`08d519c5`. Each archive includes its own member checksums.

| Evidence | Archive SHA256 |
|---|---|
| [Three-case pilot](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-cannbench-camodel-pilot-20261107/cannbench-camodel-three-cases-20261107.tar.gz) | `3f72113b64b49759d55f37fc07c69f8b24315bf9284cdab2e8c30d0edc466efb` |
| [OpenCode walkthrough with installed skills](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-cannbench-camodel-pilot-20261107/cannbench-readme-walkthrough-20261107.tar.gz) | `608869092674c2bed822ea84b3c2105ae09d1ce33c83a4deba3118369eee1740` |
| [OpenCode walkthrough in a clean stack checkout](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-cannbench-camodel-pilot-20261107/cannbench-checkout-walkthrough-20261107.tar.gz) | `7deeb72aa5140145c3c77e42886f8fcfaf60934bc222110e9f00bbe36ba8073d` |
| [Two-case skill comparison](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-cannbench-camodel-pilot-20261107/cannbench-skill-scope-comparison-20261107.tar.gz) | `af02bfb6dde8916a86abc18e77944855df9f540e12b4896c88281cf60bf55286` |

At `08d519c5`, a second cohort tried Maximum 01, StridedSlice 01 and Gather
01. Maximum passed on its first attempt. StridedSlice passed from a written
file on its fourth attempt. Both passed at seeds 19 and 23 and rejected both
suppressed launches. Gather reached the 600-second limit on its first
seed. With the original three cases, coverage within the recorded limits is
5/1,060 cases from 5/53 operators. The eight new generation attempts used
106,837 completion tokens. The [cohort archive](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-cannbench-breadth-49/cannbench-camodel-breadth-49.tar.gz)
has SHA256 `b675dc41c9f1f2ef0327e3a05981b19f177bfcfb4c18356138cce0b435807ef6`.

Gather 01 subsequently passed at full size with the same generated file and a
5,400-second phase limit. CaModel made two kernel calls per seed. Both official
comparisons passed, and both rejected the output when the calls were suppressed.
The real phase took 2,607.954 s and the suppressed phase took 4.775 s. This
verification added zero model requests. At `08d519c5` the two cohorts now
cover **6/1,060 cases across 6/53 operators**, one case per operator.
The [Gather archive](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-cannbench-breadth-49/cannbench-gather-01-long-verification.tar.gz)
has SHA256 `3fb794ab548fa13f4d3081ab4ff3604b1e63977879d61fe9d74952099c20f1df`.
