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
| Nightly generation | **Actual scheduled single-generation run passed**: GLM-5.3 add/reduce/broadcast, 3/3 with a compatible baseline comparison. Recovery notification remains **unknown**; the other lanes and punctual delivery are not established by this run. [Evidence and limits](../tests/generation/README.md#scheduled-single-generation-recovery-2026-11-27). |
| Manual generation CI | **Passed** on `ab28634`: 3/3 fresh GLM-5.3 candidates, nine Model launches, 21 numerical checks, three rejecting controls, 253.91 s executor. [Current run and archive](../tests/generation/README.md#manual-controller-acceptance-2026-11-06). No NPU or scheduled claim. |
| Generation regression comparison | **Passed** for scheduled run `45920011665` against the selected immutable baseline `45865783111`: compatible inputs, 3/3 candidates, no lost PASS or coverage gaps. Scheduled recovery notification remains unconfirmed. [Result](../tests/generation/README.md#scheduled-single-generation-recovery-2026-11-27). |
| Paired context experiment | **Passed** for complete manual measurement: 24 auditable trials; P6 4/6, P2/P3/P4 each 0/6, all failures retained. Conditional intervals include zero; no general skills benefit. Hosted recovery confirmed. [Result and archive](../tests/generation/README.md#manual-controller-acceptance-2026-11-06). |
| Targeted SwiGLU fallback guidance | **Correctness passed; timing mixed across cohorts.** The original B candidates used fewer CaModel ticks on six shapes. A separate four-session replication improved BF16 but regressed FP16 on all three shapes each. No general skill-benefit or hardware claim. [Protocols, outcomes and checksums](#fallback-guidance-comparison-2026-11-27). |
| Exp empty-slot guidance | Original B **passed** and A **failed** lowering; the A → B replication **passed both arms**, each with eight real checks and eight rejecting controls. Predeclared tie/nonreplication; no general benefit claim. [All four outcomes and checksums](#exp-empty-slot-comparison-2026-11-27). |
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

## Fallback guidance comparison — 2026-11-27

Four fresh GLM-5.3-low/OpenCode 1.19.0 sessions compared `3f489799` with
`fca9806a`, differing only in eight lines directing fallback tuning to existing
measured controls. All final candidate bytes and outcomes are retained: 38
completed requests and 82,198 output tokens, with no retries in this study.

The original checker qualified 3/4 candidates and rejected baseline FP16 before
execution for `torch.is_tensor`. A separate common checker allowing that pure
type predicate qualified all four unchanged sources: 24 real numerical calls
and 24 rejecting suppressed controls across three shapes per dtype and two
seeds. This does not replace the original result.

Separate counterbalanced timing retained 192 real calls, 144 warmed samples
and 12 rejecting controls. B used fewer Model ticks on all six fixed shapes
(A/B ratios 1.046669–1.542978); both orders agreed. Layouts, tiles and some grids
differed. Both B sessions read the paragraph but not the linked guide and used
VF off; baseline FP16 read the guide and used VF on. These candidate outcomes
do not isolate the paragraph or VF, establish general skill improvement, or
predict NPU latency and CANNBench scores.

A separate four-session replication reversed the generation order within each
dtype, retaining the same arm revisions, prompts, profile and common checker.
All four unchanged sources passed 24 real numerical checks and 24 rejecting
controls. Its 30 completed requests account for 67,192 completion tokens; every
request used the profile-native output limit of 32,768. No source was repaired,
replaced or selected from an earlier write.

The replication retained another 192 real timing calls, 144 warmed samples and
12 rejecting controls. Both counterbalanced orders agreed per shape:

| Dtype | R / H | Replication A/B Model ticks | B compared with A |
| --- | --- | ---: | --- |
| FP16 | 7 / 47 | 0.883959 | 13.13% more ticks |
| FP16 | 35 / 127 | 0.612197 | 63.35% more ticks |
| FP16 | 71 / 511 | 0.529667 | 88.80% more ticks |
| BF16 | 9 / 79 | 1.137578 | 12.09% fewer ticks |
| BF16 | 37 / 383 | 1.667380 | 40.03% fewer ticks |
| BF16 | 73 / 1025 | 1.511688 | 33.85% fewer ticks |

All replication kernels used VF off and `reuse_alloc=2`. FP16 B chose larger
row tiles and fewer blocks; BF16 B chose flat chunks instead of A's row tiles.
The original FP16 advantage did not replicate. These results do not isolate
the guidance paragraph, tile height, grid or VF as the cause, and are not pooled
into an overall improvement claim. Qualification remains limited to the six
finite, contiguous synthetic shapes and two seeds, not the CANNBench domain.

The first timing attempt stopped on a stale adapter binding for the candidates'
native resource caches. Its five real calls, two controls and three samples are
preserved and excluded. Corrected complete schedules changed only those cache
bindings; 18 guard checks retained rejection of other global/JIT changes and
post-warmup cache mutations. Candidate bytes and numerical checks stayed fixed.

| Immutable evidence | SHA256 |
|---|---|
| [Generation and both qualification protocols](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-vf-entry-study-20261127/vf-entry-study-public-20261127-01.tar.gz) | `5ea2c6a8bf74dcb7759435b727182c8f44659de02576bda2e9bd8ad9266ba97d` |
| [Timing, actual geometry and compiler UB](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-vf-entry-study-20261127/vf-entry-timing-supplement-20261127-02.tar.gz) | `a3cf4341b2d0801d63882774e4866ce86a18b957a88c53ea3f37d9ed8497fa42` |
| [Replication: all four sources, outcomes and samples](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-vf-entry-study-20261127/vf-entry-replication-public-20261127-02.tar.gz) | `83a08c8c767a3feea448cca7fbb3f810725083edb7918fca17ed7b262a1a8f9a` |

## Exp empty-slot comparison — 2026-11-27

The original pair and a separate replication used GLM-5.3-low/OpenCode 1.19.0
with the same prompt, four skills, checker and pinned runtime. Both compared
A `69c19624` with B `c9fc70ca`, differing only in the Exp ownership paragraph.
The original 13 responses account for 15,965 completion tokens; the replication
adds 17 complete responses and 9,876 completion tokens. Each request used
the profile-native output limit of 32,768. Provider retries were zero and no
candidate was repaired.

The fixed exercise used contiguous FP16 inputs of 64, 513, 2,048 and 2,049
elements, seeds 19/47, base 2, scale 0.5 and shift 0.25, with eight blocks and
256-element tiles. The four unchanged final sources had these CaModel outcomes:

| Cohort and generation order | A | B |
|---|---|---|
| Original, B → A | Failed lowering before first launch; seven pairs unrun | Passed eight real checks and eight rejecting controls |
| Replication, A → B | Passed eight real checks and eight rejecting controls | Passed eight real checks and eight rejecting controls |

Original A reassigned a runtime `PlainValue` to Python integer `0`. Both new
sources passed: 16 real checks and 16 rejecting controls in total, with zero
unexpected Model or DMA errors. This is the predeclared tie/nonreplication of
the original pass-versus-failure outcome. Keep the cohorts separate; they do
not establish a causal or general benefit from the guidance. There is no
matched timing or hardware result.

The experiment controller initially supplied a `host-review.json` with the
wrong schema, stopping setup before candidate import or launch. Corrected
metadata was used in separate qualification scopes; candidate, checker and
runtime bytes were unchanged. The failed setup remains retained and is not
counted as a candidate failure.

The syntax reference now records the observed type error and B's qualified
bounded-loop form. That addition is post-study learning and was not part of
either study's treatment.

| Immutable evidence | SHA256 |
|---|---|
| [Original sources, failure and outcomes](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-exp-empty-slots-study-20261127/exp-empty-slot-study-public-20261127-01.tar.gz) | `408880faf79639eb34572cdd5ae922727c57b5e4cd8b5d169862ee8f2251f7c5` |
| [Replication and all four outcomes](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-exp-empty-slots-study-20261127/exp-empty-slot-replication-public-20261127-01.tar.gz) | `7e704192984784aa4bb2b4ffd9e0e38d1fd3031b7148fab7d058eebe25fb7149` |

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
