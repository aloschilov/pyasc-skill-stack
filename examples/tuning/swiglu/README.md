# One measured SwiGLU tuning session

This checkout recipe lets OpenCode qualify a supplied SwiGLU baseline, propose one parameter change, compare the unchanged snapshots in CaModel, and select from actual measurements. It uses the existing installer, generation executor and numerical verifier. The skills-only archive does not include this evaluator.

**Readiness:** source and offline checks are available; the v2 source dependency is published. The clean Linux/native OpenCode preflight below must pass before a model session is admitted. No new hardware result is claimed by this recipe or its source dependency archive.

The public [source dependency](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-vf-entry-study-20261127/swiglu-row-tuning-engine-v2.tar.gz) contains the pinned CaModel qualification/timing helpers and official CANN comparator sources. Its SHA256 is `85979448d3b0d1a77e9c4b36c4f167fd90249089f332d1df732bfe61cf404f1d`. Version 2 changes only the aggregate timing budget and its diagnostics; version 1 remains preserved. The entry checks every source member against [engine-lock.json](engine-lock.json), rejects links and unexpected members, and copies the existing checkout checker by hash. Source licenses and provenance accompany the dependency; no runtime wheel, result tensor or provider transcript is included.

## Prerequisites

Use a clean reviewed checkout on ordinary Linux, Python 3.10 or later, OpenCode **1.19.0** on `PATH`, Git, and access to Docker as the current non-root user. The image in [engine-lock.json](engine-lock.json) must already be present. The recipe does not pull images, change machine configuration or install credentials. It uses the caller's UID/GID, two CPUs, 8 GiB memory and no container network. Docker must have no other running workload.

Use the project's **existing generation executor state directory** for every command. An unresolved latch blocks execution; choosing another directory is not reconciliation. Cross-machine resource ownership remains the operator's responsibility. For the real session, supply `OPENCODE_BASE_URL` and `OPENCODE_API_KEY` using the existing `cloud-glm-5.3-low` profile. They are not needed or read by the synthetic preflight.

All run directories must be new and outside the checkout. Use paths without whitespace or shell metacharacters. Preparation binds the entry script, checkout revision, complete helper source, profile and four installed skill trees; edits after preparation require a new explicit preparation, never mutation of an existing run.

## Prepare, preflight, run

Download the source archive from the link above. From the clean checkout, set `ENGINE` to that file, `STATE` to the existing executor state directory, and `RUNS` to an external evidence directory. These variables contain paths, not credentials.

```sh
python3 -B examples/tuning/swiglu/recipe.py prepare \
  --directory "$RUNS/preflight" --engine "$ENGINE" \
  --skills-ref HEAD --state-dir "$STATE"
python3 -B examples/tuning/swiglu/recipe.py verify --directory "$RUNS/preflight"
python3 -B examples/tuning/swiglu/recipe.py preflight --directory "$RUNS/preflight"

python3 -B examples/tuning/swiglu/recipe.py prepare \
  --directory "$RUNS/session" --engine "$ENGINE" \
  --skills-ref HEAD --state-dir "$STATE"
python3 -B examples/tuning/swiglu/recipe.py run \
  --directory "$RUNS/session" --preflight "$RUNS/preflight"
```

`prepare` and `verify` are offline. They install four skills through the repository installer into the consumer's context, retaining symlinks and exact alias/canonical read permissions. They neither run a candidate nor contact a model.

`preflight` first checks the six fixture/seed CPU oracles: native outputs accepted, zero and poisoned outputs rejected. A fixed **synthetic** local response sequence then drives real OpenCode tool calls. Before any CaModel work it must observe allowed baseline/reference reads and native denial of a harmless unapproved command. It qualifies the retained baseline and retained parameter derivative and exercises the comparison and decision commands. Synthetic responses are not GLM evidence or billed usage. This phase can consume at most **108 real simulator invocations and 18 suppressed controls**, separately from later sessions. It has no external provider, CANNBench or NPU access.

`run` admits exactly one fresh GLM-5.3-low session using the same tools through the existing `run_session` provider gate. The native output limit remains **32,768**. Limits are **12 requests**, **200,000 output tokens**, **900 seconds of non-tool elapsed time** within **2,700 seconds total**. Before each request the gate reserves the full native output limit against the remaining token budget; it stops rather than reducing that limit. The existing client watchdog also enforces the 900-second budget before the first request and after the final response. Only bounded, nonoverlapping evaluator intervals receive time credit: live intervals require the original PID/startticks and pinned action identity, and terminal receipts must bind the same immutable clock. The watchdog never waits behind an in-flight provider request. Raw responses and actual usage are retained privately. Unknown delivery or usage remains unresolved and blocks further execution.

There is one evaluated variant. The model must record its hypothesis and intended control before evaluation. Only bounded `_row_geometry` height parameters may change; device arithmetic, width calculation, resource query, UB loop and launch-count formula remain unchanged. No source repair or replacement is scheduled after an outcome. The exact task and selection rule are in [TUNING_TASK.md](TUNING_TASK.md).

## Evidence and limits

Every qualified snapshot must pass six numerical checks and six rejecting suppressed-launch controls, input canaries, unchanged-input checks and native-error gates. The comparison performs 24 warm series in both ABBA and BAAB order: **96 real invocations, 72 retained samples and six suppressed controls**. It uses the same module/JIT objects within each series, with actual resource queries, grids, effective compiler options, UB and CPP/ELF/source bindings. Each complete session is bounded by 108 real invocations and 18 controls; actual counts remain in the receipts.

The comparison resource chain is fixed: 540 seconds of aggregate measurement time, a 560-second inner/client bound, 580 seconds for the owned supervisor, a 585-second subprocess wait, a 595-second full action cap and the unchanged 600,000 ms native bash timeout. Before dispatch, actual preparation plus the 585-second wait and five seconds of postprocessing reserve must fit the action cap. Delayed preparation therefore stops before Model work. Resource-budget failure or incomplete timing does not classify a numerically qualified candidate as incorrect. Series counts and correctness checks are unchanged.

The baseline is retained on a tie, ambiguity, incomplete comparison or material per-shape regression. To select the variant, no shape may regress by more than 1% in either order, both orders must agree in direction for every shape, and at least one shape must improve by more than 1% in both orders. The 1% engineering margin comes from retained simulator variability; it is not a confidence interval or hardware-noise estimate.

Read `result.json`, `attempt/tuning-accounting.json` and `evaluation/*/public-receipt.json`. The evaluated sources, hypothesis, final native writes and measured selection are linked by hashes. A known terminal rejection can retire only its own latch after process/container absence is confirmed. An unknown outcome retains the latch and original evidence; no command is automatically replayed. A successful client exit without the required tuning procedure is not a successful tuning result.

The measurements cover only the three finite FP16 fixtures and synchronized CaModel launch ticks. They do not establish NPU performance, wider operator coverage, reliability across fresh sessions or general skill superiority. An A/B study must predeclare its revisions and order, use new directories with the same executor state and tools, retain every outcome, and compare cohorts separately. The preflight is operational evidence, not one of those model sessions.

Offline regression tests:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tests/unit/python -p 'test_swiglu_tuning_*.py'
```
