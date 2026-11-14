# CANNBench kernel checks

CANNBench provides external NPU evaluation without access to a hardware host.
The shared CI path checks one selected operator recipe against its current
public task contract. **SwiGlu** is the currently qualified recipe and the
standalone reference-smoke default. It uses the existing service adapter and strict hardware
result checker. There is no scheduled or full-catalog campaign.

The reference smoke below is manually invoked and credit-consuming, not a PR
CI job. A separate [manual CI dispatch](#one-fresh-candidate-through-ci) evaluates
a freshly generated candidate. The shipped
kernel is a historical PyASC reference, not a newly generated candidate. A pass
establishes packaged-kernel correctness and timings on the reported runner only.
It does not qualify fresh generation, all operators, local hardware, or the
simulator's suppressed-launch numerical-control contract.

## Verify an official case in CaModel

Start in the stack checkout with Git, Python 3.10+, Docker and a configured OpenCode provider.
Choose a case below. The checker exports its task and official evaluator from
the same pinned commit in your CANNBench clone.

| Case | Task under `tasks/` | CANNBench commit |
| --- | --- | --- |
| `exp/1` | `level1/exp` | `597a915ec49735ee7d53057554bde426ef8c1d96` |
| `softmax/1` | `level2/softmax` | `597a915ec49735ee7d53057554bde426ef8c1d96` |
| `transpose/2` | `level3/transpose` | `597a915ec49735ee7d53057554bde426ef8c1d96` |
| `maximum/1` | `level2/maximum` | `08d519c503843bce5fd4672ffa2259abeb22fb00` |
| `strided_slice/1` | `level3/strided_slice` | `08d519c503843bce5fd4672ffa2259abeb22fb00` |
| `gather/1` | `level2/gather` | `08d519c503843bce5fd4672ffa2259abeb22fb00` |

Gather receives the declared int32 indices. At `08d519c5` the official
reference converts them to int64 internally for the CPU comparison.

```sh
STACK=$(pwd)
BENCH="$HOME/cann-bench"
git clone https://gitcode.com/cann/cann-bench.git "$BENCH"  # once
PROJECT=$(mktemp -d)  # use PROJECT="$STACK" to work in a fresh stack checkout
CASE=exp/1
TASK=level1/exp
PIN=597a915ec49735ee7d53057554bde426ef8c1d96
mkdir "$PROJECT/task"
for file in desc.md cases.yaml proto.yaml golden.py; do
  git -C "$BENCH" show "$PIN:tasks/$TASK/$file" > "$PROJECT/task/$file" || exit 1
done
```

In a separate project, link the skills and render the provider profile there.
This requires your configured `OPENCODE_BASE_URL` and `OPENCODE_API_KEY`.
A stack checkout already contains the skills. Keep the generated candidate and
provider configuration local.

```sh
if [ "$PROJECT" != "$STACK" ]; then "$STACK/install.sh" --target "$PROJECT/skills"; fi
python3 "$STACK/tests/tools/opencode_profile.py" --profile cloud-glm-5.3-low \
  --skills on --require-credentials --out "$PROJECT/opencode.case.json"
cd "$PROJECT"
OPENCODE_CONFIG="$PROJECT/opencode.case.json" opencode run \
  "Write only ./candidate.py for CANNBench $CASE, using its row in task/cases.yaml and the public signature in task/golden.py. Use asc.experimental.asctile and load asctile-cannbench-kernel, asctile-api-patterns, asctile-syntax-constraints and asctile-build-run-verify. I handle CaModel verification and packaging."
```

Check the emitted bytes in two fresh network-isolated simulator processes. The
first Docker pull needs network access. If OpenCode wrote the module in a
subdirectory, pass its actual path to `--candidate`. Use a new output directory
for each check. The three original cases allow five minutes per phase. Maximum,
StridedSlice and Gather allow ten minutes per phase. A timeout records the
active seed and any completed comparisons in the result. For a longer check of
the same candidate, select a new run directory and pass a positive
`--phase-timeout-seconds`. For example, the [verified full-size Gather 01 candidate](../../docs/verification-status.md#cannbench-camodel-pilot)
used `--phase-timeout-seconds 5400` and spent about 43 minutes in the real
phase. The report records the chosen limit.

```sh
RUN="$HOME/pyasc-qualification-runs/case-$(date +%Y%m%d-%H%M%S)"
python3 "$STACK/integrations/cannbench/smoke.py" verify-case \
  --case "$CASE" --candidate "$PROJECT/candidate.py" --benchmark "$BENCH" --run-dir "$RUN"
```

The command prints the case, verification stage and path to `result.json`.
Exit zero requires the public signature, real launches with preserved inputs,
correct shape and dtype, official numerical comparisons on two seeds and
rejection on both seeds when the launcher is suppressed. Per-phase logs identify
a compile or runtime failure. An absent candidate stops at the input stage.
The report records the candidate and checker hashes, benchmark and runtime pins.
Generation provenance comes from the OpenCode session that produced the file.
If compilation fails, keep the original file and report, then ask OpenCode to
repair the reported error into a new file and verify that file. For example,
a UB overflow reports the available and required bytes; reducing live tile
storage or unrolling can bring the kernel within the compiler budget.

The [three-case pilot](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-cannbench-camodel-pilot-20261107)
includes accepted sources, failed attempts, per-seed comparisons and checksum
receipts. It covers 3 of the catalog's 1,060 cases across 3 of its 53 operators.
A separate
[skill comparison](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-cannbench-camodel-pilot-20261107/cannbench-skill-scope-comparison-20261107.tar.gz)
tested Exp 01 and Transpose 02 with one fresh attempt per case and skill version.
Both candidates produced with the case-scoped instructions passed. With the
earlier instructions, Exp passed and Transpose reached the generation deadline
without producing a candidate. Its last response has incomplete usage.

## Requirements

- A full checkout, Python 3.10+ on Linux/macOS, `curl`, and the official BenchSite
  adapter (`bash integrations/cannbench/setup-site-mcp.sh`).
- An existing aggregation scheme owned by or shared with the account. Pass its
  ID as `--aggregation-token` (or `CANNBENCH_AGGREGATION_TOKEN`); this is required
  for private submissions too ([current API contract](https://cannbench.com/docs/mcp)). The smoke never creates schemes or opts into
  third-party independent runner environments.
- `BENCHSITE_API_TOKEN` in the environment, or an existing private environment
  file passed as `--credentials-file`. Never commit or print it.
- A self-contained PyASC runtime wheel for **CPython 3.12 / Linux x86_64**, with
  `asc`, `asctile` (legacy) or `asc.experimental.asctile` (modern), `pybind11`
  and their licenses. The optional [source-build path](../../docs/source-runtime.md)
  assembles the modern layout with numpy and typing_extensions as well.
  Supply its SHA256 explicitly.
  Runtime binaries are external inputs, not checked into this repository.
  The packager copies only runtime/license files; it replaces candidate code,
  wheel metadata and RECORD. The ZIP contains the candidate sources, `setup.py`
  and `build.sh`; the build stages the exact hash-verified prebuilt wheel. Arbitrary PyASC revisions are not presumed compatible.

The qualification runtime bundle is available in the preserved history at
`origin/archive/campaign-2026`, under
`integrations/cannbench/submission/` (build recipe and vendored runtime).
See [the smoke result](../../docs/hardware-lane.md) for the exact tested binary
hash and retained archive. Use that binary or build and qualify a compatible
bundle; the pinned simulator image is not a CANNBench submission runtime.

## Prepare, submit once, collect

Use one persistent state directory per service account on the submitting machine.
Results must be outside checkout. `prepare` takes a fresh live task snapshot,
syntax-checks candidate sources and produces a deterministic package. It does
not claim local compilation or NPU correctness.

```bash
PYTHON=.tools/benchsite-mcp/bin/python
RUN="$HOME/pyasc-qualification-runs/swiglu-smoke-$(date +%Y%m%d-%H%M%S)"
"$PYTHON" integrations/cannbench/smoke.py prepare --run-dir "$RUN" \
  --runtime-bundle /absolute/path/runtime-cp312-cp312-linux_x86_64.whl \
  --runtime-sha256 <verified-sha256> --aggregation-token <existing-scheme-id>
"$PYTHON" integrations/cannbench/smoke.py submit --run-dir "$RUN"
"$PYTHON" integrations/cannbench/smoke.py collect --run-dir "$RUN"
```

Replace the hash placeholder before running. Each deliberate new submission
consumes a credit; inspect the service balance and keep within the available
10/day working budget. `submit` checks credits, active jobs and task drift;
it saves a durable intent **before** its only possible upload. It never retries
POST automatically. An explicit HTTP rejection is retained as `rejected`; prepare
corrected inputs before a deliberate new submission. Unknown delivery remains
blocked until GET reconciliation. The default ledger is `~/.local/state/pyasc-cannbench-smoke`.
Keep it and the run directory after interruptions. Do not work around an
unresolved intent by changing state directories.

`collect` only issues GETs, reconciles an uncertain upload by its original tag,
and saves `job.json` and, when terminal, `report.json`. Repeat `collect` while
queued/running (exit 2), not `submit`. Exit 0 from `submit` confirms an intent/job,
not hardware success. Exit 0 from terminal `collect` requires the exact official
case set, correct benchmark/version/hardware, successful numerics, zero anti-cheat
failures and positive finite timings. A failed hardware gate exits 1 and retains
the service report. No automatic repair/resubmission occurs.

To check a generated SwiGlu candidate later, pass `--kernel-dir` containing
`swi_glu.py` and `_pyasc_runtime.py` to `prepare`. Use the task's callable contract,
perform numerical work in the PyASC kernel, and record generation provenance
separately. No second model is required by this smoke path.

Keep full evidence privately outside Git. For accepted qualification, retain a
durable archive and publish only a short result, archive checksum and reference.

## One fresh candidate through CI

The optional manual **CI** dispatch accepts `cannbench_action=generate` for one
fresh `gateway/glm-5.3` generation and hardware evaluation selected by `operator`
(default `swi_glu`). Ordinary PR
checks do not invoke a model or consume CANNBench credits. No nightly schedule
is configured.

A dedicated runner labelled `cannbench-vm` needs OpenCode, Docker, `gh`, the
compiler image and exact runtime listed in `ci/single-runtime.json`.
Configure repository secrets `OPENCODE_API_KEY`, `BENCHSITE_API_TOKEN` and vars
`OPENCODE_BASE_URL`, `CANNBENCH_RUNTIME_BUNDLE` (runner-local absolute path),
`CANNBENCH_AGGREGATION_TOKEN` (an existing owned/shared scheme). The job installs
its service adapter into an external temporary venv. No interactive checkout or
private remote transport is part of the product path.

Release publication targets the exact generation commit. If that commit changes
`.github/workflows/` relative to the default branch, [GitHub requires workflow
write permission](https://docs.github.com/en/rest/releases/releases#create-a-release),
which the built-in `GITHUB_TOKEN` cannot receive. Configure a repository-scoped
GitHub App or fine-grained token with **Contents: write** and **Workflows: write**
as `QUALIFICATION_PUBLISH_TOKEN` for that case. For a SwiGLU performance publication, an existing lightweight `qualification-swiglu-performance-EVALUATION_RUN` tag at the exact evaluation revision lets CI publish with its built-in token. The publisher verifies that tag before attaching evidence. Only the release-publication step
receives this optional secret; it is not available to generation or CANNBench.
A publication failure retains the sealed archive. Resume the same origin through
CI after configuring the credential; never regenerate or resubmit for publication.

```bash
gh workflow run ci.yml --ref <branch> -f cannbench_action=generate -f operator=swi_glu
# Repair a compiled but numerically failed model output, preserving its CI lineage:
gh workflow run ci.yml --ref <branch> -f cannbench_action=repair -f origin_run=<failed-CI-run-ID>
# Packaging GET failed after qualification, before any submission intent:
gh workflow run ci.yml --ref <branch> -f cannbench_action=continue -f origin_run=<qualified-CI-run-ID>
# If the original submission is still pending or collection was interrupted:
gh workflow run ci.yml --ref <branch> -f cannbench_action=resume -f origin_run=<original-CI-run-ID>
```

Recipes live in `ci/operators/`: a JSON contract names the official operator and
public function, prompt, numerical probe and required preflight count. The shared
driver selects the task, packages that function, qualifies it, and binds evidence
to the recipe in the generation source snapshot. Unknown recipes fail before
model invocation or submission. `repair` must select the same task as its parent;
`continue` and `resume` use the parent's recorded recipe, ignoring dispatch input.
Checkpoint recovery accepts both current and historical artifact names.

To add an operator, check in its recipe, prompt and adversarial numerical probe,
exercise offline packaging/task-binding tests, then explicitly qualify it. The
probe must consume the candidate directory and output report path and record
`passed`, per-case `passed` values and `candidate_sha256`; the required case count
comes from the recipe. Offline alternate-operator fixtures test the plumbing,
not hardware support. Do not generalize the SwiGlu result to another recipe.

The workflow loads selected skills via OpenCode's native skill tool, records
model/configuration and unavailable provider revisions as `unknown`, and
preserves each generated/repair output separately. Session evidence contains
read-only native OpenCode records for that session, with its actual model ID;
it does not depend on large JSON surviving the CLI stdout export. A compact API summary is
extracted from the exact runtime sources; the model is limited to 12 agent
rounds and 24000 output tokens per response (including reasoning). Timeouts without a candidate stop
the run rather than repeat unproductive exploration. It allows at most three
local model attempts and one hardware submission per origin run. It does not
automate a separate reviewer or tuning session; see [the skill workflow and
follow-up](../../docs/ci-gates.md#agent-workflows). Qualification
includes exact-runtime dispatch/lowering of the official case set and separate
representative numerical checks in the pinned newer CI simulator. Only the
complete official CANNBench NPU result establishes hardware acceptance.

`generated-operator-prepared-*` is uploaded before POST. Submission state is
checkpointed afterward; `generated-operator-evidence-*` retains final/interrupted
work. Run directories live outside checkout under
`~/.local/state/pyasc-generated-ci`; the shared account ledger remains
`~/.local/state/pyasc-cannbench-smoke`. Resume uses the original source revision
and performs **GET only**. It can restore an unexpired central checkpoint when
local state is lost; an unknown delivery never authorizes another POST. If the
queue exceeds the bounded CI wait, dispatch `resume`, not `generate`. Generation
failures retain their evidence and consume no submission credit.
CI JSON GET requests retry temporary HTTP 502/503/504 responses up to four
times with bounded backoff. Submission POST is never covered by that retry.

`repair` starts a distinct CI run using a candidate with retained qualification
failure (including a compiler timeout) from the original
run's central artifact. It preserves the original records and hashes, records
the parent commit, and requires the same task snapshot. It refuses active runs
or evidence that reached packaging/submission intent: use GET-only `resume`
for those. A model response that leaves a failed candidate unchanged stops
instead of starting another identical repair attempt.

`continue` imports the central evidence of a completed run whose candidate
already passed both qualification gates but whose package preparation failed.
It verifies candidate hashes and native generation records, retains the entire
parent, and retries packaging in a new CI run without invoking the model.
Generation source/model provenance stays attributed to the original CI;
continuation source and CI identity are recorded separately. A fresh live task
snapshot must still match. Any submission intent or submission state forbids
this path: use GET-only `resume` instead. This is not a retry of an upload.

The workflow re-downloads the evaluated submission and verifies its exact hash.
A successful result includes per-case timings, the service score and the
separately computed baseline/candidate ratio. It publishes a checksum-verified
qualification archive as a GitHub Release asset under
`qualification-<recipe>-<origin-run-ID>` (historical tags remain unchanged), beyond normal artifact retention.
The [recorded generated result](../../docs/hardware-lane.md) passed 20/20 on
hardware with complete CI provenance. CI published and verified its unchanged
archive in the [qualification Release](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-swiglu-41673353296);
the separate reference qualification is retained with its own scope.
Implementation or a queued upload alone does not qualify generation.

### Generate and measure SwiGLU from CI

Run `gh workflow run ci.yml --ref main -f cannbench_action=performance` to generate three vector kernels with OpenCode in a fresh project. Actions retains the native session, exact candidate bytes and four skill digests. A network-isolated CaModel gate checks both seeds against the numerical oracle and a suppressed launch. The submitted ZIP includes the unchanged PyASC, its emitted AscendC and the host launcher. CANNBench rebuilds the wheel using the installed CANN toolkit.

OpenCode writes the compute kernels and a finite shape-based `policy.json` from the four skills. The interpreter applies the model's tiling, unroll and block allocation in CaModel and in the submitted wheel. Each generated specialization is lowered before upload; the original policy is included at `cann_bench/policy.json` in the ZIP. Its SHA-256 and the skills revision are recorded in the public receipt. A manual dispatch can set `skills_revision` to a full repository commit to evaluate independently reviewed guidance before it lands. Trusted pushes to main that change any of the four skills run the same evaluation after the PR and examples gates pass. To omit the external evaluation for a single main push, end its commit message with the Git trailer `CANNBench-Evaluation: skip`. The simulator gates and dashboard publication still run. Dashboard-only and PR updates do not submit. The dashboard reads checked service receipts into its Timing & service metrics history and shows both the generated cohort and the complete service score.

[Case-by-case comparison of the first service evaluation and local measurements](../../docs/swiglu-ci-comparison.md).

For FP32 row kernels the model can select three stages when full tiles cover both dimensions. The host dispatches bounded one-stage copies for incomplete tiles in that band. CaModel checks the full tile and its row and column boundaries with both seeds before packaging. See the [worked tuning example](../../skills/asctile-api-patterns/references/strided-row-tiles.md#full-tiles-on-long-fp32-grids).

Three generated kernels implement 19 official cases. The packed BF16 template implements case 12. The published performance ratio is the geometric mean of each generated case's official baseline time divided by its measured time in the same submission. The service score includes all 20 cases. A single result establishes performance for one candidate.

The compute turn uses GLM-5.3 high reasoning and freezes the written Python. The policy turn uses the GLM-5.3-low profile with the same native output limit. A confirmed length stop permits one bounded continuation for the missing file. If a CI generation exhausts both policy responses, or its CaModel evaluation reports a UB overflow before submission, pass its run ID as `origin_run` with `performance`. CI checks the original model write and context, then generates only the policy. After an overflow the model receives the rejected table and the compiler diagnostic. The receipt identifies the original kernel run and the failed policy.

For a split run, `performance-generate` produces the generation artifact. Use `performance-evaluate` with `-f origin_run=GENERATION_RUN_ID` to qualify and submit its bytes. If the service is still pending, `performance-resume` takes the **evaluation** run ID and only collects the existing submission. After a confirmed service-maintenance rejection, `performance-submit-prepared` takes the evaluation run ID, verifies the saved GitHub Actions package and checks that submissions have reopened before sending it. Each successful evaluation publishes a release with `performance.json`, the submitted ZIP, the generated source and their SHA-256 checksums. GitHub Pages reads that release.
