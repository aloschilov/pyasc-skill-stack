# Verification status

Gate vocabulary: `passed | failed | blocked | not-run`. A missing dependency
cannot count as successful acceptance. Historical evidence remains unchanged.

The [delivery roadmap](roadmap.md) lets local/CI work proceed independently of
hardware availability. It preserves separate software and hardware acceptance;
neither a prepared payload nor a green simulator gate qualifies an NPU.

## Local follow-up, 2026-11-02

### External hardware smoke

The repaired reference SwiGlu smoke **passed** strict CANNBench acceptance:
20/20 correct and measured cases, zero reported anti-cheat failures, 191.24 s
execution after 506.17 s queued. Profiler results identify the PyASC device
kernels. The unweighted geometric mean baseline/candidate timing ratio is
0.317322x, so this is correctness qualification with measured timings, not an
acceleration result. The first 18/20 failed attempt remains archived. This does
not qualify model-generated code or close the generation-attributed CI score
requirement. See [hardware evidence and limitations](hardware-lane.md).

### Separate-project workflow qualification

The README's installer/profile workflow was exercised in a separate project
with spaces in its path, using a fresh venv and the existing VM gateway. Agent
events confirm successful loading of the API, syntax and workflow skills and
reading of the installed kernel/design templates. OpenCode generation took
105.95 s. The unchanged kernel passed strict qualification in 14.91 s after
fixing a scorer false negative for a named literal size list. Its score is
12/16 at the unchanged threshold; JIT took 3.15 s and execution 11.72 s.
Two launches and a falsifying suppressed-launch control established simulator
numerical verification. The report conservatively records only shape `[1024]`.

The first candidate used an unsupported custom comparison and remains failed.
The second initially scored 11/16 despite its `test_sizes = [1024, 1025]` loop;
that failed manifest is retained alongside the successful requalification.
Neither candidate was manually repaired. The collector's long-argument bug
was also fixed, with regression coverage. Named-loop scoring tests reject
mutation, aliases, unknown values and wrong-scope assignments. Frozen historical
score outcomes and legacy detection tests still pass; the original scorer hash
is preserved with an explicit intentional-update entry in the cleanup baseline.

Kernel SHA256: `3ca932dfd8e561242f4b0c385cb511db1f43a7de2feaa8dbdd1c8250cc2673ba`.
Accepted archive, on workstation and VM:
`~/pyasc-qualification-runs/repo-stabilization-project-20261102-attempt2/evidence-qualified.tar.gz`;
SHA256 `3710a597eb253e2180d47373a8d80d690077c7283a4f6777cf60feb259c9317c`.
It retains the original source export, scorer before/after, failed and accepted
manifests, logs and unmodified generated source. All recorded log hashes and
original/qualified kernel equality were verified. Generation identifies base
`a700d78` plus working-copy input hashes; requalification identifies inputs by
hash, not a claimed clean Git revision. This is simulator evidence only.

The existing release workflow's build/install steps were also exercised locally;
tar and zip payloads matched (52 files), with macOS metadata disabled for the
Linux payload comparison. The archive remains skills-only as documented.

### Repository stabilization smoke

The documented `generation_acceptance.py` command passed from an isolated VM
source copy whose path contains spaces, using a new Python venv, existing
OpenCode 1.19.0 gateway credentials and the pinned ARM64 development image.
The complete attempt took 61.84 s. Its unchanged generated vector-add kernel
passed static/score (12/16), JIT and simulator numerical qualification in
12.45 s, with two counted launches and a falsifying suppressed-launch control.
The report records `shapes_verified: [[1024]]`; no broader shape, hardware or
performance qualification is claimed.

Kernel SHA256: `8c212a025bdd3a9bde485d96d1f86d83c9ad614f3d19cf288114cd70e64831f4`.
Evidence: `~/pyasc-qualification-runs/repo-stabilization-20261102-161049/evidence.tar.gz`
on the workstation and VM; SHA256
`60a5c4f98768651c31f7c09a92d3dbc48a2f8ace62028980263757427a84dbc2`.
Returned manifest log hashes and equality of the original/qualified kernel
were verified. The manifest identifies base revision `072deb5` and exact input
hashes from the working copy, including the installer regression subsequently
committed in `f349988`; it is not represented as a clean run of that later commit.
The archive retains a launcher-script quoting failure before generation began.
The existing dirty VM checkout was not changed. This run covers the bounded
generation quick start, not yet the separate-project interactive instructions.

### Fresh generation on the VM, without NPU

One fresh OpenCode 1.19.0 / `gateway/glm-5.3` attempt used the tracked
`cloud-glm-5.3` profile and skills at `513b888`. It produced a vector-add kernel
and design document in 77.87 s. Static checks and the unchanged scorer passed
(12/16). The original kernel then passed JIT and numerical verification in the
pinned ARM64 simulator image, with two counted launches and a suppressed-launch
negative control. Qualification took 12.33 s (JIT 2.89 s, execution 9.41 s).
The source exercises sizes 1024 and 1025; the report's conservative
`shapes_verified` field records only `[1024]`. This is simulator evidence only.

Kernel SHA256: `9094ff69c5b0f8e97436a936bbd8d65fd6df11130b826c9080b263d0f1399bf3`.
External archive, retained on the workstation and VM:
`~/pyasc-qualification-runs/vm-generation-20261102.tar.gz` (workstation),
`~/pyasc-qualification-runs/vm-generation-20261102/evidence.tar.gz` (VM).
Archive SHA256: `215f69194069476247b09b2a7bd8ecfea55feb901d09c800f2f4bb6ca12c0e1a`.
All 53 archived file hashes and the unchanged kernel were checked after return.

The archive also retains setup failures: missing Git in the evidence collector,
an initially bypassed image entrypoint, and an invalid manually supplied backend
name. The accepted run uses the image entrypoint and `Model` backend. Its only
harness patch lets `RunArtifacts` use an explicitly supplied revision without
invoking Git; verification scripts and the generated kernel are unchanged.
When neither Git nor an explicit revision is available, the manifest records
an unknown revision while retaining input hashes. No credentials or run logs
are committed to the repository.

### Earlier local checks

PR #19 review fixes and the shared matrix outcome policy were locally checked
on the source content committed as `dc1ebfe` (the run manifests retain the
pre-commit revision and exact dirty-tree input hashes):

- Full PR gate: 10 groups, 43 Python tests, no skips; 9.64 s.
- Pinned Docker contract: 18/18 expected outcomes, unchanged inputs; 101.37 s.
- Both historical mutation regressions detected; 3.82 s.

The unchanged add generated on David failed local JIT import: the pinned wheel
does not expose top-level `asctile`. This failure is retained. A separately
hashed manual derivative uses the shipped template's wheel/legacy import
fallback and `asc.GlobalAddress`/`asc.ConstExpr` annotations. Its algorithm is
unchanged; static/score/JIT/numerical qualification passed in 31.13 s, with four
launches at sizes 1, 1024, 1025, 4097 and a falsifying suppressed-launch control.
Derivative SHA256: `2d050f18a547829549ca2cbbb34fe3374b9c359046f165f8f1e776674c8a304f`.
This is portability requalification, not fresh generation or NPU acceptance.
Fresh local generation was not started because the tracked provider's endpoint
and key environment were absent. No credentials were copied to work around it.

External archive: `~/pyasc-qualification-runs/pr19-local-qualification-20261102.tar.gz`,
SHA256 `e740cc728743332fda5e7bbaa858a5a736e2e23b0b63b5144e390bb96be2517f`.
It contains both candidate outcomes, manifests, sources and verified log hashes.
Current-head CI results are recorded separately in [PR #19](https://github.com/aloschilov/pyasc-skill-stack/pull/19);
these local checks do not establish that a new GitHub workflow ran. The NPU
window was deferred; no new device command was sent in that qualification.

## Generation and harness qualification

The close-generation-loop implementation has passed local qualification and
real PR checks in [PR #18](https://github.com/aloschilov/pyasc-skill-stack/pull/18).
It was merged as `1ab1940` on 2026-10-30. The evidence below identifies each
tested revision explicitly; later changes require their own checks.

Enforcement is manual, by convention. On 2026-10-30, the `main` branch protection
API returned “Branch not protected” and the repository ruleset list was empty.
Neither `pr-gate` nor `examples-gate` is a required GitHub check; a red check
does not technically prevent a merge. Both must complete successfully on the
current PR head before manual acceptance. Queued, cancelled, skipped or missing
checks are not acceptance. Both jobs depend on the self-hosted arm64 runner;
if it is offline, work can remain queued and examples cannot run until pr-gate
passes. Workflow execution timeouts do not bound time waiting for a runner.

| Gate | Status | Basis |
|---|---|---|
| generation (static, historical) | passed, marginally | three runs on 2026-10-28; scores 12/12/11, threshold 12; all runtime checks SKIPped |
| generation (executed, fresh) | passed locally | one fresh attempt passed end to end; a second was initially rejected by a scorer tuple-loop defect, then its unchanged artifact passed corrected static/JIT/numerical qualification |
| generated-kernel regression | passed in PR CI | 16/16 cases; matrix 88.47 s, full examples job 106 s; includes run-1/run-3 positives, original run-2 expected compilation failure, template, fixtures and three complete skill listings |
| independent review | not-run | executing kernels is not independent review |
| hardware | blocked (execution attempted) | 2026-10-30: own David container, pinned checkout and OpenCode connectivity qualified. Direct NPU matrix attempted all 14 cases: 0 expected outcomes matched, no kernel launches; runtime setup returned GetSocVersionWrapper 507008. See [hardware lane](https://github.com/aloschilov/pyasc-skill-stack/blob/e154f4218554d6ff8119f2886de2632701b21d58/docs/hardware-lane.md#owned-container-attempt-2026-10-30). |
| performance | not-run | simulator checks are not performance evidence |

A separate 2026-10-31 own-container run generated a new add kernel and passed
static/JIT/CaModel numerics for four sizes, with a falsifying negative control.
Original GeLU and generated-kernel NPU attempts still fail before launch;
kernel logs now confirm a device-open namespace conflict, without identifying
the holder. See the [scoped result and limitations](https://github.com/aloschilov/pyasc-skill-stack/blob/e154f4218554d6ff8119f2886de2632701b21d58/docs/hardware-lane.md#own-container-generation-and-runtime-diagnosis-2026-10-31).


Local qualification on 2026-10-29 also passed both mutation checks (restoring
the template reassignment and removing harness platform setup), the fast PR
gate, portability, and nine focused harness/scoring unit tests. The 89.7 s
matrix time excludes image pull and CI queue/setup. The real PR examples job
completed in 106 s including setup, image pull and artifact upload, satisfying
the under-five-minute criterion for this run (not a future runtime guarantee).

Two fresh generations were retained, not selected by silently discarding
failures. The second used `for size in (1024, 1025)` with a numerical check
inside the loop. The scorer only recognized the list spelling and incorrectly
awarded 11/16. A tested AST recognition fix accepts that tuple form without
changing the threshold or check names. The original failed run remains failed
in its manifest; a separate requalification records PASS on the identical
kernel SHA256 `4b142083052b2ec85355ea10e717aa23230ab3c3924a38c1bbe3562202808f54`.
The first attempt passed with two launches, as did the requalified second.

Retained external archive: `close-generation-qualification-20261029.tar.gz`;
SHA256 `b2ad97856246522fac3427eef4783ad567116bf8195c10b44246def2969a86b4`.
Copies are retained outside both working checkouts on the workstation and VM;
the handoff supplies the local archive link. No shared URL is claimed for that
manual qualification archive; the separate CI evidence is linked below. [Storage and retention policy](verification-contract.md#storage).

The original acceptance requirement “all three frozen kernels PASS unchanged”
was invalidated by execution: run-2 reassigns `valid` from PlainValue to int and
raises UnsupportedSyntaxError before launching. It is retained unchanged and
must FAIL for that reason, with zero launches. The acceptance matrix explicitly
requires this result; it does not silently omit run-2.

The earlier statement that committed run-1 passed after adding a platform
preamble was misattributed to a hand-transcribed variant. The new matrix runs
copies of the exact committed files, with no source repair.

The harness owns platform setup, executes via runpy as `__main__`, counts
Launcher.run calls and reports a verification level. A suppressed-launch
control only establishes numerics when its failure is attributable to a
supported numerical comparison at the same source site that passed in the
real run. Timeout, missing report, and unrelated exceptions do not qualify.
See [the report and storage contract](verification-contract.md).

[CI run 40892900455](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/40892900455)
verified PR head `1d8231e3dd169911201a051b8c065d49c7bddd68` via GitHub's
synthetic merge revision `395ce055140133007471082aaf5fcd8ec6401bb8` on 2026-10-29:

- pr-gate: success, 15 s; ten test groups and nine focused unit tests, no skips.
- examples-gate: success, 106 s; all 16 expected outcomes, no failed cases,
  input hashes unchanged and matching the reviewed source.
- merge-gate: skipped by design on pull_request; no merge was performed.

[CI artifact](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/40892900455/artifacts/12903949791)
ZIP SHA256: `fcc53886f9b6991eabb4133c2a75374f259f3da47d412fa32dc424174f3f1e40`.
The downloaded ZIP, manifest and logs are retained outside the checkout;
artifact digest, individual log hashes and all source hashes were verified.
The Actions copy expires on 2026-11-28, so it is not the only retained copy.

These are dated checks of the identified revision, not an assertion that every
later commit passed. No nightly run is established by this change, and CI's
workflow has no scheduled generation trigger.

Commands (choose new result directories outside the checkout):

```bash
PYASC_PYTHON=python3 bash tests/ci-gate.sh --tier pr
python3 tests/tools/verification_contract.py --results-dir "$RUN_ROOT/contract"
python3 tests/tools/mutation_acceptance.py --results-dir "$RUN_ROOT/mutations"
python3 tests/tools/generation_acceptance.py --results-dir "$RUN_ROOT/generation"
```

The generation command needs the tracked profile's provider credentials and
OpenCode on PATH. It preserves every attempted generation and requires static
acceptance, JIT compilation and numerical execution of the same unmodified
artifact. Missing prerequisites yield blocked, not passed. One successful
attempt establishes a working route, not a generation reliability estimate.

Historical artifacts: [2026-10-28](evidence/generation-gate-20261028/). Their
README describes the original static-only observations and remains frozen.
Prior runtime, target-suite and infrastructure observations follow; they do
not establish the new candidate's CI or fresh-generation acceptance.

## The target-suite compile gate

Every module under `reference/asctile/target/` is compile-checked inside the
development overlay image pinned as `.dev_image.ref` in
[`../reference/images/ENV.json`](../reference/images/ENV.json), which ships
`torch==2.7.1+cpu` and `pytest==8.4.2` — upstream's own qualified versions,
see [`evidence-sources.md`](evidence-sources.md). Full sweep on Linux aarch64,
39 GB RAM:

```
compile-checked 24 module(s); 0 failed; 1 blocked
```

That is on a 39 GB development box, using the upstream overlay image. It is
identical to what the repository used to get from an overlay it built itself —
same 23 passes, same single blocked module, same per-module timings within
noise. Adopting upstream's image changed nothing about the outcome, which is
the evidence that it was a supply-chain change and not a behavioural one.

Roughly 12 minutes end to end. The one blocked module is `test_kl_div.py`:
the process is killed with SIGKILL (exit 137) even with 27 GB free.

On the CI runner, a Mac with 128 GB, the same module **passes** and the sweep
returns `24 module(s); 0 failed; 0 blocked`. Same image digest, same pinned
commit, same command — only more memory. That is what identifies the cause,
and it is worth being precise about why: exit 137 on its own would not have.
A timeout or a container limit produces the same code, so the exit status
narrows nothing. The differential does. Its largest case has shape
`[1000, 997, 1000, 2, 1]` — 1.99e9 float32 elements, **7.4 GiB for a single
input tensor**, several per case — and it clears on a host with room for them.

So: a host-capacity limit, demonstrated rather than inferred, and not a fact
about the kernel. The 39 GB development box blocks it; CI does not.

That is `blocked`, not `failed`, and the gate keeps the two apart: it fails
the job on a compile error and only reports a resource blockage. Collapsing
them would either turn the gate permanently red — at which point nobody reads
it — or hide a real regression behind an expected failure.

Two harness defects were found and fixed while establishing this, both of them
cases where a module looked broken but nothing had actually been checked:

- `run_target_test.py` built its `-k` selector as `<name>-<block_num>-`, which
  only matches modules whose ids are generated from parameters. Modules using
  an explicit `id=` on `pytest.param` — `test_gather.py` — matched nothing,
  pytest exited 5 with "14 deselected", and the module was reported as failed
  having run zero cases. It now probes candidate patterns and falls back.
- Whole-module runs compiled every case in one interpreter, and memory
  accumulates across compiles. `matmul_v3/test_matmul_v3_cube_only.py` (184
  cases) was killed at exit 137. Cases now run in batches of 24, each in a
  fresh process, and it passes in 288 s with no loss of coverage — which is
  the evidence that the problem was accumulation, whatever did the killing.

Note what this gate does **not** establish. It runs `--compile-only`, and in
that mode upstream mocks kernel launch and the numerical assertions. A green
sweep means 23 modules build against the pinned runtime. It says nothing about
whether any of them computes the right answer. Numerical execution of the
suite on the camodel remains `not-run`.

## CI actually runs

Worth stating plainly, because for a while it did not. Every job in `ci.yml`
targets `[self-hosted, arm64]`, and until 2026-10-26 the only registered
runner carried the label `cannbench-vm`. Nothing matched, so runs sat queued
indefinitely and the gate's green-ness was never tested — a workflow that
cannot be scheduled looks the same as one that has not been triggered.

An arm64 runner now serves it from a Mac via Docker Desktop
(`infra/self-hosted-runner/`). Both gates report green on it: `pr-gate` in
14 s, and `merge-gate` — which pulls the pinned overlay by digest and
compile-checks all 24 target modules — in about 27 minutes.

Bringing it up surfaced one failure mode worth knowing, because it does not
look like what it is: the prebuilt runner image carried v2.335.1, which GitHub
has retired. A retired runner connects, registers, prints "Listening for
Jobs", is then told it "cannot receive messages", deregisters, and loops — so
it never shows as online and the symptom reads as a bad credential. Auto-update
cannot fix it, since the update arrives over the channel the runner is barred
from. Rebuilding the image on a current base (v2.338.2) fixed it. The
Dockerfile now says so.

The runner is registered with a one-hour registration token rather than a PAT,
so it will not re-register after a reboot. See `.env.example` for the durable
setup.

## What the numerical gate covered

`add`, float32, shape `[1000]`, tile 128, one block. 1000 = 7×128 + 104, so the
final tile is a 104-element tail, handled the way the skills document it:
allocate a full physical tile, pass the logical extent through `real_shape`,
pad with the operation-neutral value.

`max_abs_err = 0.0` against `x + y`. The tail region `[896, 1000)` is asserted
separately, so a kernel that silently skipped it could not pass. Four
consecutive clean-cache runs, all exact.

The runnable version of this is
[`../skills/pyasc-build-run-verify/references/execution-example.md`](../skills/pyasc-build-run-verify/references/execution-example.md).

### Environment

- Runtime `pyasc-dev 1.1.1.dev20261021205441`, installed from PyPI, not a local build
- Import path `asc.experimental.asctile`
- Python 3.10, Linux aarch64
- CANN 9.0.0, simulator `Ascend950PR_9599`, Bisheng present
- No NPU

### One unexplained failure

The first simulator execution segfaulted (exit 139) before writing a result.
It has not recurred in four subsequent runs with cleared caches.

It was not the environment and not the kernel: the official `runtime-smoke.py`
passes in the same venv and path; the construct was bisected into four variants
— loop without tail, `real_shape` without loop, loop with constant
`real_shape`, and loop with per-iteration `min()` `real_shape` — and all four
pass, including the exact idiom that crashed; and `ulimit -n` is 1048576, so
this is not the simulator file-descriptor exhaustion the README warns about.

Recorded rather than explained. One unreproducible segfault is not enough to
call the numerical gate flaky, but it is not something to leave undocumented
either.

## The former blocker: an 8192-token output ceiling

Diagnosis only. No provider call was made to produce this section, and it is
kept as written so the reasoning can be checked against what followed.

**Resolved.** The tracked `cloud-glm-5.3` profile declares `output: 32768`
with reasoning turned down, and both generation entry points now render that
profile into the project instead of inheriting the host's global config —
which is where the 8192 cap lived. Under it, the gate completed in 51–145 s
and produced an artifact every time; see
[the generation gate section](#the-generation-gate-now-that-it-has-been-run)
above.

One correction the runs supplied. Option 1 below recommends `output: 131072`
and cautions that 8192 "should be confirmed, not assumed". A direct call to
the gateway confirms it was never a provider limit — it answers in about two
seconds, and `reasoning_effort: low` returns zero reasoning tokens where
`thinking: {"type": "disabled"}` still burned 40. The profile therefore
carries both keys, and 32768 (option 2) proved sufficient: no run since has
terminated on length.

Eight `gateway/glm-5.3` calls failed to produce their artifact, reported as
`incomplete-length`. Every length-terminated step reports `reasoning` and
`output` token counts summing to exactly 8192:

| reasoning | output | sum |
|---|---|---|
| 8190 | 2 | 8192 |
| 8191 | 1 | 8192 |
| 8191 | 1 | 8192 |
| 8192 | 0 | 8192 |
| 8187 | 5 | 8192 |
| 8185 | 7 | 8192 |

Six independent runs landing on the same total is a configured cap, not a
verbose model. Steps in the same sessions that end with `reason: "tool-calls"`
show reasoning counts of 53, 123 and 297 and complete normally — the model
fails precisely on the steps that need a long answer.

### Root cause

The cap is declared locally, in `~/.config/opencode/opencode.json`:

```json
"limit": { "context": 256000, "output": 8192 },
"reasoning": true
```

**Not a provider limit.** `gateway` is a custom provider defined only in that
local config; it is absent from the models.dev catalog OpenCode caches at
`~/.cache/opencode/models.json`. Every catalogued provider serving GLM-5.3
declares `output: 131072` — 16× higher — and `context: 1000000`.

**Not an OpenCode default and not the harness.** The runner invokes
`opencode run --pure --format json -m gateway/glm-5.3` and passes no token
flag, so the config entry is the only possible source.

**`reasoning: true` explains the shape of the failure.** GLM-5.3 spends
reasoning tokens from the same output budget. With 8192 total, a hard
authoring or review step exhausts the budget while still thinking and has 0–7
tokens left to write with. That is why artifacts were never written despite
the model working for over two minutes (132.2 s and 129.0 s, both ending
empty).

### Options

1. **Correct the declaration to the model's real capability** (`output:
   131072`). One line, no new spend to apply, matches every catalogued
   provider. If the gateway deployment genuinely caps lower, the right value
   is whatever it enforces — but 8192 should be confirmed, not assumed.
2. **Raise it partially**, e.g. 32768. Lower risk if gateway billing is a
   concern, still ~4× the headroom that starved these runs.
3. **Leave it and split the work into smaller steps.** Already tried: a
   two-part review split still hit the ceiling at 8187 + 5. Step size is not
   the binding constraint. Not recommended.

### A second blocker, since resolved

This section used to report a 60-second cap on the review call, citing
`rc = p.wait(timeout=60)`, and warned that observed review steps of 132.2 s
and 129.0 s would exceed it even with tokens to spare.

That line no longer exists anywhere in the repository. The driver now uses
`WORKER_TIMEOUT_S = 900` (`integrations/cannbench/workers/driver.py:44`),
which is comfortably above the observed durations. The timing observation was
real; the code it described has been replaced. Recorded rather than deleted,
because a reader who met the old diagnosis elsewhere should be able to find
out what became of it.

### Which path the diagnosis was made against

Worth being precise, because the three entry points are not the same path.
The `opencode run --pure --format json -m gateway/glm-5.3` invocation quoted
above belongs to `integrations/cannbench/workers/driver.py:485-487`, the
CANNBench worker driver. Neither `tests/integration/test-kernel-generation.sh`
nor `tests/tools/collect_generative_evidence.py` passes `--pure` or `-m`.

The ceiling still reaches them, though, and by a different route: the global
config sets `model: gateway/glm-5.3` as the *default*, so any invocation that
does not name a model inherits both the model and its 8192 cap. That is why
the blocker generalises — not because all three share an invocation, but
because they share a default.

That shared default is now closed off. Both test entry points render a
tracked profile into the project through `tests/tools/opencode_profile.py`,
and `test-kernel-generation.sh` passes `-m` with the model named in it, so
neither inherits the host's global config. The CANNBench driver already
named its model explicitly and was never exposed by this route.

## Scope limits

The frozen add examples and a fresh add attempt cannot qualify all operators,
dtypes or shapes. The upstream compile-only suite is separate from generation
and numerical execution. Independent review, hardware and performance require
their own evidence. See [evidence-sources.md](evidence-sources.md).
