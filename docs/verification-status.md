# Verification status

Gate vocabulary: `passed | failed | blocked | not-run`. A missing dependency
cannot count as successful acceptance. Historical evidence remains unchanged.

## Generation and harness qualification

The close-generation-loop implementation has passed local qualification and
real PR checks in [PR #18](https://github.com/aloschilov/pyasc-skill-stack/pull/18).
It is not merged. The evidence below identifies the tested revision explicitly.

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
| hardware | not-run | outside this change |
| performance | not-run | simulator checks are not performance evidence |

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
