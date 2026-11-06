# Fresh-session review handoffs

This is the writer/reviewer/verifier slice of [#20](https://github.com/aloschilov/pyasc-skill-stack/issues/20).
It has offline transition/receipt tests and the manual qualification below.
The selected task is one existing add adapter. This does not qualify arbitrary
operators, hardware performance, a second model, or an autonomous tuning campaign.

## Manual qualification, 2026-11-06

[Run 42183443085](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/42183443085)
on `9d2324f45a4f921b0c09ab3fd3788e2afdc67fa3` accepted one freshly generated
float32 add candidate after a separate reviewer session and unchanged-byte
simulator qualification. Both sessions used GLM-5.3 (`glm-5-3-260814`) through
OpenCode 1.19.0; this is session independence, not independent-model review.
Plan SHA256: `53a54393df5bba601c6105201741c821a060c08c94eeed6e7761eba76ce51f85`.

The reviewer accepted without findings. The verifier then made **three real Model
launches**, passed **nine output/input checks** on seeds 0, 1 and 7, and rejected
the suppressed-launch control. All 99 source hashes, full skill/handoff reads,
native session IDs, task/design/candidate bindings and provider usage were rechecked
from the public archive. Candidate SHA256:
`e92c8e40517e4a9094f1dd8656ce6944de0b974a0367bfdcdcf811124f3894b6`.

Ten requests used **4,821 output** and **95,173 input tokens**; monetary cost is
unknown. Executor wall time was **151.70 s**, including 137.21 s for both model
sessions and 13.27 s for qualification. Preceding queue/setup took 65.04 s;
the qualification CI job took 206 s. A post-run private-payload hash audit found
high reasoning and enabled thinking in both roles, with caps of 24,000 output
tokens for the writer and 8,192 for the reviewer. Other sampling defaults and
immutable provider weights revision remain unknown; provider compliance is not
asserted. Recorded processes and the simulator container were absent after the
run, and the executor released the shared latch.

The [durable Release](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-review-42183443085)
retains `qualification-review-42183443085.tar.gz`, SHA256
`f0801b055dfcebf9bf1c5f2011e6255fc4e9128c7058f912e1db1bcb37febfa1`.
It contains the original public CI archive and verification/cleanup receipts;
private provider payloads are excluded. Copies also remain outside checkout on
the workstation and VM. This run required no repair and no tuner: their transition
rules have offline tests, but no live repair or tuning improvement is claimed.
There was no NPU call, schedule activation or automatic promotion.

## Transitions and acceptance

The writer loads pinned codegen skills and writes `candidate.py` and `design.md`.
The design describes the task ABI, shapes, memory ranks, tails, precision and
verification scope. These bytes, the task contract and their hashes form the
immutable handoff. The writer cannot read the oracle or a reference candidate.
The version 2 plan supplies exact project-relative `read` arguments for each
handoff file. This follows the observed absolute-path reconstruction failure in
the paired experiment; it does not relax full-read attribution or permissions.
Version 1 receipts retain their original absolute prompt format.

A new reviewer session loads the review/build/API skills and reads the complete
task, candidate, design and binding. It can write only `review.json`; candidate
and design remain read-only. Findings include a stable ID, severity, location,
problem, proposed repair and disposition. An acceptance cannot contain an open
blocking finding. A rejection must identify one. A verdict with no findings must
still describe the scope and residual risks. Both roles may use the same configured
model: a fresh session is not an independent model.

Only an accepted review reaches the existing simulator verifier. Acceptance
requires static validity, actual JIT/execution launches, numerical checks against
the unchanged oracle and a rejecting suppressed-launch control on the same bytes.
A review is not numerical proof, and simulator time is not device performance.

Rejection or a genuine candidate verification failure can use the one declared
repair. The next writer receives the previous candidate/design and concrete
review/verifier feedback. An unchanged repair stops. Every changed candidate
needs another fresh reviewer and another qualification; previous finding IDs need
explicit dispositions. An open prior blocker cannot disappear through a severity
downgrade. Unknown requests, process interruption, changed inputs or
invalid attribution stop execution and keep the shared unresolved-run latch.
Existing runs are never replayed automatically.

Each role has an isolated project, HOME/XDG directories, native session and
provider gate. Native accesses and actual provider context must establish full
handoff/skill reads. Directory search, commands and networking are denied. The
public receipt rechecks plans, source/candidate/design hashes, sessions, model,
review lineage, oracle, numerical/control evidence and budget reservations without
executing archived code. Private raw requests/responses and native state are
excluded from the public archive.

The default ceiling is four model sessions, 772,608 output tokens including
reasoning, and 3,000 seconds of reserved execution. Input tokens, money, setup and
queue time are reported or scoped separately. A known budget rejection cannot
produce an accepted candidate. There is no model fallback or second upload ledger.
Each role allows at most 12 provider requests, including auxiliary calls. This
leaves room for sequential skill/handoff reads without requiring model-specific
tool batching; the reviewer uses at most 8,192 output tokens per response.

## Explicit use

```bash
python3 tests/tools/generation_review.py plan --root /outside-checkout/review-plan
python3 tests/tools/generation_review.py execute \
  --plan /outside-checkout/review-plan/plan.json --approve-plan <exact-digest> \
  --root /outside-checkout/new-review-run --state-dir <existing-shared-generation-state>
python3 tests/tools/generation_review.py report --root /outside-checkout/review-run
```

The `Reviewed generation` workflow plans on a hosted runner without provider
credentials; PR planning does not occupy the shared model executor. Manual `mode=execute` requires
the exact reviewed digest and reuses the existing simulator worker, shared lock,
provider gate and external evidence archive. It has no cron, hardware submission,
or automatic merge. Existing `tests/run-tests.sh --agentic` and manual skill
workflows remain separate and usable; their output does not acquire this receipt
contract merely by using the same skills.

## Optional bounded tuner

`generation_tuning.py` freezes a numerically accepted reviewed incumbent and a
caller's external device-measurement receipt. It permits one tuner proposal with
the same bounded review/repair cycle, using the codegen/API performance guidance.
The new design must state the optimization hypothesis and tradeoff. Candidate,
design, measurement and budget are immutable inputs read by the fresh tuner
session. Measurement and budget share `tuning-context.json`, so sequential reads,
two output writes and the final completion fit the same 12-request ceiling.
A repair cannot silently revert to the incumbent as a new optimization.

An accepted simulator proposal leaves the incumbent selected and enters
`awaiting-measurement`. The caller separately obtains an explicitly authorized
measurement; this helper executes no measurement command, NPU call, CANNBench
submission or reconciliation. `select` consumes at most one supplied receipt.
Missing/invalid/incomparable measurements or insufficient improvement keep the
incumbent. Concurrent transitions and replay of an interrupted proposal are
rejected; model calls still use the shared executor lease, not another ledger.

The receipt is a strict `caller-device-measurement` JSON object, with:

- `schema_version: 1`, `backend: NPU`, exact candidate/task SHA256 and `pyasc_revision`;
- `hardware`, `method`, UTC `measured_at`, durable HTTPS `source_url` and `source_sha256`;
- one case per adapter seed (`seed-0`, `seed-1`, `seed-7`), each with positive finite
  `elapsed_us`, integer `samples >= 2`, `numerically_correct: true`, and nonempty
  `device_kernels`.

The source URL must have no embedded credential or query. Hardware, method,
runtime, task, case inventory and sample counts must match the incumbent. The
unweighted geometric mean incumbent/proposal timing ratio must exceed the fixed
1.01 threshold. This is an observed comparison, not a statistical confidence or
general speedup claim. Raw simulator wall time is rejected.

**Trust boundary:** a trusted caller supplies this redacted measurement receipt
outside the model/handoff tree. The helper validates its schema, scope and byte
continuity; it cannot independently authenticate a remote measurement. The
decision therefore retains `measurement_trust: caller-supplied` and never claims
new hardware qualification. Use the existing strict CANNBench collection or an
equivalent audited device measurer to substantiate the cited source. No CANNBench
adapter for the three seed-based simulator tasks is implied by this receipt.

```bash
# File-only planning from an already accepted reviewed run and its device receipt.
python3 tests/tools/generation_tuning.py plan --root /outside-checkout/tuning \
  --incumbent /outside-checkout/reviewed-incumbent --measurement /outside-checkout/incumbent-device.json
# One explicit model proposal, fresh review and simulator qualification; no NPU.
python3 tests/tools/generation_tuning.py propose --root /outside-checkout/tuning \
  --approve-plan <exact-digest> --state-dir <existing-shared-generation-state>
# After external measurement, file-only selection. Omit --measurement to retain the incumbent.
python3 tests/tools/generation_tuning.py select --root /outside-checkout/tuning \
  --measurement /outside-checkout/proposed-device.json
# Recheck selected bytes, proposal lineage and timing ratio from retained evidence.
python3 tests/tools/generation_tuning.py report --root /outside-checkout/tuning
```

All incumbent evidence, transitions and decisions stay outside checkout. This
optional path has synthetic transition tests; no live tuning improvement or
hardware qualification is claimed. Ordinary manual skill and `--agentic` flows
remain available; they do not gain this evidence contract automatically.
