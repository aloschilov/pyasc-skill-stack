# Delivery roadmap with intermittent NPU access

Updated 2026-11-02. This replaces the hardware-first ordering of the original
hardware-lane-and-cleanup plan, not its hardware acceptance criteria.
David availability is an external dependency. Software correctness and cleanup
can be reviewed and shipped independently; simulator evidence never closes the
hardware gate. Publication and merge remain separate operator decisions.

## 1. Software changes and review — no NPU required

Scope: SSH delegation/report handling, shared simulator/NPU outcome policy,
OpenCode skill access, regression coverage, audited cleanup and documentation.
The close-generation-loop baseline is merged in PR #18; PR #19 contains the
hardware backend and cleanup. Its five review findings are addressed by:

- Frozen per-check outcomes and failure categories, plus eight restored legacy
  detector tests; check names alone are insufficient.
- Consistent SKIP/2 for invalid SSH configuration before any device contact.
- A distinct missing-report diagnostic before JSON parsing.
- One outcome matcher for both matrices, including optional launcher counts
  and rejection of `auto` falling back to JIT instead of execution.
- Ownership-aware retention/pruning guidance with no automatic deletion.

Acceptance: the current PR head passes real `pr-gate` and `examples-gate`;
the simulator manifest matches all expected outcomes with unchanged inputs.
Frozen evidence and protected detector code remain unchanged. Review replies
identify the implementing commit and validation; only then resolve findings.
Queued, missing or skipped checks are not success. NPU unavailability does not
block these software changes, but must remain explicit in the PR description.

## 2. Prepare and qualify candidates locally — no David reservation

Use a new external run root for each attempt:

```bash
export RUN_ROOT="$HOME/pyasc-qualification-runs/local-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_ROOT"
PYASC_PYTHON=python3 bash tests/ci-gate.sh --tier pr
python3 tests/tools/verification_contract.py --results-dir "$RUN_ROOT/contract"
python3 tests/tools/mutation_acceptance.py --results-dir "$RUN_ROOT/mutations"
```

The latter two commands need local Docker and the pinned simulator image, not
an NPU. The contract includes positive kernels, deliberate failures, a weak
execution-only PASS and compile-only skill listings. Both mutation regressions
must be detected. A backend unit mock does not substitute for this execution.

For changes to generation or skills, run one bounded fresh generation when the
configured provider is available:

```bash
PYASC_PYTHON=python3 python3 tests/tools/generation_acceptance.py \
  --results-dir "$RUN_ROOT/generation"
```

Accept only static/JIT/numerical verification of the same unchanged artifact,
including the suppressed-launch control. Retain failed attempts too. Missing
provider credentials block this step only; never weaken the scoring threshold
or replace numerical verification with a model's self-report. Reuse an already
qualified candidate for the next device window instead of generating on NPU
time. Pin its kernel/harness hashes, source revision, runtime/image identity,
commands, expected outcomes and time budget before booking the device.
The pinned wheel and David source build expose different module paths; qualify
against each intended runtime. If adaptation is needed, preserve the original
failed result and qualify a separately hashed derivative rather than silently
repairing a supposedly unchanged generated artifact.

## 3. Short hardware window — still open

The current [hardware evidence](hardware-lane.md#own-container-generation-and-runtime-diagnosis-2026-10-31)
is blocked before launch by a device-open namespace conflict. Human vacancy
alone did not remove it. On 2026-11-02, a new live WeLink check found an active
reservation, so no new NPU attempt was dispatched. These are dated observations,
not a current availability signal.

1. Read fresh WeLink history and its queue. Obtain the separate current transport
   writer handoff and verify the task container identity before device work.
   Do not reserve David while waiting for transport, installation or generation.
2. Recheck the queue, send `busy`, and confirm delivery and absence of a conflict.
   Run only in the owned container, with a new evidence directory and bounded
   commands. Do not add availability/DSMI probes to the requested actual-test
   retry. Do not change drivers or foreign containers.
3. First retry the existing original PyASC test and simulator-qualified generated
   kernel. If setup still fails before launch, retain the actual failure and
   stop that window; ask the resource owner to restore the container's device
   allocation. Do not repeat the full matrix against a known setup failure.
4. If real launches work, run the 14-case hardware contract in the prepared
   environment. Formal qualification still needs recorded device identity and
   the detected SoC, not a guessed platform. A diagnostic run with unprobed
   identity cannot pass that gate. See [commands and contract](hardware-lane.md).
5. Confirm all own NPU processes ended, send `free`, verify delivery, return
   transport ownership and verify the collected archive. Preserve UNKNOWN
   command identities for reconciliation; never replay them automatically.

Acceptance: all 14 expectations match, including unchanged run-2's compilation
failure and the unverified fixture's execution-only level; no SKIP qualifies.
The numerical positives need real launches and falsifying controls. Retain
manifest, separate logs, timings, identity and source hashes outside checkout;
archive accepted evidence durably and put only its summary/link/checksum in Git.
Only then change `hardware` to `passed`. A requested availability monitor is
quiet while unchanged and stops after its one completed retry and release;
this roadmap does not create or restart a monitor.

## 4. Later, separately accepted work

Independent review of a generated candidate and measured NPU performance are
distinct deliverables. A scoped source review or CaModel result does not close
the whole independent-review gate. Performance/CANNBench remains `not-run`;
restore its missing integration inputs selectively only when that lane resumes.
Do not schedule performance experiments until hardware correctness is accepted.

No nightly generation run is currently established. CI has no generation
schedule or provider credentials; adding one requires a separate provider,
budget, failure notification and external retention design. Until then use
explicit bounded generation runs and report their actual coverage.
