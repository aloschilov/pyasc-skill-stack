# Direct NPU correctness lane

Follow the [delivery roadmap](roadmap.md) to complete software review and
simulator checks before reserving scarce device time. Hardware acceptance
remains independent of those results.

The SSH backend delegates to the same host execution harness used by Docker.
It does not install packages or change a shared remote checkout. Configure an
already-authorized SSH alias and an interpreter with pyasc, CPU torch and CANN:

```bash
export PYASC_SSH_TARGET=cannlab
export PYASC_SSH_PYTHON=/path/to/venv/bin/python
export PYASC_SSH_ENV=/path/to/CANN/set_env.sh
export PYASC_SSH_REMOTE_DIR=/tmp/pyasc-hardware-runs
python3 tests/tools/hardware_acceptance.py --results-dir "$RUN_ROOT/hardware"
```

`RUN_ROOT` must be outside the checkout; choose a new directory for every attempt.
For one kernel:

```bash
python3 tests/tools/run_and_verify.py kernel.py --runtime-backend ssh \
  --backend NPU --mode execute --json
```

SSH options use the existing configuration/known_hosts and noninteractive
authentication. Host-key checks are not disabled. Each invocation creates a
unique remote subdirectory. Only Python sources from the kernel directory
(or recursively from an explicit `--project-dir`) and six harness files are
staged; hidden files, credentials, VCS data and results are excluded. Source
symlinks are rejected. Transfer uses rsync when installed at both ends, with
a staged-source tar fallback for bare CANNLab images. No `--delete` or remote
directory cleanup is performed. Payload hashes are verified before and after
execution and included with the report; both real and control evidence survive
delegation. Remote directories are retained for diagnosis.

Use a task-owned staging root in the workload container, for example
`PYASC_SSH_REMOTE_DIR=/work/pyasc-hardware-runs-<task-id>`, rather than sharing
the default root with other tasks. Review storage before each matrix and at
least weekly during active use; retain completed runs for at least seven days
and until their reports, logs and source hashes are archived and checked.
To list pruning candidates on an SSH endpoint **inside your own container**:

```bash
ssh "$PYASC_SSH_TARGET" \
  'find /work/pyasc-hardware-runs-<task-id> -mindepth 1 -maxdepth 1 -type d -name "run-*" -mtime +7 -print'
```

Replace the task placeholder with your recorded root. Age alone is not deletion
authority: for each candidate verify ownership, confirmed process completion,
successful external evidence retention and no unresolved invocation. Then remove
only the reviewed exact directory, for example:

```bash
ssh "$PYASC_SSH_TARGET" \
  'rm -r -- /work/pyasc-hardware-runs-<task-id>/run-<reviewed-uuid>'
```

Do not use a shared-root wildcard or automatic `find ... -exec rm`. Keep active,
unknown and unarchived runs; if space is insufficient, postpone dispatch.
For managed David, perform this maintenance through the agreed channel and
`docker exec` in the owned container, not direct Mac SSH or on the host. Cleanup
is an explicit operator action; neither backend nor matrix deletes evidence.

The device probe records the full `npu-smi info` table, models, native runtime
device count and SoC, toolkit install metadata, installed pyasc version and
interpreter. The SoC comes from `asc.lib.runtime.current_platform()` in NPU
mode, never a 9579/9599 default. An explicit mismatching platform fails before
launch. Missing identity/provisioning is SKIP; it cannot qualify hardware.

Invalid SSH target/directory or a backend other than NPU returns SKIP/2 before
contacting the device, with a configuration diagnostic and no kernel verdict.
Unreachable SSH, failed authentication and transfer unavailability also return
SKIP/2. Remote workload timeout, corrupt/missing report, unexpected execution
exit or changed source returns FAIL/1. A cooperative device-0 lock serializes
SSH workers, and their process group is terminated on timeout/interruption.
Check device ownership before dispatch; this lock does not control other tools.
The configured environment script runs in the same shell as the worker.

The matrix shares its expected outcomes with the simulator: historical run-1
and run-3 must be numerically verified; unchanged run-2 must fail compilation
before launch. It includes the tail-bearing template, five defective fixtures,
the unverified fixture (PASS/executed, not numerical), two positive spelling
regressions, legacy simulator dispatch and auto-failure. The old plan's “six
negative fixtures” was inaccurate: `unverified.py` is a deliberate weak PASS.
All 14 execution cases must match; no SKIP qualifies and no mismatching result
is silently repaired. Compile-only skill listings remain in simulator CI.

## Managed David route

The user's retained David session is reached through the managed Openterface
channel, not direct Mac SSH. Use its owner handoff and the current task's own container;
do not reconnect saved sessions or repair the channel as part of this harness.
All installations (including OpenCode), source staging/edits and runs belong
inside that container. Record its task owner, full ID, image and mounts; keep
shared CANN/driver/source inputs read-only and writable scratch task-specific.
The historical smoke container is an example, not an implicitly shared runtime.
Host commands are limited to readiness checks and container orchestration.
The David image permits checkout/fetch but cannot push. Return any scoped
changes and evidence with hashes to the Mac for review; publication happens
there only when authorized. Do not change credentials/network to enable push.
Follow the [hardware operating requirements](../skills/pyasc-build-run-verify/references/hardware.md#own-container-and-channel-ownership).
Once scoped, pinned source delivery has been qualified, the same matrix can
run **inside** that prepared environment:

```bash
python3 tests/tools/hardware_acceptance.py --runtime-backend host \
  --results-dir "$RUN_ROOT/hardware"
```

This mode still probes hardware and passes `--backend NPU --platform <detected>`
to every case. Transport delivery, target execution and result collection have
separate budgets and evidence. A previous smoke on another source revision is
not acceptance of this matrix. No result from an unavailable tool is invented:
if npu-smi/toolkit metadata are absent, provisioning remains blocked.

## Evidence and current limitation

Keep manifests, identity, results and generated inputs outside Git, using the
[shared storage contract](verification-contract.md#storage). Archive the chosen
accepted run durably and record its link/checksum in verification status.
Never edit historical `docs/evidence/`. Correctness does not establish latency,
throughput or a CANNBench baseline; performance remains not-run.

On 2026-10-30 the legacy CANNLab alias on the VM rejected authentication
(`Permission denied (publickey)`); the Mac has no such alias. The David owner
subsequently restored the retained channel and confirmed readiness at 11:21 MSK,
boundary 284/3667, with no pending/queued input. The prior recovery blocker is
resolved. Readiness does not transfer writer ownership: one writer is allowed
across the entire physical channel, including different PTYs.

Short batch commands and checked file returns are qualified within their recorded
sizes. The owned-container attempt below qualifies pinned source delivery and
OpenCode connectivity but does not qualify NPU execution. The live transport root
remains in `/private/tmp` with a validated durable snapshot, not continuous
backup. Transport repair/migration belongs to the channel owner. Preserve the
original invocation on UNKNOWN and reconcile it before new input. Backend
unit/transport checks are not hardware evidence.

## Owned-container attempt, 2026-10-30

An isolated task container was created from image
`sha256:ac58e6aa9482fed4082036a77cdd80e3f61c4292b37753ad0a81e3c57f08e5cd`,
with task-only writable scratch and read-only CANN/driver/runtime inputs. The
checkout is `f3c1af6cf4e69088030cd64d9e658785a683bfe9`. Node24.13.0 and
OpenCode1.2.19 were installed there; a tool-denied request to the authorized
`cannbot/GLM-5.3` provider returned exactly `READY`, exit0. This qualifies model
connectivity, not coding tools or NPU use. Credentials remain outside evidence.

At the user's direction, an external runner reused all 14 `core_cases()` and
their expected outcomes while omitting the availability probe. It passed
`--backend NPU --platform Ascend950PR_9599` explicitly and recorded device
identity as `not-probed`; it did not fabricate a ready identity or modify the
standard hardware acceptance gate. All 14 expected outcomes failed to match in
21.749 seconds, with no kernel launches. Thirteen calls returned SKIP/2 during
`set_platform`: `GetSocVersionWrapper returned 507008`. The auto case then
attempted its JIT fallback and failed importing `_base`. These results do not
establish any kernel's numerical correctness or performance.

The recorded runtime is PyASC `1.1.1+g6b3447b4`, Python3.10.14; CANN OPP
metadata is archived separately. The mounted runtime worktree's Git common
directory is outside the mounts, so its failed Git revision query is retained
instead of inventing a fresh full revision. The task checkout's revision and
all 140 input hashes were verified, as were 28 stdout/stderr hashes; inputs were
unchanged. Driver and other containers were not changed. Writer ownership was
returned after collecting the result.

External archive: `qualification-20261030.tgz`, 22,771 bytes, SHA-256
`2e62ace1d2271dbf328244d8c31e293295c298bc5c1c1d003824e3361c049614`.
It is retained outside checkout on David and at
`~/pyasc-qualification-runs/david-own-container-20261030/` on
the workstation, together with `verification.json` and transport receipts.
This is a failed execution attempt, not accepted hardware qualification.

The same pinned runner was retried after a confirmed WeLink reservation
(`Busy` at 23:46 MSK, `Free` at 23:47 MSK on 2026-10-30). All own workload
processes ended before release. The 14 cases again produced zero matched
outcomes and zero kernel launches in 21.780 seconds, with the same 507008
runtime setup error and the auto fallback import error. Human availability
therefore did not resolve this runtime failure; its cause remains unproven.
The external archive is `retry-20261030-2332/npu-direct-retry-20261030.tgz`
under the retained run root above, SHA-256
`b285e44ce1edd771b13ef7f2732ec08b594ef4159431d600951097788d8de775`.
Its 140 source hashes and 28 log hashes were verified, and inputs were unchanged.
This is a failed execution attempt, not accepted hardware qualification.


## Own-container generation and runtime diagnosis, 2026-10-31

The same pinned skill-stack checkout produced a new float32 vector-add kernel
inside the task container using OpenCode 1.2.19 and `cannbot/GLM-5.3`. The first
agentic attempt timed out without a candidate. A second, bounded request with
recorded skill excerpts completed in 80 s. The unchanged generated kernel passed
syntax/static checks, JIT compilation in 16 s, and CaModel verification in 105 s:
four real launches for sizes 1, 1024, 1025, 4097 and a suppressed-launch control
that failed at the numerical assertion. This is a scoped manual generation
result, not qualification of the full agentic integration suite or NPU lane.

The original GeLU pytest also passed compile-only in 9.49 s. Its NPU execution
failed before launch, including after removing incompatible `torch-npu` from
our writable container layer and restarting only our container. The generated
kernel likewise returned SKIP/2 on NPU, with zero launches and 507008. Runtime
logs record driver code 87; recent kernel errors explicitly report
`uda_occupy_dev_by_ns: Conflict open udevid` for device 0. The holder is not
identified by the masked namespace fields. Human WeLink availability does not
resolve this device-allocation conflict. Driver and foreign containers remain
unchanged; final `Free` was confirmed at 00:30 MSK after all own workloads ended.

Runtime provenance remains PyASC 1.1.1+g6b3447b4 with source-file hashes, not an
independently verified clean full Git revision. The selected simulator directory
is Ascend950PR_9599, while its bundled stub reports Ascend950PR_9589; these
results identify the installed simulator route, not hardware identity.
Performance and accepted NPU qualification remain unestablished.

The complete external report and receipts are retained at
`~/pyasc-qualification-runs/david-container-repair-20261030/`.

Complete external bundle: `~/pyasc-qualification-runs/david-container-qualification-20261031.tar.gz`,
SHA-256 `5073e3e620a223f157b5f6d15f0682f15661d192660e93b822b227589bbccc15`. The verified container-only archive is also retained
under `/work/evidence` on David. This is accepted evidence for the scoped
CaModel result, not accepted NPU qualification.
