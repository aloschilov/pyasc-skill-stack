# Direct NPU correctness lane

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

The device probe records the full `npu-smi info` table, models, native runtime
device count and SoC, toolkit install metadata, installed pyasc version and
interpreter. The SoC comes from `asc.lib.runtime.current_platform()` in NPU
mode, never a 9579/9599 default. An explicit mismatching platform fails before
launch. Missing identity/provisioning is SKIP; it cannot qualify hardware.

Unreachable SSH, failed authentication and transfer unavailability return
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
channel, not direct Mac SSH. Use its owner handoff and existing container;
do not reconnect saved sessions or repair the channel as part of this harness.
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
(`Permission denied (publickey)`); the Mac has no such alias. The alternative
David channel owner's fresh handoff reported stale optical evidence and missing
recovery journals/client state, and prohibited attach until recovery. No new
NPU command was sent. Hardware qualification therefore remains not-run, blocked
on access; the backend's unit/transport checks are not hardware evidence.
