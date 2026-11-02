# Hardware targets and measurement

Initial target: **Ascend 950 "David"**, **CANN 9.x.x**. Record the exact tested
versions (CANN, PyAsc, driver, SoC/board) and the model used. Never infer
universal compatibility from one environment.

Hardware measurement requires an authorized device run. Until one is performed
and recorded, the hardware gate is **blocked**, not pending-by-omission. Do not
POST hardware results automatically; obtain explicit authorization. Report
measured latency/throughput only when measured; keep queue/compile time separate
from steady-state execution where possible.

## Own container and channel ownership

For new work on shared remote Linux, install tools (including OpenCode), stage
and edit sources, and run kernels strictly in the current task's own container.
Record the task owner, full container ID, image ID/digest, mounts and task-only
writable scratch. An available SSH account or idle container is not ownership.
Preserve other tasks' and historical smoke containers. Mount only required
shared inputs read-only; do not expose the Docker socket or unrelated homes.
Host commands are limited to scoped readiness checks and container orchestration.

For David through Openterface, use the managed-channel skill and obtain fresh
writer handoff from the existing channel owner before input. Device reservation
is separate from channel ownership. Fetch pinned payloads and verify hashes in
the task container; use short managed commands for execution and result return.
Source the CANN environment in the same `docker exec` process as the workload.
Keep invocation/result records outside the source checkout. An ambiguous
completion must be reconciled under its original ID before further execution.

The David image permits checkout/fetch but cannot push changes. Keep edits and
results in task-local storage, return scoped patches/new files and evidence with
hashes to the Mac, and review there. Publication requires separate authorization;
do not alter the image's credentials or network to enable push.

## David reservation through WeLink

Before using David, read the live device group's reservation/release history
and queue. Arrange the transport writer handoff separately, recheck that the
device is free and that no queued user has priority, then send `busy` (or `занял`)
and confirm delivery before starting NPU work. The local managed-channel skill
records the group identity and access route. An old screenshot, an idle container
or a successful smoke test is not a reservation.

After all your NPU processes have ended, including failures, send
`free` (or `отпустил`) in that same group and verify delivery. Release your reservation also
if cancelled before launch. Unknown process/message state requires reconciliation
of the original operation; do not launch a duplicate or claim release prematurely.
Keep reservation, attempt and release state with external run evidence. When
the user requests a single retry after waiting, stop monitoring once that attempt,
evidence collection and release finish. Monitoring and chat messages require
authorization for the current task; the protocol alone does not grant it.

## Simulator is not hardware

`Backend.Model` (`Ascend950PR_9599` camodel) is cycle-accurate but is still a
simulator. A camodel result may be reported as a numerical pass. It may **never**
be reported as a hardware performance measurement. The two are separate gates.

## Performance baselines

A latency number alone is not a result. Performance is expressed as a ratio to
the CANN baseline, and that baseline comes from CANNBench plus the TTK manifest
— see [`../../pyasc-cannbench-kernel/references/ttk.md`](../../pyasc-cannbench-kernel/references/ttk.md).
Do not invent a baseline, and do not compare against a previous run of your own
kernel and call the delta a CANN comparison.
