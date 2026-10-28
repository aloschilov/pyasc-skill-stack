# Hardware targets and measurement

Initial target: **Ascend 950 "David"**, **CANN 9.x.x**. Record the exact tested
versions (CANN, PyAsc, driver, SoC/board) and the model used. Never infer
universal compatibility from one environment.

Hardware measurement requires an authorized device run. Until one is performed
and recorded, the hardware gate is **blocked**, not pending-by-omission. Do not
POST hardware results automatically; obtain explicit authorization. Report
measured latency/throughput only when measured; keep queue/compile time separate
from steady-state execution where possible.

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
