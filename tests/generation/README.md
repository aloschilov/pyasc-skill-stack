# Generation experiment planning

This is the first implementation slice of [nightly coverage #22](https://github.com/aloschilov/pyasc-skill-stack/issues/22).
It creates an offline, bounded inventory before an executor or schedule is enabled:

```bash
python3 tests/tools/generation_plan.py \
  --results-dir /absolute/path/outside-checkout/generation-plan
```

The directory must be new. No credentials, model, Docker, simulator or CANNBench
connection is needed. The command writes a hash-bound `plan.json`; every cell
is `not-run`. It does not submit anything or enable a schedule.

`matrix.json` selects three small upstream case groups: vector addition with a
tail, reduction and broadcast. The planner verifies the pinned source inventory,
checks that the simulator and task revisions agree, and binds selected case ASTs,
skill sources, profile template and image digest. These case groups are task
anchors; static/dynamic pytest expansion and candidate-specific oracle/probe
adapters remain to be implemented. No numerical coverage is claimed.

The proposed per-cell attempt/round/response/time caps are checked against a
serial aggregate budget. Output-token bounds include all bounded responses
(including reasoning), but do not bound input tokens or provider prices.
Execution seconds exclude setup, queue and archival overhead. These are planned
ceilings, not observed usage or spending authorization; a future executor must
enforce them and record provider-effective settings. There is no executable
command in the matrix and no `--execute` switch.

Remaining work in #22: task-aware generation and numerical adapters; process,
usage and concurrency enforcement; interruption/partial-result handling;
compatible baseline comparisons; explicit budget approval and qualification of
manual and scheduled runs. Skills-effect comparisons (#23), reviewer handoffs
(#20), and model diversity (#24) should reuse these inputs and budgets.
