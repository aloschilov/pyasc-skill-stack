# CANNBench validation helpers

The supported entrypoint is the [one-operator workflow](../README.md).
These helpers do not run a full-catalog campaign or manage model sessions:

- `contracts.py`: official callable signatures, argument construction, and
  static candidate checks. Static checks are heuristics, not a sandbox or a
  substitute for the service's anti-cheat checks.
- `local_compile_gate.py`: metadata dispatch and lowering for the selected
  operator's case inventory. `../ci/single_qualify.py` attests the runtime
  against the selected wheel before calling it. Direct use has no attested
  PyASC revision and proves neither numerical correctness nor NPU execution.
- `run_camodel_smoke.py`: optional representative numerical tests for selected
  operators in an already provisioned simulator. Pass `--ops` explicitly;
  these small routes are not the complete official case set. The CI recipe's
  numerical probe remains its own acceptance gate.
- `evalqueue.py`: private streaming upload adapter used by `../smoke.py`.
  Durable intent, reconciliation and credit checks belong to that caller;
  uploads are never automatically retried.

Keep run outputs outside checkout. Task contracts in `../tasks/` are reusable
specifications, not qualification results; the CI path snapshots the live task.
Design/review/tuning remain in the maintained skills. Automated fresh-session
handoffs and bounded tuning are tracked in
[issue #20](https://github.com/aloschilov/pyasc-skill-stack/issues/20).
