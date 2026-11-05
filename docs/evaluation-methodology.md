# Evaluation methodology

A result belongs to a complete generation and verification protocol: task,
prompt, loaded skills, model/configuration, budget, candidate, runtime and
evaluator. Use the [verification contract](verification-contract.md) to bind
these inputs to the exact measured output.

## Separate acceptance claims

- Static validity does not prove compilation.
- Compilation does not prove kernel launch or numerical correctness.
- Simulator numerics do not prove hardware behavior or speed.
- Hardware correctness plus timings qualifies the recorded cases, not every
  shape, operator or future model output.
- A service score and a baseline/candidate timing ratio are distinct quantities;
  report both with their definitions rather than relabel one as the other.

[CI gates](ci-gates.md) describe implemented checks;
[verification status](verification-status.md) records qualified scope.
The CANNBench lane evaluates one explicit recipe per run. It is not a current
multi-model dashboard or an automated skills-on/off experiment.

## Comparing interventions

To measure whether skills help, hold the task, model, runtime, budget and
acceptance checks fixed. Record the actual skills loaded in each session.
Turning off one discovery path is insufficient if global skills remain visible.
Reference candidates, extra hints and human repairs are separate interventions,
not part of an uncontaminated baseline. Retain failed and incomplete attempts;
never silently restrict the denominator to successful runs.

A new candidate or configuration needs its own qualification. A controlled
comparison also needs repeated observations and an explicit treatment of noise;
a single hardware smoke establishes neither a causal skills benefit nor a
reliable performance improvement.

## Evidence retention

Keep source/configuration hashes, native session provenance, candidate and
package bytes, task snapshots, per-case outcomes and timings outside checkout.
An accepted result needs a durable archive, verified checksum and concise link
in Git. Expiring CI artifacts alone are insufficient.

The retired dashboard schemas, campaign axes and metadata examples remain in
the [historical methodology](https://github.com/aloschilov/pyasc-skill-stack/blob/654a05556107c538dca11fc71b02fd1378543ded/docs/evaluation-methodology.md).
They are not requirements or implemented CI features of the current workflow.
