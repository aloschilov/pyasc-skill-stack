# Verification glossary

| Term | Meaning in this repository |
|---|---|
| Task contract | Public signature, shapes, dtypes, attributes, semantics and numerical tolerances of the operator being evaluated. |
| Candidate | The generated or supplied kernel under test; its hash identifies the exact bytes checked. It is not a correctness oracle. |
| Runtime pin | Exact source revision and artifact digest used for compilation or execution. Simulator and hardware packages may have different pins. |
| Operator recipe | A checked-in selection of task/function, generation prompt and numerical preflight for the shared CANNBench CI driver. |
| Compile-only | Kernel build with launch/numerical checks disabled; not an execution result. |
| Model / camodel backend | Simulator execution. Distinct from real-NPU execution. |
| Numerical verification | Comparisons against the task's trusted expected results, within a recorded case set and tolerance. |
| Suppressed-launch control | A check that the numerical test rejects a candidate when its kernel launch is suppressed. |
| Service score | CANNBench's reported aggregate metric, retained without treating it as a timing ratio. |
| Baseline/candidate ratio | Baseline elapsed time divided by candidate elapsed time for the same case; above 1 means the candidate is faster. |
| Fresh-session review | Review in a separate session/worker. It may use the same model; this does not establish independent-model review. |
| Passed / failed | The required check completed and met / violated its acceptance criteria. |
| Blocked / not-run | A prerequisite or execution obstacle prevented a verdict / the check was not attempted. Neither counts as passed. |

Use [evaluation methodology](evaluation-methodology.md) for interpretation and
[verification contract](verification-contract.md) for report requirements.
Kernel API terminology belongs to the pinned references and API skills.
The retired capability-cell enums and legacy kernel examples are preserved in
the [historical glossary](https://github.com/aloschilov/pyasc-skill-stack/blob/654a05556107c538dca11fc71b02fd1378543ded/docs/glossary.md).
