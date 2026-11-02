# External hardware smoke

PR #19 uses one CANNBench SwiGlu evaluation as its minimal hardware check.
The SSH runtime and direct-device matrix are deferred. Use the
[submission procedure](../integrations/cannbench/README.md).

## Current qualification

**Failed — 18/20 cases passed**, 2026-11-02. The reference SwiGlu candidate
in [job_f378f3c6a5f5](https://cannbench.com/workspace/jobs/job_f378f3c6a5f5)
ran on Ascend950PR_9589 / CANN 9.2.0 against official-tasks 2.0.0.
All 18 passing cases have PyASC device-kernel timings; the service reports zero
anti-cheat failures. Two cases failed: case 9 required an unavailable runner
`Slice` binary during a framework copy; case 12 had a NaN-position mismatch.
Queue time was 6818.75 s; execution from `claimed_at` to `finished_at` was
251.76 s. There is no complete performance score accepted by the strict gate.

The failed run, inputs, source, service logs and reports are retained on the
workstation and submitting VM at
`~/pyasc-qualification-runs/pr19-cannbench-20261102/attempt3-result.tar.gz`.
SHA256: `fc2efdb463efcc955802650c7dce3996b2f10dd03a1b181423a42f651d74ce69`.
The archive's 608 member hashes and prepared input hashes were verified.
Submitted ZIP SHA256:
`2ebf01e8b876d39de9b210314bfe91b9cfaeb959530c32bfc990cc0f354f2ad4`.

The replacement short-row path loads directly through PyASC, without the
framework slicing fallback used by both failing cases. Simulator regressions
cover three dtypes, partial rows, multiple cores and NaN/Inf propagation; they
do not establish the cause or resolution of the NPU NaN mismatch. A new full
CANNBench result is required. This remains a reference-kernel smoke, not
model-generated qualification or proof of acceleration.

## Historical attempts

Earlier direct-device attempts failed before kernel launch and do not qualify
hardware correctness. Their evidence is preserved; see the
[historical record](https://github.com/aloschilov/pyasc-skill-stack/blob/e154f4218554d6ff8119f2886de2632701b21d58/docs/hardware-lane.md).
Operational access procedures belong to local agent instructions, not this
product's installation or execution contract.
