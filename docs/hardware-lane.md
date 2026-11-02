# External hardware smoke

PR #19 uses one CANNBench SwiGlu evaluation as its minimal hardware check.
The SSH runtime and direct-device matrix are deferred. Use the
[submission procedure](../integrations/cannbench/README.md).

## Current qualification

**Passed — 20/20 correct and measured cases**, 2026-11-02, for the reference
candidate at `38951c4661fe165d189b8b00ef063223da2c84f0`.
[job_cb166531ba8d](https://cannbench.com/workspace/jobs/job_cb166531ba8d)
(`sub_ec203b5850b6`) evaluated official-tasks 2.0.0 / SwiGlu on the shared
Ascend950PR_9589 runner with CANN 9.2.0, Python 3.12.13 and torch_npu 2.10.0.
The strict gate verified the exact case set, successful numerical checks,
zero reported anti-cheat failures and positive finite timings for every case.
Profiler records identify `_swiglu_2d_kernel` and `_swiglu_short_rows`.

Queue time was **506.17 s**; execution from `claimed_at` to `finished_at` was
**191.24 s**. The unweighted geometric mean of the 20 baseline/candidate timing
ratios is **0.317322x**: the candidate is slower than the baseline. This ratio
is computed directly from per-case timings, not the service's aggregate score.
Cases 9 and 12 now pass (38.68 us and 447.92 us respectively). This qualifies
one repaired reference kernel and its packaged runtime; it does not establish
fresh model generation, acceleration, other operators or the complete simulator
verification contract on hardware.

The durable archive is retained on the workstation and submitting VM at
`~/pyasc-qualification-runs/pr19-cannbench-20261102/attempt4-qualification.tar.gz`.
SHA256: `ef96150798f3d506a67ed38920f4b971c7a44c26cb4a8bda0be1b28de7732098`.
All 614 member hashes were checked after return; 581 archived source files
matched the Git revision above. The archive contains task/catalog snapshots,
manifest, candidate, package, raw job/result/logs, checksum-verified service
artifacts and local repair checks. Prepared inputs and the unchanged submitted
candidate were verified. Runtime binaries and run artifacts remain outside Git.

Exact binary inputs:

- Runtime bundle SHA256: `357f3d10961b2ad1878537b28563f6889931c1fd6b7a01fea32e150d9cd95333`.
- Submitted ZIP SHA256: `4a5f933003b555cf15cd6c34d2d7a484f619d9cb6a697975daadab30ced48cb1`.
- Built wheel SHA256: `258086c260be779788a673aff33a72dfa19d2f555e51a69aa392f6c8fa179ec6`.

## Previous CANNBench attempt

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
cover three dtypes, partial rows, multiple cores and NaN/Inf propagation.
The accepted NPU run above confirms the repaired candidate passes both cases;
it does not isolate the original NaN mismatch's cause.

## Historical attempts

Earlier direct-device attempts failed before kernel launch and do not qualify
hardware correctness. Their evidence is preserved; see the
[historical record](https://github.com/aloschilov/pyasc-skill-stack/blob/e154f4218554d6ff8119f2886de2632701b21d58/docs/hardware-lane.md).
Operational access procedures belong to local agent instructions, not this
product's installation or execution contract.
