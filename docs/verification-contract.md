# Kernel verification and evidence contract

`run_and_verify.py --mode execute` configures the platform before importing the
kernel, runs it as `__main__`, and requires a counted real Launcher.run call.
Exit 0 means PASS, 1 FAIL, 2 SKIP. Auto falls back to JIT only on SKIP.
The legacy simulator mode is an alias for execute and uses the same contract.

JSON always includes status and verification_level. Failed/skipped results have
null level; launcher_calls retains calls observed before failure. Levels:

- compiled: codegen with a mocked launcher, no numerical claim.
- executed: real launches and clean completion.
- numerically_verified: executed, all observed real comparisons passed, and a
  suppressed-launch control demonstrably failed a supported comparison at the
  same source site after a launch attempt.

Supported observations are torch.allclose / numpy.allclose used directly in
`assert comparison(...)` or `if not comparison(...): raise ...` (single raise
body), numpy.testing.assert_allclose and torch.testing.assert_close. Other
styles remain executable but do not earn numerical verification. This is a
conservative observation of the script's checks, not an independent oracle or
protection against malicious scripts. It does not establish untested shapes.

The evidence field carries real/control reports and logs. A timeout, missing
report, missing dependency, signal, or unrelated exception in the control is
inconclusive and cannot raise the level. Real execution failure stays FAIL.
No-main and empty-main fixtures have distinct diagnostics. The unverified
fixture correctly returns PASS / executed, unlike the five failing fixtures.

## Acceptance entry points

`verification_contract.py` checks all three historical kernels, the template in
JIT and execute modes, six fixtures, auto failure, and complete skill JIT
listings. Incomplete and intentionally invalid listings have an explicit
inventory with reasons; new unexcluded listings enter the compile matrix.
Original run-2 is required to FAIL with UnsupportedSyntaxError and zero launches.

`generation_acceptance.py` records one fresh generation attempt. Its shell
entry point defaults to strict mode; `--static-only` is explicitly weaker.
Strict runtime qualification requires Docker, the pinned image and explicit
platform. A weak PASS (compiled/executed only), FAIL or SKIP cannot satisfy it.
Generated sources are copied and hashed before qualification, not repaired.
There are no automatic retries or commits.

## Storage

Provide a fresh `--results-dir` outside the checkout. Existing directories are
rejected. Each run has a manifest containing source revision, input hashes,
image digest where applicable, commands, timings and outcomes; logs and JSON
reports are siblings. A dirty revision is identified by the content hashes,
not falsely represented as identical to HEAD. Generated kernels and extracted
examples belong in this external run directory. Rendered credentials/configs
are excluded; raw environment variables are not archived.

Git holds harness code, permanent regression fixtures and concise conclusions.
GitHub Actions uploads run directories as artifacts, including on failure.
A local run uses the same schema outside the checkout. For accepted
qualification, retain a selected archive beyond ordinary Actions retention;
record its location and SHA256 in the qualification handoff and link the durable
archive in repository documentation when a durable shared location is available.
No historical evidence is moved or rewritten by this change.

Local checks do not prove that a new PR workflow ran. The dated PR run, tested
revision, full job duration and artifact digest are recorded in
[verification-status.md](verification-status.md). That run satisfied the
under-five-minute examples target; later revisions require their own checks.
