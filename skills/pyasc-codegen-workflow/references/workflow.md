# AscTile request workflow (agent-driven)

The agent is the interface. There is no required CLI.

1. **Clarify the contract first.** If essential fields are missing (operator,
   input dtypes, shapes, output, tolerance), ask the user. Do not infer the
   operator from files left in the project. A question gets an answer, not code.
2. **Agree the budget (optimization only).** Confirm the supplied baseline and
   time/attempt budget, or ask for the missing ones, then work within it.
   Negotiating is not measuring.
3. **Preflight the artifact.** Record the interpreter, import path, `asc` version
   and `jit`/`global_tensor`. On failure, stop and report the diagnostic; do not
   scan private environment directories, change permissions, or rebuild unless
   the user asks. A supplied unavailable-runtime fact does not require probing.
   The preflight block lives in
   [`../../pyasc-build-run-verify/references/verification-patterns.md`](../../pyasc-build-run-verify/references/verification-patterns.md).
4. **Write the kernel and a task-specific test.** Use a seeded torch reference for
   the exact operator contract.
5. **Compile, then execute numerically.** Compile through the PyAsc codegen and
   execute through the documented path; compare within the operator tolerance.
6. **Bounded optimization.** Within the agreed budget and baseline, keep the best
   numerically correct candidate; never trade correctness for speed; record
   latency only when measured.
7. **Deliver** code plus a local report
   ([`../../pyasc-build-run-verify/references/report-template.md`](../../pyasc-build-run-verify/references/report-template.md)).

## Gate taxonomy

Statuses are `passed | failed | blocked | not-run`. Missing runtime, provider or
hardware is an honest `blocked` gate, never a fabricated pass. Generation,
compile, numerical and hardware are **separate** gates: a compile success is not
a numerical pass, and a simulator result is not a hardware measurement.

## What counts as a reference

Correctness is judged against the Huawei-authored asctile target suite in
[`reference/asctile-target/`](../../../reference/asctile-target/), performance
against CANNBench. A kernel written by a model is a candidate, never a
reference. See [`docs/evidence-sources.md`](../../../docs/evidence-sources.md).
