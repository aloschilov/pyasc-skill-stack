# Minimal delivery roadmap

Updated 2026-11-02. The primary workflow is OpenCode plus PyASC skills in the
execution environment, using a full checkout or skills installed from it.
Generation needs one available model; hardware evaluation need not involve
remote-shell access. The archived multi-model campaign is not a prerequisite.

1. **Stabilize the existing workflow.** Keep installer safety, pinned simulator
   verification, numerical controls and regression tests. Require real `pr-gate`
   and `examples-gate` success on the submitted head. No SSH runtime backend.
2. **Check one operator on hardware through CANNBench.** Prepare a self-contained
   SwiGlu package, snapshot the live task/version, submit explicitly and retain
   the exact package and job identity outside checkout. Accept only complete
   official-case correctness, zero anti-cheat failures and final hardware
   timings. See [the smoke procedure](../integrations/cannbench/README.md).
   A reference-kernel smoke is not evidence of fresh model generation.
3. **Then test one generated candidate.** Use one configured OpenCode model and
   the current task contract; package its candidate with a compatible runtime.
   Compare the service's correctness and timing results with the reference.
   Performance hints guide changes; measured results determine whether they help.

The hardware smoke is manual and quota-consuming, separate from ordinary CI.
Inspect remaining credits (the working budget is 10/day); do not schedule an
automatic campaign. Reconcile ambiguous submissions by GET before further work.
No broad operator sweep, nightly, standalone packaging redesign or remote
execution backend is needed to close this first loop.

Store complete manifests, packages, logs and responses outside checkout. Keep
only checks, small permanent fixtures and a concise qualification link/checksum
in Git. Preserve historical evidence. Simulator and hardware scopes remain
separate; a passed smoke does not qualify every generated kernel or runtime.
