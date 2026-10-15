# Reproduction scope

`build_report.py` and `verify_publication.py` run with Python's standard library on the Mac. They perform no CANN execution and no network requests.

The AscendC export was run on the Ubuntu VM, from `/home/aloschilov/workspace/pyasc-skill-stack`, using the exact arguments in `../evidence/generation-command.json`. That command relies on the retained original submission qualification installation, its SDK mount and pinned Docker image; it is a provenance record, not a portable command for an arbitrary host. `../submission.zip` is the actual submission input, not the generated-code export.

`export_codegen.py` is the historical all20 qualification driver with additions that write the translated source, a stable specialization identifier and hash, and case-to-specialization links. It preserves the lowering pipeline and effective compiler options. `source_gate.py`, `source_gate_base.py`, `source_contract.py` and `helper-hashes.json` vendor its existing source checks; only import paths were adjusted to colocate them. Generated source is saved through `GELU_EXPORT_DIR`. No native C++ compile/link or numerical execution occurs in this driver.

To repeat on the retained VM:

1. Use a fresh export directory, the same installed submission wheel, SDK and image from the command record, and these reviewed scripts. Copy scoped inputs and verify hashes before execution.
2. Preserve the shared `gelu-adaptive-20260910/screen.lock` and `local.lock` execution locks. The export run acquired both with nonblocking exclusive `flock`.
3. Update the output mount, `GELU_EXPORT_DIR` and `--output` to the fresh directory; retain the qualification installation as read-only. Run via `bash scripts/cann-vm 'command'` from the primary Mac checkout.
4. Require20/20 dispatch and translation and six passed specializations. Compare compile options, constexprs, UB and source hashes. Do not overwrite the historical evidence or resubmit a job.

The first export attempt failed before importing the candidate because the copied source gate still pointed to a historical relative path. After vendoring its dependencies and correcting those import paths, the export passed. No candidate, runtime, JIT setting or compiler pass was changed.

The hardware result is independently retained in `../evidence/hardware-results.json`. The runner did not provide its full generated AscendC source through the retrieved job/log API; the exported C++ is explicitly VM-generated.
