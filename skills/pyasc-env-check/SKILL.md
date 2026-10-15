---
name: pyasc-env-check
description: Verify a current-v2 pyasc source/runtime/toolkit environment, including exact commit provenance, Python ABI, CANN, compiler, camodel, torch_npu, and NPU availability.
---

# Current pyasc v2 environment check

`compiler-team/pyasc` branch `v2` is not available from the Python package
repositories used by this project. Do not use `pip install pyasc` to stand in
for the selected source. Build or use a vendored wheel from the exact recorded
commit.

## Required checks

- selected source remote/branch/commit
- wheel SHA-256, architecture, and CPython ABI
- `import asc` and `import asctile`; `import asc2` must not be required
- Python version supported by that wheel
- CANN toolkit root/version and Bisheng compiler path
- target platform `Ascend950PR_9599`
- simulator library path for camodel
- torch and torch_npu versions for real tensor launches
- NPU presence and account access when running hardware

For the current campaign the expected source commit is
`0a631f70968c3cb7c33ce45330a85768dd5a6f06` with a matching runtime. Verify the actual wheel ABI and architecture; historical
four-operator wheels belong to their recorded pin and are not substitutes.

## Quick commands

```bash
git -C <pyasc-repo> fetch origin v2
git -C <pyasc-repo> rev-parse origin/v2
python3 -c "import asc, asctile; print(asc.__file__, asctile.__file__)"
sha256sum <wheel>
python3 --version
test -x "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng"
npu-smi info
```

The import command only proves which installed runtime is visible. Compare it
to the recorded wheel/source before using its output as campaign evidence.

## Profiles

- **QEMU/x86_64 compile gate:** dispatch and lowering only; no numerical or
  performance claim.
- **Native aarch64 camodel:** numerical execution; build the same commit using
  local LLVM when required.
- **NPU/CANNBench:** authoritative benchmark execution and timing.

If source and runtime commits differ, stop validation and rebuild the runtime.
If camodel is unavailable, continue with compile evidence but label numerical
claims unverified. Never silently substitute another platform or pyasc build.

## References

- [Environment requirements](references/env-requirements.md)
