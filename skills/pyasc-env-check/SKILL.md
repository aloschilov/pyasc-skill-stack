---
name: pyasc-env-check
description: Verify a current-v2 pyasc source/runtime/toolkit environment, including exact commit provenance, Python ABI, CANN, compiler, camodel, torch_npu, and NPU availability.
---

# pyasc v2 environment check

`asctile` is compiled into the runtime at **build time**. There are two
supported ways to obtain it, and they differ in import layout:

| Runtime | Install | `asctile` import path |
|---|---|---|
| Published wheel (default) | `pip install --pre pyasc-dev` | `from asc.experimental import asctile` |
| Source build | `PYASC_SETUP_EXPERIMENTAL=1 pip install <src>` | varies; older checkouts expose top-level `asctile` |

Plain `pip install pyasc` never provides `asctile` — the stable distribution
ships the v1 `asc` API only. Setting `PYASC_SETUP_EXPERIMENTAL=1` while
installing an already-built wheel has no effect; it is a source-build flag.

Never install `pyasc` and `pyasc-dev` into the same environment: both provide
the `asc` import package. Use a dedicated virtualenv.

## Required checks

- runtime provenance: distribution version and the upstream commit it was built from
- wheel SHA-256, architecture, and CPython ABI
- `asctile` imports, and which path it resolves from
- Python version supported by that wheel
- CANN toolkit root/version and Bisheng compiler path
- target platform `Ascend950PR_9599`
- simulator library path for camodel
- torch and torch_npu versions for real tensor launches
- NPU presence and account access when running hardware

## Quick commands

```bash
python3 -c "
try:
    from asc.experimental import asctile          # published wheel
except ImportError:
    import asctile                                # legacy source build
import asc
print(asc.__file__, asctile.__file__)
"

# Which runtime is this, and what was it built from?
python3 -c "import importlib.metadata as m; print(m.version('pyasc-dev'))"
python3 -c "import importlib.metadata as m, re; \
d = m.metadata('pyasc-dev')['Description']; \
print(re.search(r'Source: .([0-9a-f]{40}).', d).group(1))"

python3 --version
test -x "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng"
npu-smi info
```

The import command only proves which installed runtime is visible. Compare it
to the recorded wheel/source before using its output as evidence.

The published wheel tracks the upstream default branch, so there is no single
expected commit — read the actual one from the metadata above. When a task
pins a specific revision, verify the installed runtime matches that pin and
stop if it does not.

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
