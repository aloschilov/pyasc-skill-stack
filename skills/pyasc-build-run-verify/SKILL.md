---
name: pyasc-build-run-verify
description: pyasc kernel build, run, and verification skill. Provides JIT compilation diagnostics, runtime execution guidance, and output verification patterns. Trigger — after kernel implementation, when running pyasc kernels, debugging JIT errors, or verifying kernel output correctness.
---

# pyasc Build, Run, and Verify

## Overview

pyasc uses JIT (Just-In-Time) compilation: Python -> ASC-IR -> Ascend C -> Bisheng compiler -> NPU binary. This skill covers the build/run/verify lifecycle.

## Workflow

```
Kernel implementation complete
    |
    +-- JIT compilation (automatic on first call)
    |       |
    |       +-- Success -> Run kernel
    |       |
    |       +-- Failure -> Check diagnostics
    |
    +-- Run kernel
    |       |
    |       +-- Model backend (simulator, always available)
    |       |
    |       +-- NPU backend (requires hardware)
    |
    +-- Verify output
            |
            +-- torch.allclose / numpy comparison
```

## Running a pyasc kernel

### Basic execution

> **IMPORTANT**: Use `python3.10` (not `python` or `python3`). The pyasc and torch packages are installed under python3.10.

```bash
# Set up simulator environment (required for Model backend)
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH

# Run with Model backend (simulator) — specify platform explicitly
python3.10 kernel.py -r Model -v Ascend910B1

# Run with NPU backend (requires hardware)
python3.10 kernel.py -r NPU -v Ascend910B1
```

> The `-v Ascend910B1` flag is required. Do NOT use `-v Ascend910B` (missing version suffix).

### Script tool

```bash
bash scripts/run_kernel.sh {kernel_path} [backend] [platform]
```

## JIT Diagnostics

### Environment variables for debugging

| Variable | Purpose | Example |
|----------|---------|---------|
| `PYASC_DUMP_PATH` | Save generated ASC-IR and Ascend C files | `export PYASC_DUMP_PATH=/tmp/pyasc_dump` |
| `PYASC_HOME` | JIT cache root directory | `export PYASC_HOME=$HOME` |
| `PYASC_CACHE_DIR` | Specific cache directory | `export PYASC_CACHE_DIR=$HOME/.pyasc/cache` |

### Compile options for debugging

| Option | Purpose | Usage |
|--------|---------|-------|
| `always_compile=True` | Force recompilation (bypass cache) | `@asc.jit(always_compile=True)` |
| `auto_sync=True` | Let compiler insert sync automatically | `@asc.jit(auto_sync=True)` |
| `auto_sync_log="sync.log"` | Log auto-sync insertions | `@asc.jit(auto_sync=True, auto_sync_log="sync.log")` |
| `opt_level=0` | Disable optimizations for debugging | `@asc.jit(opt_level=0)` |

### Common JIT errors

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `SyntaxError` in AST visitor | Unsupported Python syntax | Check `pyasc-syntax-constraints` |
| Type error in IR builder | Wrong parameter type | Check type constraints for kernel params |
| Bisheng compilation error | Invalid generated Ascend C | Check `PYASC_DUMP_PATH` output for generated code |
| `ImportError: asc` | pyasc not installed | Run `pip install pyasc` or build from source |
| `RuntimeError` on launch | Wrong core count or missing stream | Verify `core_num` and `rt.current_stream()` |

## Verification Patterns

### torch.allclose verification

```python
import torch
result = kernel_launch(x, y)
expected = x + y  # or whatever the operation should produce
assert torch.allclose(result, expected, atol=1e-5), \
    f"Max diff: {(result - expected).abs().max()}"
```

### numpy verification

```python
import numpy as np
result_np = result.cpu().numpy()
expected_np = expected.cpu().numpy()
assert np.allclose(result_np, expected_np, atol=1e-5)
```

### Verification script

```bash
python scripts/verify_output.py {kernel_path} [--backend Model] [--atol 1e-5]
```

## Backend selection

| Backend | When to use | Availability |
|---------|-------------|-------------|
| `Model` | Development, CI, no NPU hardware | Always (requires CANN simulator libs) |
| `NPU` | Final verification, performance testing | Requires Atlas A2/A3 hardware |

**If runtime execution is unavailable**: Perform static verification (syntax check, ASC-IR dump inspection) and state the limitation explicitly in the delivery.

## References

- [JIT Diagnostics Guide](references/jit-diagnostics.md)
- [Verification Patterns](references/verification-patterns.md)
