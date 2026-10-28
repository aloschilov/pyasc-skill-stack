# pyasc Verification Patterns

## Preflight: record the exact artifact first

Before any result is claimed, record what produced it. Run this and keep the
output with the evidence:

```bash
python - <<'PY'
import sys, importlib
try:
    a = importlib.import_module("asc.experimental.asctile"); how = "asc.experimental.asctile"
except ImportError:
    a = importlib.import_module("asctile"); how = "asctile"
import importlib.metadata as md
for _d in ("pyasc-dev", "pyasc"):
    try:
        dist, ver = _d, md.version(_d); break
    except md.PackageNotFoundError:
        continue
else:
    dist, ver = "none", "unknown"
print("python", sys.version.split()[0], "| import", how)
print("asctile", getattr(a, "__file__", None), "|", dist, ver)
print("jit", hasattr(a, "jit"), "global_tensor", hasattr(a, "global_tensor"))
PY
```

Use `md.version("pyasc-dev")` as well as `"pyasc"`: the published development
wheel is distributed as `pyasc-dev`, so looking up `"pyasc"` alone raises
`PackageNotFoundError` for every wheel user.

If `jit`/`global_tensor` are absent, report `blocked` with the interpreter and
version. Do not silently fall back to a different runtime; every result belongs
to the exact artifact recorded here.

## Report gates separately

| gate | evidence |
|---|---|
| generation | candidate file + sha256 + model id |
| compilation | codegen result for the target SoC |
| numerical | mismatch count, MARE/max-diff, tolerance, seed |
| hardware perf | measured latency only, with device/build identity |

A compile success is not a numerical pass. A simulator run is not a hardware
measurement. Missing runtime, hardware or provider is `blocked`, never
fabricated.

## Standard verification with torch

`pyasc-dev` does not depend on torch; install it separately or use the numpy
form below.

```python
import torch

def verify_kernel(kernel_launch_fn, x, y, expected_fn, atol=1e-5):
    result = kernel_launch_fn(x, y)
    expected = expected_fn(x, y)
    torch.testing.assert_close(result, expected, atol=atol, rtol=1e-3)
    return True
```

## Standard verification with numpy

```python
import numpy as np

def verify_kernel_np(result_np, expected_np, atol=1e-5):
    assert np.allclose(result_np, expected_np, atol=atol), \
        f"Max diff: {np.abs(result_np - expected_np).max()}"
    return True
```

## Recommended test shapes

| Category | Shapes | Purpose |
|----------|--------|---------|
| Minimal | `(64,)` | Quick smoke test |
| Standard | `(8 * 2048,)` | Matches tutorial default |
| Multi-core | `(8 * 4096,)` | Tests multi-core distribution |
| Large | `(8 * 16384,)` | Tests tiling correctness |

## Backend selection for verification

| Scenario | Backend | Notes |
|----------|---------|-------|
| CI / no hardware | Model | Always available with CANN |
| Final validation | NPU | Hardware verification |
| Both | Model + NPU | Most thorough |

## Static verification (when runtime unavailable)

If neither Model nor NPU backend is available:

1. **Syntax check**: Verify all `@asctile.jit` code uses supported syntax
2. **ASC-IR dump**: Set `PYASC_DUMP_PATH` and check generated IR
3. **Code review**: Manual review against API patterns
4. **State limitation**: "Runtime verification pending; static checks passed"
