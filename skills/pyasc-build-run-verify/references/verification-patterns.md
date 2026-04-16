# pyasc Verification Patterns

## Standard verification with torch

```python
import torch

def verify_kernel(kernel_launch_fn, x, y, expected_fn, atol=1e-5):
    result = kernel_launch_fn(x, y)
    expected = expected_fn(x, y)
    np.testing.assert_allclose(result, expected, atol=atol, rtol=1e-3)
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

1. **Syntax check**: Verify all `@asc.jit` code uses supported syntax
2. **ASC-IR dump**: Set `PYASC_DUMP_PATH` and check generated IR
3. **Code review**: Manual review against API patterns
4. **State limitation**: "Runtime verification pending; static checks passed"
