# ConstExpr guide

Use `asc.ConstExpr[T]` or `asctile.ConstExpr[T]` for values that determine
local shapes, compile-time branch structure, unrolling, or specialization.

```python
@asctile.jit
def kernel(x_ptr: asc.GlobalAddress,
           size: asc.ConstExpr[int],
           tile_size: asc.ConstExpr[int],
           mode: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    if mode == 0:
        x = asctile.copy_in(x_gm, [0], [tile_size])
```

Different values can create different compiled kernels and cache entries.
Keep ordinary scalar data runtime-valued; specialize only when generated code
or static tensor shape actually changes.
