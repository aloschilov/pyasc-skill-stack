# pyasc ConstExpr Guide

## What is ConstExpr?

`asc.ConstExpr[T]` wraps a parameter type to indicate it must be resolved at compile time, not at runtime. The JIT compiler uses the concrete value to specialize the generated code.

## When to use

1. **Compile-time branching**: When an `if` condition determines code structure
2. **Unsupported runtime types**: When you need `str`, `tuple`, `list`, or `dict` as parameters
3. **Variable scoping**: When a variable is only assigned in one branch of an `if`

## Example: Compile-time branching

```python
@asc.jit
def kernel(x: asc.GlobalAddress, z: asc.GlobalAddress, mode: asc.ConstExpr[int], n: int):
    offset = asc.get_block_idx() * n
    x_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset, n)
    z_gm.set_global_buffer(z + offset, n)

    if mode == 1:
        # This branch is resolved at compile time
        # Only the matching branch is compiled
        ...
    elif mode == 2:
        ...
```

## Example: If-scoped variable fix

Without ConstExpr (broken):
```python
@asc.jit
def func(x):           # x is runtime, compiler can't resolve branch
    if x:
        y = 1
    z = y + 2           # ERROR: y may not be defined
```

With ConstExpr (fixed):
```python
@asc.jit
def func(x: asc.ConstExpr[int]):
    if x:               # resolved at compile time
        y = 1
    z = y + 2           # OK: compiler knows which branch is taken
```

## Limitations

- `ConstExpr` parameters become part of the cache key
- Different `ConstExpr` values produce different compiled kernels
- Overuse increases compilation time and cache size
