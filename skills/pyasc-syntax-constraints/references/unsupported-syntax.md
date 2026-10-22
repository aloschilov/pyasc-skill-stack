# Unsupported current-v2 JIT syntax

Do not use nested functions, class-method kernels, `global`, `nonlocal`,
`break`, `continue`, imports inside JIT, file I/O, `print`, `try`/`except`,
`raise`, `with`, lambda, generators, async syntax, or returns from the middle
of runtime control flow.

```python
# Invalid: runtime branch can leave y undefined.
@asctile.jit
def helper(x):
    if x:
        y = 1
    return y
```

Initialize values on every runtime path or make the controlling attribute a
`ConstExpr` when it genuinely changes generated code. Restructure loops rather
than relying on `break`/`continue`.
