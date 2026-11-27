# Unsupported selected-runtime JIT syntax

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

Keep a variable's JIT value type consistent across runtime branches. On PyASC
`ed793e45`, a generated Exp kernel computed `count = stop - first` as a
`PlainValue`, then assigned `count = 0` in a dynamic branch. Lowering rejected
the Python `int` reassignment before any launch. For that contiguous ownership
loop, computing `stop = min(start + group, num_tiles)` and iterating
`asctile.range(start, stop, unroll_factor=1)` excluded empty bodies and passed
partial tails and entirely idle blocks. This qualified alternative does not
permit changing value types in other control-flow patterns.
[Retained generated sources and compiler result](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-exp-empty-slots-study-20261127).
