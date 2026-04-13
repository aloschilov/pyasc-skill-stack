---
name: pyasc-syntax-constraints
description: pyasc Python syntax support and restrictions reference. Documents supported and unsupported Python syntax within @asc2.jit functions, type constraints, and ConstExpr usage. Trigger — when writing kernel code, checking if a Python construct is supported, or debugging JIT compilation errors related to syntax.
---

# pyasc Syntax Constraints

## Overview

pyasc compiles Python code to Ascend C via JIT. Only a subset of Python syntax is supported inside `@asc2.jit`-decorated functions. This skill documents what is and is not allowed.

## Supported Syntax

| Syntax | Example | Notes |
|--------|---------|-------|
| Attribute access | `nums.__len__()` | |
| `while` loop | `while i < n: ...` | |
| `for` loop | `for i in asc2.range(n): ...` | Use `asc2.range()` inside kernel, not `range()` |
| Binary operators | `a + b * c` | `+`, `-`, `*`, `/`, `%`, `//`, `**`, `&`, `\|`, `^`, `<<`, `>>` |
| Unary operators | `~a`, `-a`, `not a` | |
| `tuple` | `return x, y, z` | |
| `list` | `nums = [a, b, c]` | |
| `pass` | `pass` | |
| Ternary `if` | `ans = x if x > 0 else 0` | |
| `if`/`elif`/`else` | Full conditional blocks | |
| Constants | `a = 1; b = True` | |
| Comparisons | `>`, `<`, `>=`, `<=`, `==`, `!=` | |
| Function calls | `func()` | Must be `@asc2.jit`-decorated or asc2 built-in |
| Boolean logic | `and`, `or` | |
| Augmented assign | `+=`, `-=`, `*=`, `/=`, `%=` | |
| Assignment | `x = expr` | |
| Annotated assign | `count: int = 3` | |
| f-strings + assert | `assert cond, f"msg {val}"` | |
| Subscript | `nums[0]` | |
| Slice | `tensor[offset:]` | |

## Unsupported Syntax

| Syntax | Status | Workaround |
|--------|--------|------------|
| Nested functions | Not supported | Define separate `@asc2.jit` functions |
| `global` | Not supported | Pass values as parameters |
| `return` from kernel | Not supported | Kernels write to output tensors; device functions may return |
| `return` inside `if`/`for` | Not supported | Use top-level return only in device functions |
| `continue` | Not supported | Restructure loop logic |
| `print()` | Not supported | Use `assert` with f-strings |
| `with open(...)` | Not supported | Do file I/O outside JIT |
| `raise` | Not supported | Use `assert` instead |
| `try`/`except` | Not supported | Handle errors outside JIT |
| `yield` / `yield from` | Not supported | No generators |
| `lambda` | Not supported | Define named functions |
| `break` | Not supported | Restructure loop logic |
| `import` / `from...import` | Not supported | All imports outside JIT |
| `async def` / `async with` | Not supported | No async |
| `nonlocal` | Not supported | Pass values as parameters |
| Class methods as kernels | Not supported | Use module-level functions |
| If-scoped variables | Restricted | Initialize before `if`; use `asc.ConstExpr` for compile-time branching |
| Unsigned int comparison | Restricted | Use `np.int64` or truthiness check instead |

## Type Constraints

### Kernel parameter types (runtime)

| Supported | Not supported |
|-----------|---------------|
| `bool`, `int`, `float` | `str` |
| `numpy` scalars and `ndarray` | `tuple`, `list`, `dict` |
| `torch.Tensor` | Complex objects |
| `asc.GlobalAddress` | |

### Compile-time parameters

Use `asc.ConstExpr[T]` for values that must be resolved at compile time:
```python
@asc2.jit(always_compile=True)
def kernel(x_ptr: asc.GlobalAddress, tile_size: asc.ConstExpr[int]):
    # tile_size is resolved at compile time and included in the JIT cache key
    ...
```

### Built-in whitelist inside JIT

Only these built-ins are available: `dict`, `float`, `int`, `isinstance`, `issubclass`, `len`, `list`, `range`, `repr`, `str`, `tuple`, `type`

Note: For iteration inside `@asc2.jit`, use `asc2.range()` instead of `range()`. `asc2.range()` supports `unroll_factor` and `parallel` options.

## Kernel vs Device Function Rules

| Rule | Kernel function | Device function |
|------|----------------|-----------------|
| Launched from | Host via `kernel[core_num](...)` | Other `@asc2.jit` functions |
| `return` value | Not allowed | Allowed |
| `@asc2.jit(...)` options | Effective (always_compile, etc.) | Ignored |
| Nesting | Cannot be nested | Cannot be nested |

## References

- [Supported Syntax Details](references/supported-syntax.md)
- [Unsupported Syntax Details](references/unsupported-syntax.md)
- [Type Constraints](references/type-constraints.md)
- [ConstExpr Guide](references/constexpr-guide.md)
