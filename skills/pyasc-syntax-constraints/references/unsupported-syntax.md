# pyasc Unsupported Syntax Details

None of the syntax below is valid inside `@asc.jit`-decorated functions.

## Nested functions

```python
# NOT SUPPORTED
@asc.jit
def outer(x):
    def inner():        # nested function definition
        return x * 2
    return inner()
```

**Workaround**: Define `inner` as a separate `@asc.jit` function at module level.

## global / nonlocal

```python
# NOT SUPPORTED
count = 0
@asc.jit
def increase():
    global count        # global statement
    count += 1
```

**Workaround**: Pass the value as a parameter.

## return from kernel

```python
# NOT SUPPORTED - kernels cannot return values
@asc.jit
def kernel(x):
    return x * 2        # return from kernel

# SUPPORTED - device functions can return (top-level only)
@asc.jit
def device_func() -> int:
    return 42           # OK for device functions
```

## continue / break

```python
# NOT SUPPORTED
@asc.jit
def func(n):
    for i in range(n):
        if i % 2 == 0:
            continue    # not supported
        if i > 10:
            break       # not supported
```

**Workaround**: Restructure loop logic to avoid these constructs.

## print

```python
# NOT SUPPORTED
@asc.jit
def func(name):
    print(name)         # not supported
```

**Workaround**: Use `assert` with f-strings for debug messages.

## File I/O, exceptions, raise

```python
# NOT SUPPORTED
@asc.jit
def func():
    with open('f.txt') as f: ...    # with open
    raise ValueError("error")       # raise
    try:                             # try/except
        x = 1 / 0
    except:
        pass
```

## Generators (yield, yield from)

```python
# NOT SUPPORTED
@asc.jit
def gen():
    yield 1
    yield from other_gen()
```

## Lambda

```python
# NOT SUPPORTED
@asc.jit
def func(x):
    f = lambda a: a * a
```

## Import inside JIT

```python
# NOT SUPPORTED
@asc.jit
def func():
    import math
    from random import randint
```

## Async

```python
# NOT SUPPORTED
@asc.jit
async def func():
    await something()
```

## Class methods as kernels

```python
# NOT SUPPORTED
class MyKernel:
    @asc.jit
    def kernel(self, x):
        ...
```

## If-scoped variable leakage

```python
# NOT SUPPORTED - y undefined on else path
@asc.jit
def func(x):
    if x:
        y = 1
    z = y + 2   # y may not be defined
```

**Workaround**: Initialize `y` before `if`, or use `asc.ConstExpr[int]` for `x`.

## Unsigned integer comparison

```python
# PROBLEMATIC - uint64 comparison with 0
@asc.jit
def func(mask_low: np.uint64, mask_high: np.uint64):
    if mask_low > 0 and mask_high > 0:  # unsigned comparison issue
        ...
```

**Workaround**: Use `if mask_low and mask_high` (truthiness) or use `np.int64`.
