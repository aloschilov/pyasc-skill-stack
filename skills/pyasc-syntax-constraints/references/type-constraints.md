# pyasc Type Constraints

## Kernel parameter types

### Supported as runtime parameters

| Type | Description | Example |
|------|-------------|---------|
| `bool` | Boolean | `def kernel(flag: bool)` |
| `int` | Integer | `def kernel(n: int)` |
| `float` | Float | `def kernel(scale: float)` |
| `numpy` scalar | numpy scalar types | `def kernel(val: np.float32)` |
| `numpy.ndarray` | numpy array | Passed as data |
| `torch.Tensor` | PyTorch tensor | Auto H2D/D2H transfer |
| `asc.GlobalAddress` | Global memory pointer | Explicit memory management |

### Not supported as runtime parameters

| Type | Workaround |
|------|------------|
| `str` | Use `asc.ConstExpr[str]` for compile-time strings |
| `tuple` | Use `asc.ConstExpr[tuple]` or pass elements separately |
| `list` | Use `asc.ConstExpr[list]` or pass elements separately |
| `dict` | Use `asc.ConstExpr[dict]` or decompose |
| Complex objects | Decompose into supported scalar/tensor types |

## Compile-time parameters (ConstExpr)

For values resolved at compile time:

```python
@asc.jit
def kernel(x: asc.GlobalAddress, mode: asc.ConstExpr[int]):
    if mode == 1:    # resolved at compile time, not runtime
        ...
    elif mode == 2:
        ...
```

`ConstExpr` is useful for:
- Branch selection that should be resolved at compile time
- Container parameters (tuple, list, dict)
- String parameters
- Any value that changes the compiled code structure

## Built-in whitelist

Only these built-in functions are available inside `@asc.jit`:

| Built-in | Available |
|----------|-----------|
| `dict` | Yes |
| `float` | Yes |
| `int` | Yes |
| `isinstance` | Yes |
| `issubclass` | Yes |
| `len` | Yes |
| `list` | Yes |
| `range` | Yes |
| `repr` | Yes |
| `str` | Yes |
| `tuple` | Yes |
| `type` | Yes |
| `print` | **No** |
| `open` | **No** |
| `input` | **No** |
| `map`, `filter`, `zip` | **No** |
| `sorted`, `reversed` | **No** |
| Any stdlib | **No** |

## Tensor dtype

Tensor data types are accessed from the tensor itself:

```python
data_type = x.dtype           # from GlobalAddress parameter
buffer_size = n * data_type.sizeof()
```

Common dtypes: `asc.float32`, `asc.float16`, `asc.int32`, `asc.int8`
