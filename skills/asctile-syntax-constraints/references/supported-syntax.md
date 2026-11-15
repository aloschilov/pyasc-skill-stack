# Supported selected-runtime JIT syntax

AscTile supports ordinary assignments, annotated assignments,
`if`/`elif`/`else`, ternary expressions, `while`, compile-time Python `range`,
device `asctile.range`, arithmetic/comparison/bitwise/boolean operators,
module-level JIT helper calls, tuple/list construction and indexing when types
are resolvable, slicing, assertions, and top-level returns from device helpers.

```python
@asctile.jit
def helper(x):
    y = x * 2.0
    return y

@asctile.jit
def kernel(x_ptr: asc.GlobalAddress, n: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [n])
    for i in asctile.range(asctile.block_idx(), n, asctile.block_num()):
        x = asctile.copy_in(x_gm, [i])
        y = helper(x)
```

`asctile.block_idx` and `asctile.block_num` are functions and must be
called. Bare names materialize as function objects and the loop fails to
lower with `TypeError: Unsupported value type for materialization:
function`. This exact signature was measured on the pinned runtime in
multiple CANNBench candidates; always write `asctile.block_idx()` /
`asctile.block_num()`.

The public CANN 9.2 AscTile JIT accepts two operands per boolean AST node.
For example, `a >= 0 and a < D and b >= 0` contains two operators and
three operands in one node. Compilation raises `UnsupportedSyntaxError`
for that chain. Group the checks as `(a >= 0 and a < D) and (b >= 0 and b < H)`
or assign the two pairs to names before combining them. The same rule
applies to conditions in ternary expressions. Ordinary host Python
conditions can use longer chains.

For a quick source check on this wheel, inspect each device function for
`ast.BoolOp` nodes with `len(node.values) > 2`. Then compile inputs that
select each branch. A clean SwiGLU generation compiled on the aligned
path and failed on a padded path with a four-operand chain. After the skill
grouped the predicates, an independent generation compiled and ran the
padded case on David. [Compiler and device results](https://github.com/aloschilov/pyasc-skill-stack/pull/60#issuecomment-6891057620).

A name may not be re-bound to a value of a *different type* within a JIT
function. This bites hardest when clamping a partial-tile tail, where the
obvious `if` form rebinds a traced value to a plain Python `int`:

<!-- skill-code: counter-example -->
```python
# Rejected: UnsupportedSyntaxError: 'valid' was re-assigned to an object
# with different type: initial type is PlainValue, new type is int
valid = size - tile_offset      # PlainValue (traced)
if valid > tile_size:
    valid = tile_size           # int (compile-time constant)
```

```python
# Correct: one binding, one type
valid = min(size - tile_offset, tile_size)
```

`min()` and `max()` lower correctly over a traced value and a `ConstExpr`
bound, so they are the idiomatic way to clamp. Measured on the pinned
pyasc-sim-dev image: the `if` form fails codegen before any launch, the
`min()` form compiles and executes.

Check optional loop and JIT keywords against the exact selected source.
