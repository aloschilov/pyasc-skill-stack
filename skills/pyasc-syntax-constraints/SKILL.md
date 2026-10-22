---
name: pyasc-syntax-constraints
description: Current pyasc v2 syntax, typing, ConstExpr, loop, and kernel/device-function constraints. Use when writing asctile kernels or diagnosing AST/type/JIT errors.
---

# Current pyasc v2 syntax constraints

Use the syntax implementation and tests from the exact selected `v2` commit.
For the current CANNBench campaign that is
`0a631f70968c3cb7c33ce45330a85768dd5a6f06`. New tile kernels use
`@asctile.jit`; `@asc2.jit` is obsolete.

## Supported core constructs

- assignments and annotated assignments
- `if`/`elif`/`else` and ternary expressions
- `for` with compile-time Python `range` or device `asctile.range`
- `while`
- arithmetic, comparison, bitwise, unary, and boolean operators
- tuple/list creation, indexing, and slicing where the compiler can resolve
  the involved types
- calls to module-level JIT helpers and current pyasc builtins
- assertions with simple/f-string messages

Inside a device loop use only keywords present in the selected source.
Current v2 `asctile.range` supports `unroll_factor` and `gm_barrier`; it does
not support a `parallel` keyword. Use a plain Python `range` for a genuinely
compile-time loop that should be fully unrolled.

## Unsupported or unsafe constructs

- nested function definitions
- `global` or `nonlocal`
- `break` or `continue`
- `try`/`except`, `raise`, or `with`
- `yield`, lambda, async definitions, or `await`
- imports inside a JIT function
- class methods as kernels
- host I/O such as `print` or file access
- variables defined on only one runtime branch and read afterwards

The outer launched kernel writes outputs and must not return a Python value.
A module-level JIT device helper may return at its top level. Do not return
from the middle of runtime control flow.

## Runtime and compile-time values

Runtime parameters may be scalar bool/int/float, numpy scalars/arrays,
torch tensors on the supported path, and `asc.GlobalAddress`. Values that
control local tensor shapes, static iteration, specialization, or compile-time
branches must be `asc.ConstExpr[T]`/`asctile.ConstExpr[T]` or otherwise known
when the JIT specializes the function.

```python
@asctile.jit
def kernel(x_ptr: asc.GlobalAddress,
           size: asc.ConstExpr[int],
           tile_size: asc.ConstExpr[int],
           mode: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    if mode == 0:
        for i in asctile.range(asctile.ceildiv(size, tile_size)):
            ...
```

Container and string attributes normally remain host-side and choose a
specialized launch. Avoid passing arbitrary Python objects into the JIT.

Two measured official-contract attribute pitfalls:

- Official cases encode some numeric attrs as strings, e.g.
  `epsilon: '1e-05'` (group_norm, rms_norm, add_rms_norm_dynamic_quant,
  apply_adam_w, mhc_sinkhorn, mla_prolog) and `norm_eps: '1e-05'`
  (engram_gate_fusion). Passing the raw attr into a JIT call fails with
  `TypeError: Argument type in JIT function is not supported: str` (measured,
  all 20 cases). Coerce in host code before the launch:
  `eps = float(norm_eps)`. The same applies to int-like strings.
- List-valued attrs such as `group_list` (grouped_matmul_swiglu_quant) are
  host-side Python lists, not tensors. Calling tensor methods on them fails
  with `AttributeError: 'list' object has no attribute 'contiguous'`
  (measured, all 20 cases). Keep list attrs as metadata; when the kernel
  needs the values on-device, materialize with the whitelisted host factory
  `torch.tensor(group_list, dtype=torch.int32, device=...)` and pass that to
  the launch (`torch.tensor` is in the static whitelist; `torch.as_tensor`
  is NOT whitelisted and has no FakeTorch support).

## Values and tensor APIs

`copy_in` without `shape` and reductions can produce `PlainValue`. Many unary
and binary tile APIs require `LocalTensor`. Do not assume a scalar result can
be passed to tensor-only `rsqrt`, `maximum`, or `where`; use supported scalar
arithmetic or materialize an aligned tensor with `asctile.full`.

Global descriptors, offsets, local shapes, and `real_shape` must have matching
rank. For 2-D-or-higher UB copies, the final physical dimension must satisfy
32-byte transfer alignment. Current transpose axes are separate positional
integers (`tile.transpose(*perm)`), not one list argument.

## Placement and pipelining

Use `asctile.TensorLocation.L1/L0A/L0B/L0C/UB`. Software pipelining via
`unroll_factor=2` can double live local/L0 buffers, so validate memory use for
the largest specialization. Do not use the removed `run_asc2_passes` option.

## Validation rule

Syntax validity is only the first gate. Parse/compile every host-selected
specialization against the exact v2 runtime, then execute representative
ordinary and special-value cases on camodel. QEMU compile-only results do not
prove numerical correctness or performance.

## References

- [Supported syntax](references/supported-syntax.md)
- [Unsupported syntax](references/unsupported-syntax.md)
- [Type constraints](references/type-constraints.md)
- [ConstExpr guide](references/constexpr-guide.md)
