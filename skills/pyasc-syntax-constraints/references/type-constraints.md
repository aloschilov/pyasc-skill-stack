# Current-v2 type constraints

Runtime parameters may include bool/int/float, supported numpy scalars and
arrays, torch tensors on the launch path, and `asc.GlobalAddress`. Use
`ConstExpr` for static shapes and compile-time attributes. Do not pass arbitrary
objects or host containers unless the exact JIT contract explicitly supports
them as compile-time values.

`global_tensor` shapes and offsets are integer-like runtime/static values;
local `copy_in` shapes are static. `copy_in` without a shape and reductions may
produce `PlainValue`. APIs such as tile `rsqrt`, `where`, or `maximum` can
require `LocalTensor`, so materialize an aligned tensor when necessary.

`PlainValue` defines no `__int__`/`__float__`/`__index__`: calling Python
`int()`/`float()` on a device value fails at codegen with `TypeError:
int() argument must be a string, a bytes-like object or a real number,
not 'PlainValue'`. Never convert a `reduce_sum`/scalar load result into a
Python control-flow value inside JIT code. Keep the value in the device
domain: compare and index with it directly, or restructure so host code
precomputes the control flow. If a scalar index must come from memory,
load it with a shaped `copy_in` and use device-side arithmetic
(measured on pinned-runtime moe_re_routing).

`asctile.where(cond, a, b)` branches must be tensors or explicitly cast
values, e.g. `asctile.where(mask, asctile.cast(ONE, out.dtype),
asctile.cast(ZERO, out.dtype))` or `asctile.full`-materialized constants.
Passing bare Python floats fails with `AttributeError: 'float' object has
no attribute 'dtype'` (measured on pinned-runtime nms).

Common dtypes are exposed by both `asc` and `asctile`, including float16,
bfloat16, float32, signed integers, and bool. Individual operations support
only subsets; verify their validation function in `python/asctile/language/`.
