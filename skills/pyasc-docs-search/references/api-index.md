# Current pyasc v2 API index

Verify every signature against the selected commit's `python/asctile` source.

| Area | Current API |
|---|---|
| JIT | `asctile.jit` |
| GM descriptor | `asctile.global_tensor(base, shape)` |
| GM -> local | `asctile.copy_in(src, offsets, shape, location=..., real_shape=..., pad_value=...)` |
| local -> GM | `asctile.copy_out(src, dst, offsets, real_shape=...)` |
| local movement | `asctile.copy(src, offsets=None, shape=None, location=None)` |
| placement | `asctile.TensorLocation` |
| program model | `asctile.block_idx`, `block_num`, `sub_block_idx`, `sub_block_num` |
| loops | `asctile.range`, `static_range` |
| creation | `full`, `zeros`, `full_like`, `zeros_like`, `cast` |
| unary | `abs`, `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `erf`, `softmax`, `rms_norm`, ... |
| binary | `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `pow`, comparisons, ... |
| shape | `reshape`, `transpose`, `broadcast_to`, `expand_dims`, `squeeze`, ... |
| reduction | `reduce_sum`, `reduce_max`, `reduce_min`, `reduce_prod` |
| atomic | `atomic_add`, `atomic_max`, `atomic_min` |
| shared types | `asc.GlobalAddress`, `asc.ConstExpr`, dtypes, backend/platform configuration |

Current pyasc does not export the old tile names `asc2`, `tensor`, `load`, or
`store`.
