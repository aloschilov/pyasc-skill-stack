# pyasc asc2 kernel contract (follow EXACTLY — every rule below was learned from real failures on this hardware)

## Module shape

Your file becomes `cann_bench/<module>.py` inside the submission wheel. It must contain:

- imports at module top: `import torch`, `import asc`, `import asc2`,
  `from ._pyasc_runtime import ensure_npu_platform` (and `import math` if needed)
- one or more `@asc2.jit` kernel functions
- ONE public callable matching the operator schema exactly (name and signature)
- wrapper body: call `ensure_npu_platform()` first; make inputs contiguous if
  needed (`x = x.contiguous()` is allowed); allocate outputs with
  `torch.empty_like(x)` or `torch.empty(shape, dtype=..., device=x.device)`;
  launch `kernel[cores](tensor_args..., int_args..., float_args..., constexpr_args...)`;
  return contiguous NPU tensor(s)

## Kernel authoring rules

- Global memory views: `asc2.global_tensor(ptr, [size])` (1-D) or
  `asc2.global_tensor(ptr, [rows, cols])` (2-D). Ranks of global_tensor /
  copy_in / copy_out / offsets must ALL match — never mix 1-D and 2-D.
- Kernel params: pointers typed `asc.GlobalAddress`; sizes as plain `int`
  (runtime); tile sizes as `asc.ConstExpr[int]` (compile-time; REQUIRED for any
  value used inside a copy_in tile shape); scalars as `float`.
- Grid-stride tile loop (the proven pattern):

```python
for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):
    off = t * tile_size
    n = tile_size if off + tile_size <= size else size - off   # tail handling
    x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
    ...compute on tiles...
    asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])
```

- Launch: `kernel[cores](...)` with `cores = min(72, num_tiles)` (72 AIV cores
  on this 950PR box). No stream argument.
- Available tile ops: `+ - * /` (tile-tile and tile-scalar), `asc2.abs`,
  `asc2.exp`, `asc2.exp2`, `asc2.log`, `asc2.log2`, `asc2.sqrt`,
  `asc2.rsqrt`, `asc2.tanh`, `asc2.erf`, `asc2.sin`, `asc2.cos`,
  `asc2.floor`, `asc2.ceil`, `asc2.relu`, `asc2.maximum`, `asc2.minimum`,
  comparisons (`x >= 0.0`, `asc2.less(a, b)`, ... — NO int64 operands),
  `asc2.where(cond, a, b)`, `asc2.reduce_sum(x)`, `asc2.reduce_max(x)`,
  `asc2.reduce_min(x)`, `asc2.full([shape], scalar, dtype=...)`,
  `asc2.cast(tile, dtype)` / `tile.to(dtype)` casts, integer
  `asc2.left_shift`/`asc2.right_shift`, tile-shape ops `asc2.reshape`,
  `asc2.transpose`, `asc2.ravel`, `asc2.expand_dims`, `asc2.squeeze`,
  `asc2.broadcast_to`, `asc2.concat`, unary `-x`.
- int8 tiles: loading (copy_in) is fine but NO vector op accepts int8 input
  (not even `.to`); convert with `asc2.cast(t, asc.float16)` first. There is
  no uint8 tile dtype at all.
- Scalars go on the RIGHT of tile arithmetic (Tile has no `__rmul__`):
  write `x * 0.5`, NEVER `0.5 * x`. Same for `+ - /`.
- f16/bf16 inputs: promote to f32 in-kernel (`xf = x.to(asc.float32)`),
  compute in f32, cast back on copy_out (`y.to(x.dtype)`).
- UB (unified buffer) budget: ~253952 bytes total under static allocation.
  Every distinct f32 tile value costs `4 * TILE` bytes, `unroll_factor=2`
  doubles the total, and the compiler adds hidden temporaries — MEASURED
  calibration: the sigmoid chain (f16 load, f32 cast, `-x`, `exp`, `+1`,
  `div`, f16 store ≈ 6 visible values) uses 155648 bytes at TILE=2048 and
  311296 (OVERFLOW) at TILE=4096, i.e. real usage ≈ 1.6x the naive
  `visible_values * 4 * TILE * 2` estimate. Budget with that 1.6x factor.
  Rule of thumb: TILE=2048 for short chains (< 8 values), 1024 for medium,
  512 for long (> 16). A launch failing with `RuntimeError: UB overflow: X
  bytes are available, Y bytes are used` means: halve TILE (do NOT drop
  cases).
- `asc2.where` / comparison destination tiles must be a multiple of 256 bytes
  (`TILE * 4 % 256 == 0` for f32 — any TILE >= 64 is safe).
- Loop-carried scalar accumulators (VERIFIED on this build): seed with
  `acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))`
  (a bare `acc = 0.0` fails codegen with "re-assigned to an object with
  different type"), then `acc = acc + asc2.reduce_sum(x)` inside a plain
  `asc2.range(...)` loop. `asc2.range` accepts ONLY `unroll_factor` and
  `parallel` — there is NO `gm_barrier` kwarg on this build (it raises
  TypeError).
- Cross-core reductions (VERIFIED): `asc2.atomic_add(src_tile, dst_gm,
  [offset])` atomically accumulates a tile into global memory (dtypes int16/
  int32/f16/bf16/f32; also `asc2.atomic_max`). Host must zero the
  destination first (`torch.zeros(...)` — tensor creation is allowed).
  Pattern: each core reduce_sums its tiles into a scalar, widens it with
  `asc2.full([8], s, dtype=...)`, and atomic_adds slot [0]; a second tiny
  kernel (or the same one on one core) applies any final transform.
- Scalar reduction results must be widened before store:
  `asc2.copy_out(asc2.full([8], s, dtype=...), out_gm, [0], real_shape=[8])`
  style (min 32 bytes).
- Inside `@asc2.jit`: NO `print`, NO imports, NO `break`/`continue`/early
  `return`, NO exceptions, NO Python `range()` over runtime values (use
  `asc2.range`), NO `math.*` calls (precompute module-level constants).

## Numerical stability (MANDATORY — f32 cases use ranges like [-88, 88] and [-100, 100] under a ~1.2e-4 relative-error threshold)

- Never let `exp()` see a positive argument that can overflow; never subtract
  nearly-equal quantities (catastrophic cancellation); never rely on
  `log(1 + tiny)` (flushes to 0 below tiny < 6e-8).
- Proven cancellation-free building blocks (all verified on this harness):
  - `sigmoid(s) = exp(min(s, 0)) / (1 + exp(-|s|))`
  - `1 + tanh(u) = 2 * sigmoid(2u)`
  - `tanh(softplus(x))`: with `w = exp(-|x|)`, equals
    `(1 + 2w) / (1 + 2w + 2w^2)` for `x >= 0`, `(w^2 + 2w) / (w^2 + 2w + 2)`
    for `x < 0` (exact identities; blend with `asc2.where(xf >= 0.0, ...)`)
  - `erfc(z)` for `z >= 0`: Numerical Recipes fit `t * exp(-z*z + P(t))`,
    `t = 1/(1 + z/2)`, rel. err < 1.2e-7 (see the gelu reference module for
    the 9-coefficient Horner chain)
- IEEE special values (inf/nan scalars or extreme inputs) propagate correctly
  through the hardware ops — do NOT special-case them with host branches
  unless the golden does.

## Anti-cheat (violations zero the submission)

- ALL numerical work happens inside `@asc2.jit` kernels launched on the NPU.
- torch usage is allowed ONLY for: output allocation (`torch.empty`,
  `torch.empty_like`), metadata (`.shape`, `.numel()`, `.stride()`, `.dtype`,
  `.is_contiguous()`), contiguity (`.contiguous()`), and views (`.view`,
  `.reshape`, `.narrow`, indexing that returns a view).
- FORBIDDEN anywhere in the module: torch math/compute ops (`torch.mul`,
  `torch.norm`, `torch.nn.functional.*`, tensor arithmetic like `a + b`,
  `x.sigmoid()`, `.to(dtype)` casts of device data, `torch.cat`,
  `torch.clone`, `torch.sum`, ...). The harness hooks torch dispatch and
  rotates input data pointers between calls — caching outputs by `data_ptr`
  is detected and scored as cheating.
- Outputs must be contiguous NPU tensors with exactly the golden's
  shape/dtype. Do not return views of inputs.
