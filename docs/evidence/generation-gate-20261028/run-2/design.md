# Vector Add — Design Document

## Semantics

Element-wise addition of two float32 tensors:

```
out[i] = x[i] + y[i],  i in [0, 1024)
```

Both inputs and the output are contiguous 1-D float32 tensors of size 1024
(4 KiB each). The operation is purely element-wise with no reduction, no
axis normalization, and no data-dependent control flow.

## Public signature

- Kernel (device, `@asctile.jit`):

  ```python
  def vector_add(x_ptr: asctile.GlobalAddress,
                 y_ptr: asctile.GlobalAddress,
                 out_ptr: asctile.GlobalAddress,
                 size: asctile.ConstExpr[int],
                 tile_size: asctile.ConstExpr[int])
  ```

- Host launcher: `vector_add_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor`

- Launch syntax: `vector_add[core_num](x, y, out, size, TILE_SIZE)` — no
  stream argument.

## Shape / dtype matrix

| Case | Shape | Dtype | Notes |
|------|-------|-------|-------|
| primary | [1024] | float32 | the required target |
| generic | [N], N multiple of tile_size | float32 | exercised by the same code path |
| tail | [N], N not a multiple of tile_size | float32 | handled via `real_shape` + neutral pad |

No attribute dispatch, no special values beyond IEEE float32 addition
semantics (infinities/NaN propagate as in the reference).

## Host dispatch and JIT specializations

- `size` and `tile_size` are `asctile.ConstExpr[int]`, so the fixed target
  `(1024, 128)` compiles exactly one specialization. Any other size produces
  a separate specialization through the same source.
- Host code owns allocation (`torch.empty_like`) and the launch; all
  numerical work is inside the JIT kernel. Tensors are passed directly to
  the launch (no `data_ptr()`), letting the JIT perform pointer dtype
  specialization.
- `core_num = min(CORE_NUM, num_tiles)`; for the target this is
  `min(72, 8) = 8` cores on Ascend950PR_9599.

## Global memory descriptors and transfer ranks

Inside the kernel body (never on the host):

```python
x_gm   = asctile.global_tensor(x_ptr,   [size])   # rank-1 GM descriptor
y_gm   = asctile.global_tensor(y_ptr,   [size])
out_gm = asctile.global_tensor(out_ptr, [size])
```

`copy_in` / `copy_out` use rank-1 offsets `[tile_offset]`, physical shape
`[tile_size]`, and `real_shape=[valid]`. Ranks of descriptor, offsets,
shape, and `real_shape` all match (rank 1). Offsets precede shape in the
argument order.

## Tiling, tails, padding, cores, UB budget

- `TILE_SIZE = 128` floats (512 bytes) → `num_tiles = ceildiv(1024, 128) = 8`.
- Grid-stride device loop:
  `asctile.range(asctile.block_idx(), num_tiles, asctile.block_num())`,
  so the kernel is correct for any core count ≤ or > the tile count.
- Tail handling: `valid = min(size - tile_offset, tile_size)` expressed as a
  runtime `if` (built-in `min` is not relied on inside the JIT). The target
  size 1024 divides evenly, so the tail path is dormant but keeps the kernel
  generic.
- Padding: `pad_value=0` is the operation-neutral identity for addition;
  padded lanes compute `0 + 0 = 0` and are excluded from the store by
  `real_shape` in `copy_out`.
- Alignment: rank-1 transfers; the physical tile of 128 float32 (512 bytes)
  satisfies any 32-byte transfer alignment concern trivially.
- UB budget: at most three live rank-1 tiles of 512 bytes each
  (`x`, `y`, `out` ≈ 1.5 KiB total), far below UB capacity. No L1/L0
  staging is needed for an element-wise op.

## Numerical precision

Inputs are already float32; addition is performed directly in float32 with
no casts. Element-wise add has no reassociation, so results are bit-exact
against the torch reference; `torch.allclose` (default tolerances) is used
as the acceptance check.

## Fallback routes

None required. Every case (aligned and tail) is handled by the same
asctile numerical kernel; there is no host numerical fallback.

## Validation plan

1. `python3 -m py_compile kernels/add/kernel.py` (static syntax gate).
2. Exact-v2 compile/lowering of the `(1024, 128)` specialization.
3. Camodel numerical check: random float32 inputs vs `x + y`.
4. Real-NPU run under `if __name__ == "__main__":` comparing with
   `torch.allclose`; the guard keeps import side-effect free.
