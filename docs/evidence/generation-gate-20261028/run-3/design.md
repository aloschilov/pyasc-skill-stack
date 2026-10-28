# Vector Add Kernel Design

## Operator

Elementwise add of two contiguous float32 tensors of logical size `N = 1024`:

```
out[i] = x[i] + y[i],  0 <= i < N
```

## Strategy

The tensors are 1-D and contiguous, so the problem flattens naturally to a
single logical dimension. We use the standard current-v2 elementwise pattern:

1. Construct a rank-1 GM descriptor per operand with `asctile.global_tensor`
   inside the kernel body (never on the host).
2. Grid-stride over tiles with `asctile.range(asctile.block_idx(),
   asctile.ceildiv(size, tile_size), asctile.block_num())` so the tile
   count is spread across the whole core grid.
3. Per tile, `copy_in` both operands at `[offset]` with shape
   `[tile_size]`, add, and `copy_out` the result.

## Tiling

- `tile_size = 256` float32 elements = 1024 bytes per operand tile, well
  within UB and a clean DMA size.
- `N = 1024` divides exactly into 4 tiles of 256; the tail path
  (`real_shape=[valid]`, `pad_value=0`) is still written so the kernel is
  correct for any size, not just 1024. The pad value `0` is the add-neutral
  identity: padded lanes compute `0 + 0` and are excluded from the store by
  `real_shape`.
- `CORE_NUM = 8` launch grid via `kernel[CORE_NUM](...)` with no stream
  argument; tiles are distributed across cores by the grid-stride range.

## Syntax constraints honored

- `@asctile.jit` device code only; no `print`/`break`/`continue`/`lambda`,
  no imports or host I/O inside the kernel.
- `tile_size` is `asctile.ConstExpr[int]` because it sizes the local tiles.
- `size` stays a runtime scalar; only shapes/specialization constants are
  ConstExpr.
- The launched kernel writes outputs and returns nothing.

## Verification

Host-side guard under `if __name__ == "__main__":` allocates `x`, `y` as
float32 tensors of size 1024, launches the kernel, and checks the result
against `x + y` with `torch.allclose`. The guard ensures importing the
module does not execute the launch or the check.
