# Kernel Design: vector_add

## 1. Operation description

**Mathematical definition**: `out = x + y`, elementwise over a flat float32
vector of size 1024.

**Input tensors**:
- `x`: `[1024]`, float32, contiguous
- `y`: `[1024]`, float32, contiguous

**Output tensors**:
- `out`: `[1024]`, float32, contiguous

## 2. pyasc API selection

| Purpose | API | Module |
|---------|-----|--------|
| Kernel decorator | `@asctile.jit(always_compile=True)` | `asctile` |
| Global memory wrapper | `asctile.global_tensor(ptr, [size])` | `asctile` |
| Load tile from GM | `asctile.copy_in(gm, [offset], [tile_size], real_shape=[valid], pad_value=0)` | `asctile` |
| Store tile to GM | `asctile.copy_out(tile, gm, [offset], real_shape=[valid])` | `asctile` |
| Tile loop | `asctile.range(block_idx(), num_tiles, block_num())` | `asctile` |
| Block index | `asctile.block_idx()` | `asctile` |
| Computation | `x + y` (tile operator `+`) | `asctile` / operators |
| Kernel params | `asc.GlobalAddress`, `asc.ConstExpr[int]` | `asc` |
| Tiling math | `asctile.ceildiv(size, tile_size)` | `asctile` |

Only current-v2 surface is used: `asc2`/`tensor/load/store` and
`TileLocation` do not appear. `global_tensor`, `copy_in`, and `copy_out` are
called exclusively inside the `@asctile.jit` body on descriptors built from
`GlobalAddress` parameters, never at host level.

## 3. Multi-core strategy

- **Platform core count**: 72 (Ascend950PR_9599)
- **Tile size**: 256 elements (1 KB per float32 tile)
- **Tiles**: `ceildiv(1024, 256) = 4`
- **Launch core count**: `min(CORE_NUM, num_tiles) = 4`
- **Work distribution**: grid-stride from `asctile.block_idx()` with step
  `asctile.block_num()`; cores whose index is at or past `num_tiles` simply
  receive zero loop iterations, so any core count is safe.
- **Launch syntax**: `add_kernel[core_num](x, y, out, size, TILE_SIZE)` —
  bracket subscript launch with no stream argument.

## 4. Tails and memory budget

- **Physical tile shape / bytes**: `[256]` float32 = 1024 bytes per tile,
  rank-1 so no 32-byte final-dimension constraint applies beyond the natural
  4-byte element alignment.
- **Logical tail (`real_shape`)**: `valid = min(size - offset, tile_size)`.
  For size 1024 and tile size 256 every tile is full (`valid == 256`), but
  the tail path is kept so the kernel stays correct for any size.
- **Padding identity**: `pad_value=0` — the neutral element for addition.
  Padded lanes compute `0 + 0` and are excluded from the store by
  `real_shape` on `copy_out`.
- **Live UB estimate**: 3 local tiles (`x`, `y`, `out`) x 1024 bytes =
  3072 bytes per core, far below UB capacity; no unroll buffering
  (`unroll_factor=1` default loop).

## 5. Verification plan

- **Reference**: `x + y` computed by torch on the host.
- **Check**: `torch.allclose(out, x + y, atol=1e-5, rtol=1e-5)`.
- **Data**: `torch.randn(1024, dtype=torch.float32)` with a fixed seed.
- **Guard**: verification runs only under `if __name__ == "__main__":`;
  importing the module executes nothing.
- **Gates beyond this file** (not run here): exact-v2 compile/lowering,
  camodel numerics, then real-NPU execution.

## 6. Syntax compliance check

- [x] All constructs inside `@asctile.jit` are in the supported set
      (assignments, `if`, `asctile.range` loop, operators, builtin `min`)
- [x] No unsupported syntax (no `print`, `break`, `continue`, `lambda`,
      `try`, nested defs, or imports inside the JIT function)
- [x] Kernel does not return a value
- [x] Variables are initialized before any conditional use
- [x] Offsets precede shape in every `copy_in` / `copy_out`
