# Kernel Design: {kernel_name}

## 1. Operation description

**Mathematical definition**: {describe the operation, e.g., out = x + y}

**Input tensors**: {list input tensors with shapes and dtypes}

**Output tensors**: {list output tensors with shapes and dtypes}

## 2. pyasc API selection

| Purpose | API | Module |
|---------|-----|--------|
| Kernel decorator | `@asc2.jit(always_compile=True)` | `asc2` |
| Global memory wrapper | `asc2.tensor(ptr, [shape])` | `asc2` |
| Load tile from GM | `asc2.load(gm, [tile_shape], offsets=[...])` | `asc2` |
| Store tile to GM | `asc2.store(tile, gm, offsets=[...])` | `asc2` |
| Tile loop | `asc2.range(n)` | `asc2` |
| Block index | `asc2.block_idx()` | `asc2` |
| Computation | {e.g., `x + y` or `asc2.abs(x)`} | `asc2` / operators |
| Kernel params | `asc.GlobalAddress`, `asc.ConstExpr[int]` | `asc` |
| Tiling math | `asc.ceildiv(a, b)` | `asc` |

## 3. Multi-core strategy

- **Core count**: {e.g., 16}
- **Tile size**: {e.g., 128 elements}
- **Tiles per block**: `asc.ceildiv(num_tiles, core_num)`
- **Work distribution**: Each core processes `tile_per_block` tiles starting from `block_idx() * tile_size * tile_per_block`

## 4. Verification plan

- **Backend**: Model (simulator) and/or NPU
- **Reference**: {e.g., numpy x + y}
- **Tolerance**: {e.g., atol=1e-5, rtol=1e-5}
- **Test sizes**: {list of sizes to test}

## 5. Syntax compliance check

- [ ] All constructs inside `@asc2.jit` are in the supported set
- [ ] No unsupported syntax (print, break, continue, lambda, etc.)
- [ ] Kernel does not return a value
- [ ] All device function returns are top-level only
- [ ] Variables initialized before conditional use
