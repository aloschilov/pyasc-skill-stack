# Kernel Design: {kernel_name}

## 1. Operation description

**Mathematical definition**: {describe the operation, e.g., out = x + y}

**Input tensors**: {list input tensors with shapes and dtypes}

**Output tensors**: {list output tensors with shapes and dtypes}

## 2. pyasc API selection

| Purpose | API | Module |
|---------|-----|--------|
| Kernel decorator | `@asctile.jit(always_compile=True)` | `asctile` |
| Global memory wrapper | `asctile.global_tensor(ptr, [shape])` | `asctile` |
| Load tile from GM | `asctile.copy_in(gm, offsets, tile_shape, real_shape=...)` | `asctile` |
| Store tile to GM | `asctile.copy_out(tile, gm, offsets, real_shape=...)` | `asctile` |
| Tile loop | `asctile.range(n)` | `asctile` |
| Block index | `asctile.block_idx()` | `asctile` |
| Computation | {e.g., `x + y` or `asctile.abs(x)`} | `asctile` / operators |
| Kernel params | `asc.GlobalAddress`, `asc.ConstExpr[int]` | `asc` |
| Tiling math | `asctile.ceildiv(a, b)` | `asctile` |

## 3. Multi-core strategy

- **Core count**: {e.g., 72 on Ascend950PR_9599}
- **Tile size**: {e.g., 128 elements}
- **Core count**: `min(platform_cores, num_tiles)`
- **Work distribution**: grid-stride from `block_idx()` with step `block_num()`

## 4. Tails and memory budget

- **Physical tile shape / bytes**: {shape, dtype, alignment}
- **Logical tail (`real_shape`)**: {formula}
- **Padding identity**: {0, 1, +/-inf, or other justified value}
- **Live UB estimate**: {temporaries * bytes * buffering}

## 5. Verification plan

- **Backend**: Model (simulator) and/or NPU
- **Reference**: {e.g., numpy x + y}
- **Tolerance**: {e.g., atol=1e-5, rtol=1e-5}
- **Test sizes**: {list of sizes to test}

## 6. Syntax compliance check

- [ ] All constructs inside `@asctile.jit` are in the supported set
- [ ] No unsupported syntax (print, break, continue, lambda, etc.)
- [ ] Kernel does not return a value
- [ ] All device function returns are top-level only
- [ ] Variables initialized before conditional use
