# Kernel Design: add (vadd)

## 1. Operation description

**Mathematical definition**: z = x + y (element-wise vector addition)

**Input tensors**:
- `x`: 1D float32 tensor, shape `(N,)` where N = 8 * 2048 = 16384
- `y`: 1D float32 tensor, shape `(N,)` same as x

**Output tensors**:
- `z`: 1D float32 tensor, shape `(N,)`, z[i] = x[i] + y[i]

## 2. pyasc API selection

| Purpose | API | Module |
|---------|-----|--------|
| Kernel decorator | `@asc.jit` | `asc` |
| Global memory | `asc.GlobalTensor`, `asc.GlobalAddress` | `asc.language.core` |
| Local memory | `asc.LocalTensor` | `asc.language.core` |
| Data transfer | `asc.data_copy` | `asc.language.basic` |
| Computation | `asc.add` | `asc.language.basic` |
| Sync | `asc.set_flag`, `asc.wait_flag` | `asc.language.core` |
| Block index | `asc.get_block_idx()` | `asc.language.core` |

## 3. Multi-core strategy

- **Core count**: 8
- **Work distribution**: total_length / USE_CORE_NUM elements per core
- **Block index**: `asc.get_block_idx()` to compute per-core offset

## 4. Buffer strategy

- **Buffer count**: 2 (double buffering)
- **Tile count**: 8 tiles per core
- **Total iterations**: TILE_NUM * BUFFER_NUM = 16 per core
- **Memory positions**: VECIN for x_local and y_local, VECOUT for z_local

## 5. Sync strategy

- **Pipeline stages**: MTE2 (copy in) -> V (compute) -> MTE3 (copy out)
- **Events**: MTE2_V, V_MTE3, MTE3_MTE2
- **Mode**: Manual set_flag/wait_flag (tutorial 01_add style)

## 6. Verification plan

- **Backend**: Model (simulator)
- **Reference**: `torch` element-wise addition: `x + y`
- **Tolerance**: default (torch.allclose defaults)
- **Test shapes**: `(16384,)` = 8 * 2048

## 7. Syntax compliance check

- [x] All constructs inside `@asc.jit` are in the supported set
- [x] No unsupported syntax (print, break, continue, lambda, etc.)
- [x] Kernel does not return a value
- [x] Variables initialized before conditional use (no conditional branches)
- [x] Only `for i in range(...)` loop used
