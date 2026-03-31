# Kernel Design: {kernel_name}

## 1. Operation description

**Mathematical definition**: {describe the operation, e.g., z = x + y}

**Input tensors**: {list input tensors with shapes and dtypes}

**Output tensors**: {list output tensors with shapes and dtypes}

## 2. pyasc API selection

| Purpose | API | Module |
|---------|-----|--------|
| Kernel decorator | `@asc.jit` | `asc` |
| Global memory | `asc.GlobalTensor`, `asc.GlobalAddress` | `asc.language.core` |
| Local memory | `asc.LocalTensor` | `asc.language.core` |
| Data transfer | `asc.data_copy` | `asc.language.basic` |
| Computation | {e.g., `asc.add`} | `asc.language.basic` |
| Sync | `asc.set_flag`, `asc.wait_flag` | `asc.language.core` |

## 3. Multi-core strategy

- **Core count**: {e.g., 8}
- **Work distribution**: {e.g., divide total elements equally across cores}
- **Block index**: `asc.get_block_idx()` to determine per-core offset

## 4. Buffer strategy

- **Buffer count**: {1 or 2 (double buffering)}
- **Tile count**: {number of tiles per core}
- **Memory positions**: {VECIN, VECOUT, etc.}

## 5. Sync strategy

- **Pipeline stages**: MTE2 (copy in) -> V (compute) -> MTE3 (copy out)
- **Events used**: {MTE2_V, V_MTE3, MTE3_MTE2}
- **Manual or auto**: {manual set_flag/wait_flag or auto_sync=True}

## 6. Verification plan

- **Backend**: Model (simulator) and/or NPU
- **Reference**: {e.g., torch x + y}
- **Tolerance**: {e.g., atol=1e-5}
- **Test shapes**: {list of shapes to test}

## 7. Syntax compliance check

- [ ] All constructs inside `@asc.jit` are in the supported set
- [ ] No unsupported syntax (print, break, continue, lambda, etc.)
- [ ] Kernel does not return a value
- [ ] All device function returns are top-level only
- [ ] Variables initialized before conditional use
