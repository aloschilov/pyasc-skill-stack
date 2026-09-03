# Design: MaskedScale pyasc asc2 kernel

## Algorithm

`y = x * mask * scale`, elementwise. Both inputs and the output share the same
shape. Flatten to 1-D on the host (the operation is shape-agnostic). Host
dispatches dtype combos; one kernel source covers all via JIT dtype
specialization.

## Pinned-v2 API surface

| Step | API |
|------|-----|
| Global memory | `asc2.global_tensor(ptr, [size])` — 1-D |
| Load tile | `asc2.copy_in(gm, [off], [TILE], real_shape=[n])` |
| Int8 mask promotion | `asc2.cast(m, asc.float16)` then `.to(asc.float32)` |
| Float mask promotion | `m.to(asc.float32)` |
| Compute | `xf * mf * scale` — scalar on right |
| Store | `asc2.copy_out(y.to(x_dtype), out_gm, [off], real_shape=[n])` |

## All 20 cases

| # | shape (1-D numel) | x dtype | mask dtype | scale | notes |
|---|---|---|---|---|---|
| 1 | 1048576 | f16 | int8 | 1.0 | |
| 2 | 4194304 | f32 | uint8 | 1.0 | uint8 fixup |
| 3 | 16777216 | bf16 | f16 | 1.0 | mixed-16bit: cast mask straight to f32 |
| 4 | 67108864 | f16 | f32 | 0.5 | |
| 5 | 67108864 | f32 | int8 | 2.0 | |
| 6 | 1046529 | bf16 | uint8 | -1.0 | uint8 fixup |
| 7 | 1030189 | f16 | int8 | 0.0 | mask range [0,127] — signed int8, no fixup |
| 8 | 1181953 | f32 | f16 | 10.0 | |
| 9 | 4977117 | bf16 | f32 | 1.5 | |
| 10 | 1049337 | f16 | uint8 | 1.0 | mask [0,255], uint8 fixup |
| 11 | 1093219 | f32 | int8 | 1.0 | 4-D |
| 12 | 1000007 | bf16 | bf16 | inf | 1-D, inf propagation |
| 13 | 10521419 | f32 | f32 | nan | 5-D, nan propagation |
| 14 | 3074517 | f16 | int8 | 1.0 | x all zeros, 5-D |
| 15 | 1049088 | f32 | uint8 | 1.0 | uint8 fixup |
| 16 | 2089215 | bf16 | int8 | 1.2 | |
| 17 | 2093057 | f16 | uint8 | -0.5 | uint8 fixup |
| 18 | 2100222 | f32 | f16 | 0.75 | 3-D |
| 19 | 2099232 | bf16 | f32 | 3.0 | 3-D |
| 20 | 10554624 | f32 | int8 | 1.0 | 5-D |

uint8 cases: 2, 6, 10, 15, 17 — host `mask.view(torch.int8)`, pass `is_uint8=1`
flag, kernel applies +256 fixup in f32 via `asc2.where`.

## Tiling & tails

- 1-D grid-stride: `for t in asc2.range(block_idx, num_tiles, block_num, unroll_factor=2)`
- Tail: `n = TILE if off + TILE <= size else size - off`; `real_shape=[n]` on
  both copy_in and copy_out.
- Two kernel variants dispatched by host:
  - `_masked_scale_kernel` — normal (int8 or float mask, no uint8 fixup)
  - `_masked_scale_kernel_uint8` — adds `asc2.where(mf < 0.0, mf + 256.0, mf)` after cast
- Both accept `is_uint8` as `asc.ConstExpr[int]`? No — simpler: two separate
  `@asc2.jit` functions to avoid branching cost on every tile.

## UB budget

| f32 tile | bytes per | count | subtotal |
|----------|-----------|-------|----------|
| xf | 4*TILE | 1 | 4T |
| mf (after cast) | 4*TILE | 1 | 4T |
| product xf*mf | 4*TILE | 1 | 4T |
| scaled (xf*mf*scale) | 4*TILE | 1 | 4T |
| where-fixup tile | 4*TILE | 1 (uint8 only) | 4T |
| **visible @ TILE=2048** | | | 32768 (normal) / 40960 (uint8) |
| **x1.6 measured** | | | ~52428 / ~65536 |
| **x2 unroll** | | | ~104857 / ~131072 |

Budget 253952 B. TILE=2048 fits comfortably for both variants.

## Numerical risks

- **inf/nan** (cases 12, 13): IEEE `*` propagates correctly — no special casing.
- **Zero scale** (case 7): `0.0 * anything = 0.0` in IEEE, even with nan input
  — golden uses same semantics, no issue.
- **Large mask values** (case 10 mask up to 255 via uint8): f32 represents
  exactly, no precision loss.
- **f16/bf16 mixed** (cases 3, 6, 16): both 16-bit tiles are ONLY cast to f32,
  never touched by arithmetic or comparisons at 16-bit.
- **Precision thresholds**: f16 needs MERE < 2^-10 (~9.8e-4), f32 < 2^-13, bf16
  < 2^-7. Single f32 multiply chain is exact for int8/uint8 masks and within
  1 ulp for float masks — well within all thresholds.

## Anti-cheat constraints

- All compute in `@asc2.jit` kernels — no torch arithmetic on tensors.
- torch used only for: `torch.empty_like(x)` allocation, `.numel()`, `.shape`,
  `.dtype`, `.contiguous()`, `.view(torch.int8)` for uint8 reinterpret.
- `ensure_npu_platform()` called first.
- Output is contiguous, same shape and dtype as x.

## Local validation ladder

1. `python3 -m py_compile candidate.py` — syntax gate.
2. Worker static contract check — callable signature, imports, anti-cheat.
3. exact-v2 compile gate: all 20 dtype/constexpr combos must JIT-compile via
   pinned pyasc v2 (`verified-local-compile`).
4. No numerical claim until camodel or CANNBench evaluation.

## Host dispatch pseudocode

```
def masked_scale(x, mask, scale=1.0):
    ensure_npu_platform()
    x = x.contiguous(); mask = mask.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0: return out

    is_uint8 = mask.dtype == torch.uint8
    if is_uint8:
        mask = mask.view(torch.int8)   # no-copy reinterpret

    TILE = 2048
    num_tiles = ceildiv(size, TILE)
    cores = min(72, num_tiles)

    if is_uint8:
        _masked_scale_uint8[cores](x, mask, out, size, num_tiles, scale, TILE)
    else:
        _masked_scale_normal[cores](x, mask, out, size, num_tiles, scale, TILE)
    return out
```

DESIGN_DONE
