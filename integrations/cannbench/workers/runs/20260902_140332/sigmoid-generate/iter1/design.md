# Design: Sigmoid (pyasc asc2 CANNBench kernel)

## Algorithm

`y = 1 / (1 + exp(-x))`, all compute promoted to f32 inside the kernel.

**Naive formula is safe here.** IEEE saturation handles extremes correctly:
- `x → +∞`: `exp(-x) → 0` → `y = 1.0`
- `x → -∞`: `exp(-x) → +inf` → `1/(inf+1) = 0.0`
- `x = NaN`: propagates through as NaN
- `x = 0`: `exp(0) = 1` → `y = 0.5`

The numerically-stable split (`exp(min(x,0)) / (1+exp(-|x|))`) is unnecessary — the
reference module using the naive formula scores 100% accuracy across all 20 cases,
including f32 ranges [-88,88] and [-100,100] where `exp(100)` overflows f32 to inf but
IEEE gives the correct answer (0). For f16 [-65504,65504] the same holds after f32
promotion.

## Pinned-v2 API surface

| Operation | API |
|---|---|
| Global memory view | `asc2.global_tensor(ptr, [size])` (1-D) |
| Tile load | `asc2.copy_in(gm, [off], [TILE], real_shape=[n])` |
| Dtype promote | `x.to(asc.float32)` |
| Negate | `-xf` (unary) |
| Exponential | `asc2.exp(tile)` |
| Add scalar | `tile + 1.0` (scalar on right) |
| Divide | `asc2.div(1.0, tile)` |
| Cast back | `y.to(x.dtype)` via `asc2.cast` |
| Tile store | `asc2.copy_out(tile, gm, [off], real_shape=[n])` |
| Grid-stride loop | `asc2.range(block_idx(), num_tiles, block_num(), unroll_factor=2)` |

No `tensor`/`load`/`store` legacy spelling. Kernel params: `asc.GlobalAddress` for
pointers, `int` for runtime sizes, `asc.ConstExpr[int]` for tile dimensions.

## All 20 cases

| # | Shape | numel | dtype | range | Tail? |
|---|---|---|---|---|---|
| 1 | 1024×1024 | 1.05M | f16 | [-1,1] | no |
| 2 | 2048×2048 | 4.19M | f32 | [-2,2] | no |
| 3 | 4096×4096 | 16.78M | bf16 | [-3,3] | no |
| 4 | 8192×8192 | 67.11M | f16 | [-10,10] | no |
| 5 | 8192×8192 | 67.11M | f32 | [-100,100] | no |
| 6 | 1023×1023 | 1.05M | bf16 | [-0.1,0.1] | yes |
| 7 | 1009×1021 | 1.03M | f16 | [-1,2] | yes |
| 8 | 1537×769 | 1.18M | f32 | [-5,10] | yes |
| 9 | 363×367×373 | 49.69M | bf16 | [-50,100] | yes |
| 10 | 2049×513 | 1.05M | f16 | [-65504,65504] | yes |
| 11 | 3×7×13×4001 | 1.09M | f32 | [-88,88] | yes |
| 12 | 1000003 | 1.00M | bf16 | [-inf,inf] | yes |
| 13 | 11×13×17×67×67 | 10.91M | f32 | [nan,nan] | yes |
| 14 | 3×7×11×13×1009 | 3.03M | f16 | [0,0] | yes |
| 15 | 512×2049 | 1.05M | f32 | [-0.5,0.5] | yes |
| 16 | 255×8193 | 2.09M | bf16 | [-1,3] | yes |
| 17 | 4097×511 | 2.09M | f16 | [-1000,1000] | yes |
| 18 | 2×511×2049 | 2.09M | f32 | [-0.2,0.2] | yes |
| 19 | 4×255×2049 | 2.09M | bf16 | [-3,6] | yes |
| 20 | 2×3×17×1024×101 | 10.55M | f32 | [-20,40] | yes |

All cases flatten to 1-D via `numel()`. No broadcasting, no attributes.

## Tiling strategy

Two tile sizes selected by host dispatch:
- **Wide tile `TILE=3072`**: when `numel >= 72 * 3072 = 221184`. Covers cases 1–20
  (minimum numel ~1M >> 221K).
- **Narrow tile `TILE=1024`**: fallback for shapes below the threshold (none in this
  set, but included for robustness).

Grid-stride loop: `asc2.range(block_idx(), num_tiles, block_num(), unroll_factor=2)`.
Cores: `min(72, num_tiles)`. 72 vector cores on Ascend 950PR.

## Tail handling

```python
n = tile_size if off + tile_size <= size else size - off
```
Passed as `real_shape=[n]` to both `copy_in` and `copy_out`. No host-side padding,
no shape mutation. Ternary expression for the tail condition is supported syntax.

## UB budget

Op chain visible f32 tiles: `xf`, `neg_xf`, `exp_val`, `denom`, `result` = 5 tiles.
With `unroll_factor=2` and measured 1.6× overhead factor:

| TILE | Estimated UB | Status |
|---|---|---|
| 1024 | ~52K B | safe |
| 2048 | ~156K B (measured: 155648) | safe |
| 3072 | ~233K B | safe (< 253952) |
| 4096 | ~311K B | **OVERFLOW** |

**Selected: TILE=3072 (wide), TILE=1024 (narrow).** Both fit within the 253952 B
budget.

## Numerical risks

| Risk | Mitigation |
|---|---|
| `exp(-x)` overflow for large negative x | IEEE `inf` → `1/inf = 0`, matches golden |
| f16 denormals near 0 | f32 promotion gives full mantissa range |
| NaN propagation (case 13) | IEEE NaN passes through exp/div correctly |
| ±inf (case 12) | bf16 ±inf → f32 ±inf → sigmoid gives 1.0/0.0 |
| Catastrophic cancellation | None — no subtraction of near-equals in this formula |
| Precision thresholds | f32 internal compute exceeds f16 threshold 2^-10, bf16 2^-7, f32 2^-13 |

## Anti-cheat constraints

- All math in `@asc2.jit` kernel on NPU
- `torch` used only for: `torch.empty_like(x)`, `x.numel()`, `x.is_contiguous()`,
  `x.contiguous()`, `x.dtype`
- No `torch.sigmoid`, no tensor arithmetic, no `.to(dtype)` on device tensors
- No output caching, no `data_ptr` reuse
- Output is a fresh contiguous tensor with golden's exact shape/dtype

## Host dispatch

```python
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    ensure_npu_platform()
    x = x.contiguous() if not x.is_contiguous() else x
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0: return out
    if size >= _MAX_CORES * _WIDE_TILE:
        num_tiles = asc.ceildiv(size, _WIDE_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _sigmoid_kernel[cores](x, out, size, num_tiles, _WIDE_TILE)
    else:
        num_tiles = asc.ceildiv(size, _NARROW_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _sigmoid_kernel[cores](x, out, size, num_tiles, _NARROW_TILE)
    return out
```

## Syntax compliance (per pyasc-syntax-constraints)

- `for` loop with `asc2.range()`, no `break`/`continue`/`return`
- Ternary `if` for tail condition (supported)
- No `print`, no imports, no `try/except`, no `lambda`, no `math.*`
- Scalars on the right of tile operations (`exp_val + 1.0`, not `1.0 + exp_val`)
- Device functions may `return`; kernel writes to output tensors
- `asc.ConstExpr[int]` for `tile_size` parameter (compile-time)

## Local validation ladder

1. `python3 -m py_compile candidate.py` — syntax gate only
2. Worker static contract check — schema, imports, anti-cheat
3. Exact-v2 compile gate for all 20 cases — `verified-local-compile` label
4. **No numerical claims** without camodel (`verified-camodel`) or CANNBench
   (`verified-cannbench`) execution

DESIGN_DONE
