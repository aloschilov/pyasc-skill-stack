# Mish Operator Design

## Algorithm

`mish(x) = x * tanh(softplus(x))` where `softplus(x) = ln(1 + e^x)`.

Use the **cancellation-free** identity for `tanh(softplus(x))` (from task.md):

With `w = exp(-|x|)`:
- **x >= 0**: `tanh_sp = (1 + 2w) / (1 + 2w + 2w^2)`
- **x < 0**: `tanh_sp = (w^2 + 2w) / (w^2 + 2w + 2)`

Then `y = x * tanh_sp`. No `exp()` ever sees a positive argument; no `log(1+tiny)` appears.

## Pinned-v2 APIs

| Concern | Pinned API |
|---------|-----------|
| Global views | `asc2.global_tensor(ptr, [size])` |
| Load | `asc2.copy_in(gm, [off], [tile_size], real_shape=[n])` |
| Store | `asc2.copy_out(tile, gm, [off], real_shape=[n])` |
| Tile loop | `asc2.range(start, stop, step, unroll_factor=2)` — NO `parallel`, NO `gm_barrier` |
| Launch | `kernel[cores](...)`, `cores = min(72, num_tiles)` |
| ConstExpr | `tile_size: asc.ConstExpr[int]` for copy shape |
| Dtypes | Promote `xf = x.to(asc.float32)`, cast back `y.to(x.dtype)` |
| Element-wise | `asc2.abs`, `asc2.exp`, `asc2.where`, `tile >= 0.0`, arithmetic `+ - * /` |
| Runtime | `ensure_npu_platform()` before any NPU op |

All scalar operands on the RIGHT of tile operators (`tile + 1.0`, never `1.0 + tile`).

## All 20 Cases Coverage

| Metric | Coverage |
|--------|----------|
| dtypes | f16 (cases 1,4,7,10,14,17), f32 (2,5,8,11,13,15,18,20), bf16 (3,6,9,12,16,19) |
| ranks | 1D (12), 2D (1,2,3,4,5,6,7,8,10,15,16,17), 3D (9,18,19), 4D (11), 5D (13,14,20) |
| shapes | Contiguous + non-power-of-2 (1023, 1009, 1021, 363, 367, 373, 8193, 4001, 1000003) |
| values | Tiny ([-0.1,0.1]), moderate ([-1,1]), wide ([-65504,65504], [-1000,1000], [-inf,inf], nan) |
| edges | All-zero (14), inf/-inf (12), all-NaN (13), max-f16 range (10,17) |

**Strategy**: flatten to 1D (`x.numel()`), single kernel dispatch. Shape/dtype preserved via `torch.empty_like(x)` host-side. All 20 cases route through one code path — no specialization.

## Tiling

- **TILE = 1024** (primary) — fits UB budget for the Mish chain (see below).
- **Grid-stride**: `for t in asc2.range(block_idx, num_tiles, block_num, unroll_factor=2)`
- **Tails**: `n = tile_size if off + tile_size <= size else size - off`; passed as `real_shape=[n]`.
- **Core dispatch**: `cores = min(72, num_tiles)`. All cases have > 970 tiles (min numel ~1M), so 72 cores always saturated.

## UB Budget

15 visible f32 tiles in the compute chain:

```
xf          x.to(f32)
abs_xf      asc2.abs(xf)
neg_abs     -abs_xf
w           asc2.exp(neg_abs)
w2          w * w
two_w       w + w
two_w2      w2 + w2
pos_num     two_w + 1.0
pos_den     pos_num + two_w2
pos_th      pos_num / pos_den
neg_num     w2 + two_w
neg_den     neg_num + 2.0
neg_th      neg_num / neg_den
th          asc2.where(cond, pos_th, neg_th)
y           xf * th
```

Plus 1 bool tile (`xf >= 0.0`) — negligible cost (1 byte/element).

**Measured UB model**: `15 * 4 * TILE * unroll(2) * 1.6x = 192 * TILE`

| TILE | Estimated UB | Fits 253952? |
|------|-------------|--------------|
| 512 | 98,304 B | Yes |
| 1024 | 196,608 B | Yes |
| 2048 | 393,216 B | No |

**Selected TILE = 1024.** If UB overflow at compile time, fall back to 512.

## Numerical Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `exp(x)` overflow for x > 88 (f32) / x > 11 (f16) | Never called with positive arg; always `exp(-\|x\|)` with `-` sign |
| `log(1 + tiny)` flushing to 0 | Never called; algebraic cancellation eliminates `ln` |
| Catastrophic cancellation in subtraction | No subtraction of near-equal quantities in the formulas |
| IEEE special values (inf, NaN, -0) | Propagate naturally: w=exp(-\|inf\|)=0 yields correct limits; NaN arithmetic yields NaN |
| Large x (e.g., x=65504 in f16) | w underflows to 0, result = x * 1.0 = x — exact |
| x = -inf | w = 0, num = 0, den = 2, th = 0, y = -inf * 0 = NaN. Matches golden (IEEE). |
| x = +inf | w = 0, num = 1, den = 1, th = 1, y = inf * 1 = inf. Matches golden. |
| x = NaN | All ops propagate NaN. Matches golden. |
| Precision thresholds | f16: 2^-10, bf16: 2^-7, f32: 2^-13. The cancellation-free formulas have relative error well within these bounds since all intermediate quantities remain O(1) or smoothly bounded. |

## Anti-Cheat Constraints

| Rule | Compliance |
|------|-----------|
| All numerics in `@asc2.jit` kernel | Yes — Mish math entirely in kernel |
| No `torch.*` compute ops | No torch.mul, torch.nn.functional, tensor arithmetic, .sigmoid(), .to(dtype) on data |
| torch used only for alloc/metadata/view | `torch.empty_like(x)`, `.numel()`, `.is_contiguous()`, `.contiguous()`, `x.dtype` only |
| No output caching by `data_ptr` | Fresh `torch.empty_like` each call |
| Output shape/dtype matches golden exactly | `torch.empty_like(x)` ensures this |
| Contiguous NPU tensors returned | `torch.empty_like` returns contiguous; kernel writes contiguous |
| `ensure_npu_platform()` called first | Yes, first line of `mish()` wrapper |

## Local Validation Ladder

1. **Syntax gate**: `python3 -m py_compile candidate.py` — pure Python AST check.
2. **Static contract review**: verify all pinned-v2 APIs, no forbidden syntax (no `break`, `continue`, `return`, `print`, `import` inside JIT; `asc2.range` not `range`; scalars on right; `ConstExpr` for tile_size).
3. **Exact-v2 local compile gate**: invoke the local compile harness per `pyasc-cannbench-kernel` skill references. Compile failure = measured feedback → fix and retry. Never drop cases.
4. **Evidence label**: result is `verified-local-compile` at best until camodel/cannbench evidence arrives.

DESIGN_DONE
