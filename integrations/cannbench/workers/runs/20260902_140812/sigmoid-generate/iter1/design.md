# Design: Sigmoid (CANNBench)

## Algorithm

Numerically stable sigmoid using the cancellation-free identity:

```
sigmoid(x) = exp(min(x, 0)) / (1 + exp(-|x|))
```

- For x ≥ 0: numerator = 1, denominator = 1 + exp(-x) (x ≥ 0 so -x ≤ 0 → no overflow)
- For x < 0: numerator = exp(x) (< 1, underflows to 0), denominator = 1 + exp(x) (x < 0 → no overflow)
- **exp() never sees a positive argument** → no f32 overflow possible even for |x| up to 65504.

Compute pipeline (all in f32, promoted from input dtype):

1. `xf = x.to(f32)`
2. `xneg = asc2.minimum(xf, 0.0)` — numerator exponent (always ≤ 0)
3. `xabs = asc2.abs(xf)`
4. `neg_xabs = -xabs` — denominator exponent (always ≤ 0)
5. `num = asc2.exp(xneg)`
6. `exp_neg_abs = asc2.exp(neg_xabs)`
7. `den = exp_neg_abs + 1.0`
8. `yf = asc2.div(num, den)`
9. `y = yf.to(input_dtype)` — cast back for store

Special inputs propagate through IEEE arithmetic without branching:
- `sigmoid(0) = exp(0)/(1+exp(0)) = 1/2 = 0.5` ✓
- `sigmoid(+inf) = 1/(1+0) = 1` ✓ (exp(-inf)→0 in IEEE)
- `sigmoid(-inf) = 0/(1+0) = 0` ✓ (exp(-inf)→0)
- `sigmoid(NaN) = NaN` ✓ (IEEE propagation)

## Pinned-v2 API Surface

| API | Usage |
|-----|-------|
| `asc2.global_tensor(ptr, [size])` | 1-D GM view of input/output, always 1-D |
| `asc2.copy_in(gm, [off], [tile_size], real_shape=[n])` | Load tile with tail handling |
| `asc2.copy_out(tile, gm, [off], real_shape=[n])` | Store tile with tail handling |
| `asc2.range(start, end, step, unroll_factor=2)` | Grid-stride loop over tiles |
| `asc2.block_idx()` / `asc2.block_num()` | Core identity and count |
| `asc2.minimum`, `asc2.abs`, `asc2.exp`, `asc2.div` | Elementwise ops |
| `.to(dtype)` | Cast tile between dtypes |
| `asc.ConstExpr[int]` | Compile-time tile size |

**No `gm_barrier`** on this build (TypeError). No `parallel` kwarg on `asc2.range`. Only `unroll_factor` is portable.

## All 20 Cases

| Case | Shape | dtype | Elements | Range | Key concern |
|------|-------|-------|----------|-------|-------------|
| 1 | 1024² | f16 | 1,048,576 | [-1,1] | Baseline |
| 2 | 2048² | f32 | 4,194,304 | [-2,2] | Baseline f32 |
| 3 | 4096² | bf16 | 16,777,216 | [-3,3] | Large bf16 |
| 4 | 8192² | f16 | 67,108,864 | [-10,10] | Largest shape, f16 sat zone |
| 5 | 8192² | f32 | 67,108,864 | [-100,100] | Naive exp overflow at x=-100 |
| 6 | 1023² | bf16 | 1,046,529 | [-0.1,0.1] | Non-power-of-2 |
| 7 | 1009×1021 | f16 | 1,030,189 | [-1,2] | Asymmetric primes |
| 8 | 1537×769 | f32 | 1,181,953 | [-5,10] | Non-power-of-2 |
| 9 | 363×367×373 | bf16 | 49,691,433 | [-50,100] | 3D, ~50M |
| 10 | 2049×513 | f16 | 1,051,137 | [-65504,65504] | Extreme f16 → f32 promotion |
| 11 | 3×7×13×4001 | f32 | 1,092,273 | [-88,88] | Near f32 exp limit |
| 12 | 1000003 | bf16 | 1,000,003 | [-inf,inf] | Inf values, 1-D |
| 13 | 11×13×17×67² | f32 | 10,912,759 | [NaN,NaN] | All-NaN input |
| 14 | 3×7×11×13×1009 | f16 | 3,030,027 | [0,0] | All zeros → 0.5 |
| 15 | 512×2049 | f32 | 1,049,088 | [-0.5,0.5] | Normal |
| 16 | 255×8193 | bf16 | 2,089,215 | [-1,3] | Non-power-of-2 |
| 17 | 4097×511 | f16 | 2,093,567 | [-1000,1000] | Large f16 range |
| 18 | 2×511×2049 | f32 | 2,094,078 | [-0.2,0.2] | 3-D small range |
| 19 | 4×255×2049 | bf16 | 2,089,980 | [-3,6] | 3-D |
| 20 | 2×3×17×1024×101 | f32 | 10,549,248 | [-20,40] | 5-D |

Element counts: ~1M to ~67M. All flattened to 1-D, no rank-specific logic.

## Tiling Strategy

- Flatten all inputs to 1-D (`x.numel()`) on the host side.
- Two tile sizes selected by host dispatch:
  - **Wide tile = 2048**: when `size >= 72 * 2048 = 147,456` elements (covers all 20 cases since min is ~1M)
  - **Narrow tile = 1024**: fallback for tiny tensors (not needed for any of the 20 cases, but safe fallback)
- Grid-stride: `for t in asc2.range(block_idx, num_tiles, block_num, unroll_factor=2)`
- Core count: `cores = min(72, num_tiles)`

## Tail Handling

```python
n = tile_size if off + tile_size <= size else size - off
x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
```

- Ternary `if` is supported in `@asc2.jit`.
- `real_shape=[n]` ensures partial tiles are loaded/stored correctly without host padding.
- No special handling needed for non-power-of-2 shapes (cases 6, 7, 8, 10, 16, 17, etc.) — the flatten+grid-stride approach is rank-agnostic.

## UB Budget

Approx. 10 visible f32 tile values in the safe sigmoid chain:
`xf, xneg, xabs, neg_xabs, num, exp_neg_abs, den, yf` + the input tile `x` (f16/bf16) + output tile `y`.

At TILE=2048 with `unroll_factor=2`:
- Raw: 10 × 4 × 2048 × 2 = 163,840 B
- With 1.6× overhead: ~262,144 B → **slightly over** 253,952 B budget

Mitigation: reduce to ~9 live values by fusing or reuse:
- `neg_xabs` and `xneg` are related: `neg_xabs = -abs(x) = min(x,0) - max(x,0)`. Could compute `xneg` once and derive `neg_xabs = xneg - abs(xneg)` ... no, `min(x,0) = x` when x<0, and `-|x| = x` when x<0 too. Actually `min(x,0) == -|x|` when x<0, and `min(x,0)=0, -|x|=-x` when x≥0. They're NOT the same.
- We need both `exp(min(x,0))` and `exp(-|x|)`. These are two separate exp calls.
- Alternative: only compute `exp(-|x|)` once, then:
  - `num = exp(-|x|)` if x<0, else `1.0`
  - `den = exp(-|x|) + 1.0` always
  - This uses `asc2.where(xf >= 0.0, 1.0, exp_neg_abs)` for numerator → 1 fewer exp, but adds a where + comparison tile. Net: similar count.

Revised approach — **single exp, where-based**:
1. `xf = x.to(f32)`
2. `xabs = asc2.abs(xf)` — tile (f32)
3. `neg_xabs = -xabs` — tile (f32)
4. `e = asc2.exp(neg_xabs)` — tile (f32), always in [0,1]
5. `den = e + 1.0` — tile (f32), always in [1,2]
6. `num = asc2.where(xf >= 0.0, 1.0, e)` — tile (f32)
7. `yf = num / den` — tile (f32)
8. `y = yf.to(input_dtype)` — cast back

That's 8 distinct tile values. At TILE=2048: 8 × 4 × 2048 × 2 × 1.6 = 209,715 B. **Under budget.** ✓

Comparison `xf >= 0.0` produces a bool tile (1 byte/element), negligible.

## Numerical Risks

| Risk | Mitigation |
|------|-----------|
| exp overflow for large \|x\| | exp always sees non-positive arg → no overflow |
| Catastrophic cancellation | No subtraction of near-equal quantities; `den = e + 1.0` where e ∈ [0,1], so den ∈ [1,2], well-conditioned |
| Division by zero | den ≥ 1.0 always, no risk |
| f16/bf16 precision loss | All compute in f32, cast back only on store |
| NaN propagation (case 13) | IEEE: NaN through abs→NaN, exp(NaN)=NaN, where cond is false→selects e=NaN, NaN/den=NaN ✓ |
| Inf propagation (case 12) | inf→abs→inf→neg→-inf→exp(-inf)=0, where selects 0→0/(0+1)=0 ✓; -inf→abs→inf→same path→0 ✓ |
| x=0 (case 14) | abs(0)=0, exp(0)=1, where: 0>=0→1.0, 1.0/2.0=0.5 ✓ |
| Extreme f16 range [-65504,65504] (case 10) | Promoted to f32, exp(-65504) underflows to 0, where selects 0→0/1=0 ✓ |
| f32 threshold 2^-13 ≈ 1.22e-4 | All ops in f32 with IEEE rounding; safe form has no precision loss vs torch.sigmoid |

## Anti-Cheat Constraints

- **All compute inside `@asc2.jit` kernel** — no torch math ops
- **torch usage limited to**: `torch.empty_like`, `.numel()`, `.shape`, `.dtype`, `.is_contiguous()`, `.contiguous()`
- **No** `torch.sigmoid`, `x.sigmoid()`, `.to(dtype)` on device tensors, tensor arithmetic
- **No caching** — fresh `torch.empty_like` each call, data pointers rotate
- **Output**: contiguous NPU tensor, exact shape/dtype match with golden
- **Import `ensure_npu_platform`** and call before any kernel launch

## Local Validation Ladder

1. **`python3 -m py_compile candidate.py`** — Python syntax gate (no imports resolved)
2. **Worker static contract check** — verifies callable name, import structure, anti-cheat rules
3. **Exact-v2 local compile gate** — JIT compile all 20 cases through pinned pyasc v2; catches UB overflow, API misuse, unsupported syntax
4. **camodel execution** (different model) — numerical evidence against golden
5. **CANNBench on real NPU** — final acceptance + performance oracle

Evidence labels: `verified-local-compile` (step 3), `verified-camodel` (step 4), `verified-cannbench` (step 5). No label promotion without numerical evidence.

---

DESIGN_DONE
