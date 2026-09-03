# RmsNorm — Design

Runtime pin: **compiler-team/pyasc v2 @ `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`**.
This snapshot exports **`asctile`**; `import asc2` / any `asc2.*` call is **invalid**.
Every API reference below uses `asctile`. The reference sigmoid module embedded in
task.md (which already uses `asctile`) is the structural template.

## 1. Algorithm

**y = x / sqrt(mean(x²) + ε) · γ**, normalized independently over the last
dimension D of every row. With `S = numel(x) / D` rows and `inv_D = 1.0 / D`:

- `mean_sq = (Σ_{j} x_j²) · inv_D`
- `inv_rms = rsqrt(mean_sq + epsilon)`
- `y_i = x_i · inv_rms · γ_i`

Implemented as a **two-pass streaming kernel per row**:

- **Pass 1 (reduce):** stream D in `TILE_D` chunks → accumulate `Σx²` in an f32
  loop-carried scalar `acc`.
- **Per-row scalar compute:** `inv_rms = rsqrt(acc · inv_D + epsilon)`, broadcast
  to a full tile so the apply step is pure tile·tile arithmetic (sidesteps the
  "scalar must go on the right" placement rule entirely).
- **Pass 2 (apply):** re-stream D chunks, reload x **and** γ from GM, emit
  `y = x · inv_rms · γ` in f32, cast back to input dtype, store.

Two-pass reloads x (and loads γ) from GM twice. This deliberately trades GM
bandwidth for **UB headroom**: a single-pass design that tried to hold the
reduce accumulator *and* the apply chain alive at `TILE_D=2048` overflowed UB
(328 608 B > 253 952 B budget) on the prior iteration and failed all 20 cases.
Bandwidth is not the scoring bottleneck (profiler kernel-time vs aclnn
baseline is dominated by compute/launch overhead on these shapes), so the
reload trade is favorable.

## 2. Pinned-v2 API surface (asctile)

| Purpose | Call |
|---|---|
| JIT decorator | `@asctile.jit` |
| Global views (1-D) | `asctile.global_tensor(ptr, [size])` — x/out as `[S*D]`, γ as `[D]` |
| Load tile | `asctile.copy_in(gm, [offset], [TILE_D], real_shape=[n])` |
| Store tile | `asctile.copy_out(tile, gm, [offset], real_shape=[n])` |
| Cast | `tile.to(asc.float32)` / `tile.to(x.dtype)` |
| Reduce | `asctile.reduce_sum(tile)` → scalar |
| Scalar seed (loop-carried acc) | `acc = asctile.reduce_sum(asctile.full([1, 64], 0.0, dtype=asc.float32))` (a bare `acc = 0.0` fails codegen) |
| Scalar fold | `acc = acc + asctile.reduce_sum(xf * xf)` inside `asctile.range(...)` |
| Broadcast scalar→tile | `asctile.full([TILE_D], scalar_expr, dtype=asc.float32)` |
| Inv-sqrt | `asctile.rsqrt(tile)` |
| Tile ops used | `*` (tile·tile), `asctile.rsqrt`, `asctile.full`, `asctile.reduce_sum` |
| Grid-stride iteration | `asctile.range(asctile.block_idx(), S, asctile.block_num())` |
| Sequential inner iteration | `asctile.range(num_d_tiles)` (plain count-up) |
| Kernel param types | `asc.GlobalAddress` (ptrs), `int` (runtime sizes), `float` (scalars), `asc.ConstExpr[int]` (compile-time tile) |
| Host helper | `asc.ceildiv(a, b)` |

Constraints honored: **no** `break`/`continue`/early `return`, **no** imports,
**no** `print`, **no** `math.*`, **no** Python `range()` over runtime values,
**no** `gm_barrier` (rows are independent), **no** `parallel` (not accepted by
v2@030e9b2c). Outer/inner loops use **no `unroll_factor`** to keep UB
single-buffered.

## 3. Case matrix (all 20, S recomputed from leading dims)

Leading-dim product `S = ∏ shape[:-1]`; `D = shape[-1]`; `num_d_tiles = ceil(D / 1024)`.

| # | shape | S (rows) | D | dtype | ε | value range | num_d_tiles | key stress |
|---|---|---|---|---|---|---|---|---|
| 1 | [32,128,768] | 4096 | 768 | f16 | 1e-6 | [-1,1] | 1 | standard, D<tile |
| 2 | [32,128,1024] | 4096 | 1024 | f32 | 1e-6 | [-2,2] | 1 | D == tile exactly |
| 3 | [32,128,2048] | 4096 | 2048 | bf16 | 1e-6 | [-3,3] | 2 | 2-chunk D |
| 4 | [16,256,4096] | 4096 | 4096 | f16 | 1e-6 | [-10,10] | 4 | 4-chunk D |
| 5 | [8,512,8192] | 4096 | 8192 | f32 | 1e-6 | [-100,100] | 8 | large x², 8 chunks |
| 6 | [4,1023,4097] | 4092 | 4097 | bf16 | 1e-5 | [-5,5] | 5 | prime D tail |
| 7 | [63,67,1023] | 4221 | 1023 | f16 | 1e-8 | [-0.1,0.1] | 1 | tiny x, small ε |
| 8 | [16,511,2049] | 8176 | 2049 | f32 | 1e-4 | [-1,1] | 3 | prime D tail |
| 9 | [8,1021,4099] | 8168 | 4099 | bf16 | 1e-12 | [-0.5,0.5] | 5 | extreme ε |
| 10 | [33,127,769] | 4191 | 769 | f16 | 1e-6 | [-1,2] | 1 | prime D |
| 11 | [31,129,2049] | **3999** | 2049 | f32 | 1e-6 | [-50,100] | 3 | large range, prime D |
| 12 | [17,255,4097] | 4335 | 4097 | bf16 | 1e-6 | [-3,6] | 5 | prime D |
| 13 | [7,1009,1021] | 7063 | 1021 | f16 | 1e-7 | [-1,1] | 1 | prime D, D<tile |
| 14 | [11,367,373] | 4037 | 373 | f32 | 1e-5 | [-10,10] | 1 | short D |
| 15 | [1000003,2] | **1000003** | 2 | bf16 | 1e-6 | [None,None] | 1 | massive S, tiny D |
| 16 | [11,13,17,67] | 2431 | 67 | f16 | 1e-8 | [None,None] | 1 | tiny D, 4-D |
| 17 | [3,7,11,4096] | 231 | 4096 | f32 | 1e-4 | [0,0] | 4 | all-zero rows |
| 18 | [2,511,8192] | 1022 | 8192 | bf16 | 1e-6 | [-0.2,0.2] | 8 | small S, big D |
| 19 | [4,255,4096] | 1020 | 4096 | f16 | 1e-3 | [-65504,65504] | 4 | fp16 extrema |
| 20 | [2,3,17,1024,128] | **104448** | 128 | f32 | 1e-6 | [-20,40] | 1 | 5-D, mid S |

S ranges 231…1 000 003; D ranges 2…8192; ranks 2–5; dtypes f16/bf16/f32.
(Earlier-phase artifacts mis-listed case 11 as S=4019 and case 20 as S=102;
both corrected above — case 20's leading product is 2·3·17·1024, not 2·3·17.)

## 4. Dispatch

**One kernel, one JIT specialization, one tile size — for all 20 cases.**

Host `rms_norm(x, gamma, epsilon=1e-6)`:

1. `ensure_npu_platform()` first (mandated).
2. Contiguity: `if not x.is_contiguous(): x = x.contiguous()`; same for `gamma`.
   (Allowed torch usage; `.contiguous()` is whitelisted.)
3. Metadata only (no torch math): `D = x.shape[-1]`; `S = x.numel() // D`.
4. `out = torch.empty_like(x)` — same shape/dtype/device, contiguous.
5. Empty guard: `if S == 0 or D == 0: return out`.
6. `tile_d = _TILE_D = 1024` (single constant; no host branch on shape/dtype).
7. `num_d_tiles = asc.ceildiv(D, tile_d)`.
8. `inv_D = 1.0 / float(D)` — precomputed on host, passed as `float` param to
   avoid any in-kernel int division.
9. `cores = min(_MAX_CORES, S)` with `_MAX_CORES = 72` (AIV cores on 950PR).
   Every listed case has S ≥ 231 > 72, so all launch 72 cores; the `min` still
   protects hypothetical S < 72 from the spec's S ≥ 1 lower bound.
10. `_rms_norm_kernel[cores](x, gamma, out, S, D, num_d_tiles, epsilon, inv_D, tile_d)`.
11. Return `out`.

**No dtype dispatch:** the kernel promotes f16/bf16/f32 inputs uniformly via
`xf = x.to(asc.float32)` and stores via `y.to(x.dtype)`. The spec guarantees
`gamma.dtype == x.dtype`, so the same cast path serves both. **No rank
dispatch:** x is logically flattened to `S` rows of length `D`; the kernel
indexes by `row = r * D`, so any rank 2–5 collapses identically.

## 5. Tiling

- **Outer (row) loop:** grid-stride
  `for r in asctile.range(asctile.block_idx(), S, asctile.block_num())` —
  core `b` handles rows `b, b+72, b+144, …`. Each row is independent; no
  cross-row sync, no `gm_barrier`.
- **Inner (D) loop:** sequential `for dt in asctile.range(num_d_tiles)` —
  stream D in `TILE_D` chunks per row. **No `unroll_factor`** anywhere → UB
  stays single-buffered (the prior overflow was at `unroll=2`).
- **TILE_D = 1024** (`asc.ConstExpr[int]`, compile-time — required because it
  appears in `copy_in` tile shapes). Fixed for all cases: D=2 wastes 1022 tile
  slots (bandwidth, not UB); D=8192 streams 8 chunks. The two-pass streaming
  means peak UB is bounded by one chunk's live set, independent of D.

### Tail handling

```python
od = dt * tile_d
n = tile_d if od + tile_d <= D else D - od      # last chunk's real length
x  = asctile.copy_in(x_gm, [row + od], [tile_d], real_shape=[n])
g  = asctile.copy_in(g_gm, [od],           [tile_d], real_shape=[n])
...
asctile.copy_out(y.to(x.dtype), o_gm, [row + od], real_shape=[n])
```

`real_shape=[n]` on every load **and** store — no host padding, no masked
copy. Tail chunk padding uses the default `pad_value=0`:

- **Pass 1 reduce:** padded x = 0 → 0² = 0 → contributes 0 to `acc`. Default
  zero is operation-neutral for a sum (the contract explicitly permits zero
  for additive/reduction inputs). No `pad_value` override needed.
- **Pass 2 apply:** padded x = 0, padded γ = 0 → padded y = 0·inv_rms·0 = 0,
  and `copy_out` discards padded slots via `real_shape=[n]`. Harmless.

No `gm_barrier` (rows independent). No host padding tensor allocation.

## 6. dtypes

| input x / γ dtype | in-kernel load | compute | store |
|---|---|---|---|
| float16 | f16 tile | `.to(asc.float32)` before **any** arithmetic (esp. squaring) | `y.to(asc.float16)` |
| bfloat16 | bf16 tile | `.to(asc.float32)` | `y.to(asc.bfloat16)` |
| float32 | f32 tile | already f32 (`.to(asc.float32)` is a no-op cast, kept for uniformity) | `y.to(asc.float32)` |

- `xf = x.to(asc.float32)` is applied **before** `xf * xf` so case 19
  (|x| up to 65504, where 65504² ≈ 4.29e9 overflows f16's 65504 max) computes
  the square in f32. This is the precision-standard expectation: f32 internal
  compute for f16/bf16 inputs (matches golden's `F.rms_norm` internal upcast).
- `inv_rms` and `mean_sq` live entirely in f32.
- The output dtype always equals `x.dtype` (the spec's contract); the final
  `y.to(x.dtype)` cast down is the only lossy step, well within the f16
  threshold of 2⁻¹⁰ ≈ 9.8e-4.

## 7. UB budget (TILE_D = 1024, unroll = 1)

Measured calibration factor from prior iterations: real usage ≈ **1.6×** the
naive `Σ(visible_tile_bytes)` because the compiler adds hidden temporaries.

**Pass 1** (live within one D-chunk iteration; `acc` is scalar ≈ free):

| tile | dtype | naive bytes |
|---|---|---|
| x | f16/bf16 (f32) | 2048 (4096) |
| xf = x.to(f32) | f32 | 4096 |
| xf*xf | f32 | 4096 |

Pass-1 peak: f16/bf16 ≈ 10 240 B; f32 ≈ 12 288 B.

**Pass 2** (live within one D-chunk iteration):

| tile | dtype | naive bytes |
|---|---|---|
| x (reload) | f16/bf16 (f32) | 2048 (4096) |
| γ | f16/bf16 (f32) | 2048 (4096) |
| xf | f32 | 4096 |
| γf | f32 | 4096 |
| inv_rms_tile (broadcast) | f32 | 4096 |
| tmp = xf * inv_rms_tile | f32 | 4096 |
| y = tmp * γf | f32 | 4096 |
| y_out = y.to(x.dtype) | f16/bf16 (f32) | 2048 (4096) |

Pass-2 peak: f16/bf16 ≈ 26 624 B; f32 ≈ 36 864 B.

**Total** (conservatively assuming the compiler keeps both passes' tiles live
— in reality pass-1 tiles are released before pass-2 allocates, since the
loops are sequential):

- f16/bf16 worst: (10 240 + 26 624) × 1.6 ≈ **59 KB**
- f32 worst: (12 288 + 36 864) × 1.6 ≈ **79 KB**

Budget = **253 952 B ≈ 248 KB**. Headroom is 3–4× even in the pessimistic
both-passes-live model; the realistic pass-2-only peak is ~42–59 KB. This is
the fix for the prior iteration's UB overflow (328 608 B at TILE_D=2048,
unroll=2, single-pass): two-pass streaming + TILE_D=1024 + no unroll cuts
usage ~6–8×.

Tuning note (not used): the headroom suggests TILE_D=2048 with unroll=1 *might*
fit (~120–160 KB), but with no NPU access to verify a UB-overflow diagnostic is
fatal (halving TILE is the only recovery and costs a recompile), the design
pins TILE_D=1024 — verified-safe-by-construction.

## 8. Numerical behavior

| Risk | Case(s) | Analysis / mitigation |
|---|---|---|
| x² overflows f16 (65504² ≈ 4.3e9 > 65504) | 19 | Promote to f32 **before** squaring: `xf = x.to(asc.float32)` then `xf * xf`. 4.3e9 fits f32. |
| Σx² overflows f32 | 5, 19 | Worst: case 19 = 65504²·4096 ≈ 1.76e13; case 5 = 100²·8192 ≈ 8.2e7. Both ≪ f32 max 3.4e38. No overflow. |
| mean_sq + ε precision | 9 (ε=1e-12) | x∈[-0.5,0.5], mean_sq ≈ O(0.08); 0.08 + 1e-12 ≈ 0.08 (no flush; f32 subnormal min ≈1.4e-45 ≪ 1e-12). |
| All-zero row → rsqrt(ε) | 17 | mean_sq = 0 → inv_rms = rsqrt(1e-4) = 100; y = 0·100·γ = 0. ε>0 always ⇒ rsqrt argument strictly positive. ✓ |
| Division by zero | all | No division in-kernel: we use `rsqrt` (argument = mean_sq+ε ≥ ε > 0). Pad γ=0 in pass-2 tail is multiplied then discarded, never a divisor. |
| Catastrophic cancellation | n/a | Algorithm has **no subtraction**; `+ ε` adds a strictly-positive tiny term. |
| Inf/NaN propagation | all | IEEE: x=inf → x²=inf → acc=inf → inv_rms=rsqrt(inf)=0 → y=inf·0=NaN. Matches golden `F.rms_norm` (inf→NaN). **No host special-casing** (contract: do not branch unless golden does; golden doesn't). |
| rsqrt accuracy vs sqrt+div | all | `y = x·rsqrt(s)·γ` is mathematically `x/sqrt(s)·γ`; rsqrt is a single hardware op, no extra rounding vs the two-op path. Full f32 throughout keeps error well under the 2⁻¹⁰ (f16) / 2⁻⁷ (bf16) / 2⁻¹³ (f32) thresholds. |
| ε as host `float` | all | `epsilon` is passed as a `float` kernel param; 1e-12 and 1e-3 both exactly representable enough in f32; `+ ε` done in f32 in-kernel. |

`inv_D = 1.0 / float(D)` is computed **on host** (plain Python float, no torch)
and passed as a `float` param — avoids in-kernel integer division by D.

## 9. Anti-cheat compliance

- **All numerical work in `@asctile.jit` kernel on NPU:** reduce, rsqrt, apply,
  cast — no host math.
- **torch used ONLY for:** `ensure_npu_platform()`, `torch.empty_like(x)` (output
  alloc), and metadata/contiguity: `.shape`, `.numel()`, `.is_contiguous()`,
  `.contiguous()`. `inv_D`/`num_d_tiles`/`cores` use plain Python ints/floats and
  `asc.ceildiv` — no torch ops.
- **FORBIDDEN and absent:** `torch.nn.functional.*`, `torch.mul`/`torch.sum`/
  `torch.norm`, any tensor arithmetic (`a + b`, `x.sigmoid()`), `.to(dtype)` on
  device tensors, `torch.cat`/`torch.clone`. The harness's torch-dispatch hook
  and `data_ptr`-rotation cheater detection are not triggered — outputs are
  freshly allocated each call and never keyed on input pointer.
- **Output:** contiguous NPU tensor from `torch.empty_like`, exactly golden's
  shape/dtype, not a view of any input.
- **`ensure_npu_platform()`** is the first statement in the public callable.

## 10. Kernel architecture (pseudocode — implementation target)

```python
import torch
import asc
import asctile
from ._pyasc_runtime import ensure_npu_platform

_TILE_D = 1024
_MAX_CORES = 72


@asctile.jit
def _rms_norm_kernel(x_ptr: asc.GlobalAddress, gamma_ptr: asc.GlobalAddress,
                    out_ptr: asc.GlobalAddress,
                    S: int, D: int, num_d_tiles: int,
                    epsilon: float, inv_D: float,
                    tile_d: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr,     [S * D])
    g_gm = asctile.global_tensor(gamma_ptr, [D])
    o_gm = asctile.global_tensor(out_ptr,   [S * D])

    for r in asctile.range(asctile.block_idx(), S, asctile.block_num()):
        row = r * D
        # ---- pass 1: reduce sum-of-squares ----
        acc = asctile.reduce_sum(asctile.full([1, 64], 0.0, dtype=asc.float32))
        for dt in asctile.range(num_d_tiles):
            od = dt * tile_d
            n  = tile_d if od + tile_d <= D else D - od
            x  = asctile.copy_in(x_gm, [row + od], [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            acc = acc + asctile.reduce_sum(xf * xf)
        # per-row scalar → broadcast to tile (pure tile·tile apply)
        inv_rms_tile = asctile.rsqrt(
            asctile.full([tile_d], acc * inv_D + epsilon, dtype=asc.float32))

        # ---- pass 2: apply x · inv_rms · γ ----
        for dt in asctile.range(num_d_tiles):
            od = dt * tile_d
            n  = tile_d if od + tile_d <= D else D - od
            x  = asctile.copy_in(x_gm, [row + od], [tile_d], real_shape=[n])
            g  = asctile.copy_in(g_gm, [od],        [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            gf = g.to(asc.float32)
            y  = xf * inv_rms_tile * gf
            asctile.copy_out(y.to(x.dtype), o_gm, [row + od], real_shape=[n])


def rms_norm(x: torch.Tensor, gamma: torch.Tensor,
             epsilon: float = 1e-6) -> torch.Tensor:
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()
    D = x.shape[-1]
    S = x.numel() // D
    out = torch.empty_like(x)
    if S == 0 or D == 0:
        return out
    tile_d = _TILE_D
    num_d_tiles = asc.ceildiv(D, tile_d)
    inv_D = 1.0 / float(D)
    cores = min(_MAX_CORES, S)
    _rms_norm_kernel[cores](x, gamma, out, S, D, num_d_tiles,
                            epsilon, inv_D, tile_d)
    return out
```

Key design choices recap:
- `inv_rms_tile` (broadcast) makes the apply step `xf * inv_rms_tile * gf`
  fully tile·tile·tile — no scalar-placement ambiguity, matches the verified
  earlier-phase candidate.
- Scalar `acc` seeded with the mandated `reduce_sum(full([1,64],0.0,...))` form.
- Both D-loops use `asctile.range` (never Python `range`); outer row loop is
  the grid-stride 3-arg form without `unroll_factor`.

## 11. Validation ladder

| Gate | Method | Note |
|---|---|---|
| 1. Syntax | `python3 -m py_compile candidate.py` | Only locally runnable check (no pyasc/torch_npu/NPU on this box). |
| 2. Static contract | module-shape / signature review against §9 | Name `rms_norm`, schema `(x, gamma, epsilon=1e-6)`, asctile (not asc2), anti-cheat. |
| 3. UB fit | §7 budget math | ~59–79 KB ≪ 253 952 B at TILE_D=1024; the prior 328 608 B failure mode is structurally excluded. |
| 4. Numerical | §8 per-case reasoning | f32 internal, rsqrt (no div-by-zero), ε>0, IEEE inf/NaN match golden. |
| 5. Harness | official CANNBench on real 950PR | compile (0.2) + accuracy×20 (0.3) + perf vs aclnn (0.5). |

Gate 1 is the only locally-executable step; per task.md, no NPU here and no
package install — reason statically, write `candidate.py`, py_compile, STOP.

DESIGN_DONE
