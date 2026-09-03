# RmsNorm — Design (DESIGN phase, provenance-gated workflow)

Skills loaded: `pyasc-cannbench-kernel`, `pyasc-syntax-constraints`.
Source of truth: `task.md` only (proto.yaml + golden.py + 20 cases + previous
evaluator feedback). No submission modules or pyasc source inspected.

## 1. Algorithm

Per row (last dim D, independent):

```
acc      = sum_{j=0..D-1} x_j^2          # fp32 accumulation
mean_sq  = acc / D
rscale   = rsqrt(mean_sq + epsilon)     # broadcast scalar -> tile
y_i      = x_i * rscale * gamma_i
```

Flatten all leading dims into `S = numel / D` rows of `D`. Each AIV core
grid-strides over rows; for each row it makes two passes over `D` in
aligned chunks (pass 1 = accumulate `x^2` in fp32, pass 2 = normalize and
store). `gamma` is shared across rows, streamed per chunk alongside `x`.

## 2. Pinned-v2 APIs (exact spelling, per skill contract)

- `asc2.global_tensor(ptr, [size])` — 1-D GM views for x, gamma, out (all
  1-D to avoid the rank-mixing failure; offsets are scalar `r*D + dt*TILE`).
- `asc2.copy_in(gm, [off], [TILE], real_shape=[n])` — tail-safe tile load.
- `asc2.copy_out(tile.to(x.dtype), gm, [off], real_shape=[n])` — store.
- `asc2.range(block_idx(), S, block_num(), unroll_factor=2)` — outer row
  grid-stride (proven pattern).
- `asc2.range(num_d_tiles)` — inner D-tile loop (plain; loop-carried acc
  must NOT be unrolled, per accumulator rule).
- `asc2.reduce_sum`, `asc2.rsqrt`, `asc2.full([TILE], scalar, dtype=asc.float32)`.
- `tile.to(asc.float32)` / `tile.to(x.dtype)` casts; tile-tile `*`, tile-scalar
  `*` / `+` (scalar on RIGHT).
- Launch: `_rms_norm_kernel[cores](x, gamma, out, D, S, inv_d, epsilon,
  num_d_tiles, TILE_D)`; `cores = min(72, S)`; `TILE_D` is `ConstExpr[int]`.
- Host: `ensure_npu_platform()`, `.contiguous()`, `torch.empty_like`,
  `.numel()`/`.shape` only.

## 3. Kernel sketch (1-D, single kernel, specialized per dtype = 3 specs)

```
@asc2.jit
def _rms_norm_kernel(x_ptr, gamma_ptr, out_ptr, D:int, S:int,
                     inv_d:float, epsilon:float, num_d_tiles:int,
                     TILE_D: asc.ConstExpr[int]):
    x_gm  = asc2.global_tensor(x_ptr,  [S*D])
    g_gm  = asc2.global_tensor(gamma_ptr, [D])
    o_gm  = asc2.global_tensor(out_ptr, [S*D])
    for r in asc2.range(asc2.block_idx(), S, asc2.block_num(),
                        unroll_factor=2):           # rows, grid-stride
        base = r * D
        acc  = asc2.reduce_sum(asc2.full([1,64], 0.0,        # MANDATORY seed
                                         dtype=asc.float32))
        for dt in asc2.range(num_d_tiles):            # pass 1: sum of squares
            off = base + dt * TILE_D
            n   = TILE_D if off + TILE_D <= base + D else (base + D) - off
            x   = asc2.copy_in(x_gm, [off], [TILE_D], real_shape=[n])
            xf  = x.to(asc.float32)
            x2  = xf * xf
            acc = acc + asc2.reduce_sum(x2)
        # rscale tile (all transforms on a tile to avoid scalar div/rsqrt)
        rscale = asc2.rsqrt(asc2.full([TILE_D], acc, dtype=asc.float32)
                            * inv_d + epsilon)
        for dt in asc2.range(num_d_tiles):            # pass 2: normalize+store
            off = base + dt * TILE_D
            n   = TILE_D if off + TILE_D <= base + D else (base + D) - off
            x   = asc2.copy_in(x_gm, [off], [TILE_D], real_shape=[n])
            g   = asc2.copy_in(g_gm, [dt*TILE_D], [TILE_D], real_shape=[n])
            yf  = (x.to(asc.float32) * g.to(asc.float32)) * rscale
            asc2.copy_out(yf.to(x.dtype), o_gm, [off], real_shape=[n])
```

Host computes `inv_d = 1.0 / float(D)` (plain Python float — not torch math)
so the kernel only does tile `* inv_d` (scalar on right), never scalar `/D`.

## 4. All 20 cases (TILE_D = 1024 → num_d_tiles = ceil(D/1024))

| # | x shape | D | S | dtype | ndt | ε | value range / note |
|---|---|---|---|---|---|---|---|
| 1 | [32,128,768] | 768 | 4096 | fp16 | 1 | 1e-6 | [-1,1] |
| 2 | [32,128,1024] | 1024 | 4096 | fp32 | 1 | 1e-6 | [-2,2] |
| 3 | [32,128,2048] | 2048 | 4096 | bf16 | 2 | 1e-6 | [-3,3] |
| 4 | [16,256,4096] | 4096 | 4096 | fp16 | 4 | 1e-6 | [-10,10] |
| 5 | [8,512,8192] | 8192 | 4096 | fp32 | 8 | 1e-6 | [-100,100] |
| 6 | [4,1023,4097] | 4097 | 4092 | bf16 | 5 | 1e-5 | [-5,5]; tail=1 |
| 7 | [63,67,1023] | 1023 | 4221 | fp16 | 1 | 1e-8 | [-0.1,0.1]; tiny x² |
| 8 | [16,511,2049] | 2049 | 8176 | fp32 | 3 | 1e-4 | [-1,1]; tail=1 |
| 9 | [8,1021,4099] | 4099 | 8168 | bf16 | 5 | 1e-12 | [-0.5,0.5]; tiny ε, tail=3 |
| 10 | [33,127,769] | 769 | 4191 | fp16 | 1 | 1e-6 | [-1,2] |
| 11 | [31,129,2049] | 2049 | 3999 | fp32 | 3 | 1e-6 | [-50,100] |
| 12 | [17,255,4097] | 4097 | 4335 | bf16 | 5 | 1e-6 | [-3,6]; tail=1 |
| 13 | [7,1009,1021] | 1021 | 7063 | fp16 | 1 | 1e-7 | [-1,1] |
| 14 | [11,367,373] | 373 | 4037 | fp32 | 1 | 1e-5 | [-10,10] |
| 15 | [1000003,2] | 2 | 1000003 | bf16 | 1 | 1e-6 | [None,None]; huge S, tiny D |
| 16 | [11,13,17,67] | 67 | 2431 | fp16 | 1 | 1e-8 | [None,None]; 4-D |
| 17 | [3,7,11,4096] | 4096 | 231 | fp32 | 4 | 1e-4 | [0,0] all-zero rows |
| 18 | [2,511,8192] | 8192 | 1022 | bf16 | 8 | 1e-6 | [-0.2,0.2] |
| 19 | [4,255,4096] | 4096 | 1020 | fp16 | 4 | 1e-3 | [-65504,65504] fp16 max |
| 20 | [2,3,17,1024,128] | 128 | 104448 | fp32 | 1 | 1e-6 | [-20,40]; 5-D |

Every case has `S ≥ 231`, so `cores = 72` for all (no tiny-S launch hazard).
`numel == 0` is guarded by an early host return (no case hits it).

## 5. Tiling & tails

- Single tile size `TILE_D = 1024` (constexpr) for every case; no host
  padding. `real_shape=[n]` on every `copy_in`/`copy_out` handles non-aligned
  D tails (cases 6,8,9,12 with tail∈{1,1,3,1}; cases 1,7,10,13,14,15,16,20
  where `D < TILE_D` → `real_shape = D`).
- gamma tail mirrors the x tail: `copy_in(g_gm, [dt*TILE_D], [TILE_D],
  real_shape=[n])` — same `n`, so gamma and x tiles always match shape.
- Row grid-stride: `r = block_idx() + k*block_num()`; `unroll_factor=2` on
  this outer loop only (each row is independent — safe to pipeline).
- Inner D-tile loop is plain `asc2.range` (NO unroll) because it carries the
  scalar accumulator `acc`; the contract forbids unrolling acc loops.

## 6. UB budget (253952 B cap)

Previous attempt: `TILE_D=2048` overflowed — f32 279456 B, f16/bf16 328608 B
(measured by the compile gate). Those numbers are exactly `2 ×` a
single-buffered footprint, i.e. the outer `unroll_factor=2` doubled the
inner tile buffers. Halving `TILE_D` to 1024 therefore halves both:

| dtype | UB @ 2048 (meas) | UB @ 1024 (projected) | margin |
|---|---|---|---|
| f32 | 279456 | ~139728 | ~114 k |
| f16 / bf16 | 328608 | ~164304 | ~89 k |

Both fit. The projection is reliable because the measured 2048 figures already
include the 1.6× hidden-temporary factor, so linear scaling by `TILE_D` holds.

**Repair ladder if the gate still overflows** (never drop cases):
1. Drop the outer `unroll_factor=2` → single-buffered ≈ 70 k / 82 k (very safe).
2. Halve `TILE_D` to 512 → ≈ 35 k / 41 k (last-resort; valid for all D since
   `real_shape` tails still cover D=2..8192).

`asc2.where`/comparison 256-byte alignment: `TILE_D * 4 % 256 == 0` for
`TILE_D ∈ {1024,512}` → satisfied.

## 7. Numerical risks

1. **f16/bf16 square overflow (case 19, |x|=65504).** fp16 `x²` overflows at
   |x|>256 (256²=65536 > 65504 max). Mitigation: `xf = x.to(asc.float32)`
   BEFORE `x2 = xf*xf`. Matches golden's "fp16/bf16 internally accumulates in
   fp32" note. MANDATORY.
2. **Tiny-x underflow (cases 7,9,18).** mean(x²) can be ~1e-4..1e-2; ε as low
   as 1e-12 (case 9). `rsqrt(mean+ε)` stays finite because ε>0 guarantees the
   argument is positive; worst case `rsqrt(1e-12)=1e6`, output still fits bf16
   exponent range (case 9 is bf16 → fine). No special-casing.
3. **All-zero rows (case 17, x≡0).** acc=0, rscale=rsqrt(0+1e-4)=100,
   `y = 0 * 100 * gamma = 0` (0×finite = 0, no NaN). Correct, no branch needed.
4. **inf / nan inputs (value_range None cases 15,16).** IEEE propagation
   through hardware `*`, `rsqrt`, `+` must match golden (inf→x²=inf→
   rsqrt(inf)=0→y=inf·0=NaN). Do NOT clamp or branch — the skill forbids
   special-casing IEEE values unless the golden does (it does not).
5. **ε as large as 1e-3 (case 19).** No overflow; `+ε` is addition of two
   positive fp32 terms (no catastrophic cancellation).
6. **Accumulation order vs golden.** Tile-wise fp32 summation differs in ULP
   order from torch's fp32 reduction; within the 2^-10 (fp16) / 2^-13 (f32) /
   2^-7 (bf16) MERE thresholds. Acceptable.
7. **rscale computed on a tile (not a scalar).** Avoids uncertain scalar
   `rsqrt`/`/D`; uses proven `asc2.full([TILE_D], acc, ...)` then tile
   `* inv_d + ε` then `asc2.rsqrt(tile)`. `acc` is used as a fill value (same
   pattern as the cross-core reduction's `full([8], s, ...)`).

## 8. Anti-cheat constraints

- ALL math (`to`, `*`, `reduce_sum`, `rsqrt`, `full`, `copy_in/out`) is inside
  `@asc2.jit`. Host does only: `ensure_npu_platform()`, `.contiguous()` on x
  and gamma, `torch.empty_like(x)`, `.numel()`/`.shape`, and the launch.
- `inv_d = 1.0 / float(D)` is a plain Python float (not a torch op) — allowed.
- No `torch.*` math, no `.to(dtype)` on device data, no `torch.cat`/`clone`/
  `sum`/`nn.functional`, no `data_ptr` caching. Output is a fresh contiguous
  NPU tensor with exactly x's shape/dtype.
- `gamma.contiguous()` before launch (gamma is `(D,)`, normally contiguous;
  guard anyway). No view of an input is returned.

## 9. Local validation ladder

This box has no pyasc/asc/asc2/torch_npu and no NPU, so local execution is
impossible. The ladder is:

1. **Static syntax**: `python3 -m py_compile candidate.py` (syntax only;
   catches indentation/paren errors, not JIT errors).
2. **Worker static contract check** (signatures, imports, callable name).
3. **Exact-v2 local compile gate** (run by the evaluator on the pinned
   pyasc-v2 commit): lowers all 20 cases' arg-types + `TILE_D` constexpr
   through the real compiler. This is the gate that produced the previous
   "UB overflow" feedback — the primary go/no-go signal. A failure is
   measured feedback → apply the §6 repair ladder and re-run all 20.
4. **camodel execution** (numerical evidence vs golden) — not available
   locally; only the real NPU run gives accuracy/perf. Do not claim numerical
   correctness from step 3.
5. **CANNBench on NPU** — the sole acceptance/performance oracle
   (0.2 compile + 0.3 accuracy + 0.5 perf).

Evidence label for this design: `suspected` (no local compile/execute yet);
will promote to `verified-local-compile` only after step 3 passes all 20.

## 10. Open risks / decisions for the implement phase

- **Outer `unroll_factor=2` + inner loop-carried acc:** rows are independent
  so pipelining is logically safe, and the UB math (§6) says it fits. If the
  compiler refuses to overlap rows around the inner reduction, fall back to
  `unroll_factor=1` (UB ~70 k / 82 k) — correctness unaffected.
- **`asc2.full([TILE_D], acc, ...)` with a reduce-derived scalar fill value:**
  same pattern as the verified cross-core reduction; if rejected, compute
  `rscale` in a tiny second kernel on one core (skill's documented fallback).
- **Scalar arithmetic `acc + asc2.reduce_sum(x2)`:** explicitly verified in
  the skill contract; relied upon for the accumulator.

No `candidate.py` written in this phase.

DESIGN_DONE
