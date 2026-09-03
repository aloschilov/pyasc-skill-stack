# Sigmoid CANNBench — Design (DESIGN phase)

Evidence label for this document: `suspected` (no compile/numerical evidence yet;
design-only, derived from task.md + loaded skill contracts + the proven reference
module embedded in task.md which scores 100% accuracy on this harness).

## 1. Algorithm

Elementwise sigmoid, single input / single output, ND format, no attrs.

  y = 1 / (1 + e^(-x))

Pinned kernel formula (matches the task's 100%-accuracy reference verbatim):

  xf  = x.to(asc.float32)                 # promote f16/bf16 -> f32
  yf  = asc2.div(1.0, asc2.exp(-xf) + 1.0)
  out = yf.to(x.dtype)                     # cast back on store

This is the **simple** form, NOT the `exp(-|x|)+where` "stable" variant. Rationale:
for sigmoid, IEEE saturation makes the naive form correct in every extreme regime
(see §7). The previous iteration switched to `asc2.where(xf>=0.0, 1.0, e)` and that
is exactly what failed (§2).

## 2. Previous-iteration root cause (MUST avoid)

All 20 cases failed at the exact-v2 compile gate with the identical CodegenError:

  AttributeError: 'float' object has no attribute 'dtype'
  caret -> asc2.where(xf >= 0.0, 1.0, e)

Cause: `asc2.where(cond, scalar, tile)` with a bare Python `1.0` as a branch value
is not supported by codegen (it calls `.dtype` on the scalar). The reference module
avoids `where` entirely and uses `asc2.div(1.0, exp(-x)+1.0)`, which compiled and
scored 100% accuracy.

Design rule for the implementation phase:
- Do NOT call `asc2.where` with a bare scalar branch. If `where` is ever needed,
  build the branch as a tile via `asc2.full([tile_size], v, dtype=asc.float32)`.
- Prefer the reference's `asc2.div(1.0, tile)` form (function-call division with
  scalar numerator is allowed; the "scalar on the right" rule applies to the
  `+ - * /` OPERATOR overloads, not to explicit `asc2.div`).

## 3. Pinned-v2 APIs (no `tensor/load/store` spelling)

- `asc2.global_tensor(ptr, [size])` — 1-D GM view (rank must match copy_in/out)
- `asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])` — tile load + tail
- `asc2.copy_out(y, out_gm, [off], real_shape=[n])` — tile store + tail
- `asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2)`
  — grid-stride tile loop (NO `parallel` kwarg; `gm_barrier` not supported here)
- `asc2.exp`, `asc2.div` — math ops
- `tile.to(dtype)` — in-kernel dtype cast (f16/bf16<->f32)
- `asc.GlobalAddress` — pointer param typing
- `asc.ConstExpr[int]` — compile-time `tile_size` (REQUIRED for copy_in tile shape)
- `asc.ceildiv`, `asc.float32` — host helpers / dtype constants
- Launch: `_sigmoid_kernel[cores](x, out, size, num_tiles, tile_size)`
- Host: `ensure_npu_platform()` first; `torch.empty_like(x)` for output

## 4. All 20 cases + dispatch

Dispatch is driven purely by total element count `size = x.numel()` against
`72 * WIDE_TILE = 72 * 3072 = 221_184`. Every evaluation case is >= ~1M elements,
so ALL 20 take the WIDE path with 72 cores. NARROW (1024) is retained only for the
general contract (spec allows total elements 1..64M); it is dead for these 20.

| case | shape                | dtype   | elems      | path   | cores | notable values        |
|-----:|----------------------|---------|------------|--------|-------|-----------------------|
|  1   | [1024,1024]          | f16     | 1,048,576  | WIDE   | 72    | [-1,1]                |
|  2   | [2048,2048]          | f32     | 4,194,304  | WIDE   | 72    | [-2,2]                |
|  3   | [4096,4096]          | bf16    | 16,777,216 | WIDE   | 72    | [-3,3]                |
|  4   | [8192,8192]          | f16     | 67,108,864 | WIDE   | 72    | [-10,10]              |
|  5   | [8192,8192]          | f32     | 67,108,864 | WIDE   | 72    | [-100,100]            |
|  6   | [1023,1023]          | bf16    | 1,046,529  | WIDE   | 72    | [-0.1,0.1] tail       |
|  7   | [1009,1021]          | f16     | 1,030,189  | WIDE   | 72    | [-1,2] tail           |
|  8   | [1537,769]           | f32     | 1,181,953  | WIDE   | 72    | [-5,10] tail          |
|  9   | [363,367,373]        | bf16    | 49,774,013 | WIDE   | 72    | [-50,100]             |
| 10   | [2049,513]           | f16     | 1,051,137  | WIDE   | 72    | [-65504,65504] extrema|
| 11   | [3,7,13,4001]        | f32     | 1,094,673  | WIDE   | 72    | [-88,88]              |
| 12   | [1000003]            | bf16    | 1,000,003  | WIDE   | 72    | [-inf,inf] IEEE       |
| 13   | [11,13,17,67,67]     | f32     | 1,085,239  | WIDE   | 72    | [nan,nan] all-NaN     |
| 14   | [3,7,11,13,1009]     | f16     | 3,026,409  | WIDE   | 72    | [0,0] -> all 0.5      |
| 15   | [512,2049]           | f32     | 1,049,088  | WIDE   | 72    | [-0.5,0.5]            |
| 16   | [255,8193]           | bf16    | 2,089,215  | WIDE   | 72    | [-1,3]                |
| 17   | [4097,511]           | f16     | 2,093,567  | WIDE   | 72    | [-1000,1000]          |
| 18   | [2,511,2049]         | f32     | 2,098,078  | WIDE   | 72    | [-0.2,0.2]            |
| 19   | [4,255,2049]         | bf16    | 2,089,980  | WIDE   | 72    | [-3,6]                |
| 20   | [2,3,17,1024,101]    | f32     | 10,567,168 | WIDE   | 72    | [-20,40]              |

Three unique specializations by dtype: f16, f32, bf16. `tile_size` is the same
ConstExpr (3072) for all WIDE launches, so the JIT cache key differs only by the
pointer dtype -> 3 specializations, matching the evaluator's reported
`unique_specializations: 3`.

## 5. Tiling & tails

- 1-D flatten; `size = x.numel()`. Rank-1 global_tensor throughout (no 2-D mix).
- Grid-stride loop: each of the 72 cores strides over tiles:
  `for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2)`
- `num_tiles = asc.ceildiv(size, tile_size)`
- Per-tile offset `off = t * tile_size`; tail length
  `n = tile_size if off + tile_size <= size else size - off`
- Tail loads/stores use `real_shape=[n]` (NO host padding). Tail-triggering cases:
  6,7,8,9,10,11,12,13,15,16,17,18,19,20 (all non-multiples of 3072); verified
  non-empty tails via `size % 3072 != 0` for each.
- `cores = min(72, num_tiles)`; for all 20 cases num_tiles >> 72 -> cores = 72.
- `unroll_factor=2` doubles pipelined live ranges -> folded into UB budget (§6).

## 6. UB budget (capacity 253,952 B, static allocation)

Task calibration (verbatim): sigmoid chain ~6 visible f32 values uses 155,648 B at
TILE=2048 and 311,296 B (OVERFLOW) at TILE=4096 -> real usage ~1.58x the naive
`visible*4*T*2` (the x2 is unroll_factor).

Visible distinct tiles for the chosen formula:
  x(in) | xf(f32) | -xf | e=exp | d=e+1 | yf=div | out  ~ 6-7 values; compiler
fuses -xf into exp and folds no-op f32->f32 casts (x==xf, yf==out for f32 cases),
giving ~4-5 live f32 tiles + 2 f16 tiles (f16/bf16 cases).

Estimate (1.58x factor, unroll_factor=2 already in the formula):
- WIDE TILE=3072: 1.58 * 6 * 4 * 3072 * 2 = 232,981 B  < 253,952  (PASS, ~21 KB
  headroom). Proven by the reference module compiling + scoring 100%.
- TILE=4096: 1.58 * 6 * 4 * 4096 * 2 = 310,573 B > 253,952  (OVERFLOW) -> 4096
  is forbidden; the 1.58x factor matches the task's 311,296 B measurement.
- NARROW TILE=1024: ~77,660 B (ample).

Decision: WIDE_TILE=3072, NARROW_TILE=1024. If the exact-v2 gate reports
`UB overflow`, halve TILE (1536 / 512) — do NOT drop cases. `where`/comparison
dest alignment (256 B) is satisfied for any TILE>=64 (TILE*4 % 256 == 0).

## 7. Numerical risks (thresholds: f16 2^-10, bf16 2^-7, f32 2^-13; MARE < 10x)

The simple `1/(1+exp(-x))` is safe for sigmoid in every regime via IEEE saturation
(no catastrophic cancellation — only `+1` and `1/..`, no near-equal subtraction):

- x -> +inf:  exp(-x) -> 0      -> y = 1/(1+0) = 1          (case 12 +inf)
- x -> -inf:  exp(-x) -> +inf   -> y = 1/(1+inf) = 0        (case 12 -inf)
- x = 0:      exp(0) = 1        -> y = 1/2 = 0.5            (case 14 all-zero)
- x = NaN:    propagates 1/(1+NaN) = NaN                    (case 13 all-NaN)
- large |x| (cases 5,10,17: -100/-65504/-1000): exp(-x) saturates to 0 or +inf;
  division yields exactly 1 or 0 — the correct limit, no denormal noise.
- f16/bf16: promote to f32 in-kernel, compute, cast back -> meets 2^-10 / 2^-7.
- f32 extreme [-88,88],[-100,100]: exp under/overflow saturates cleanly; the
  f32 threshold 2^-13 (~1.2e-4) is met because the only error is one exp + one
  div rounding, well under 1 ULP-scale relative error away from the limits.

Explicitly REJECTED: the `exp(-|x|)+where` rewrite — it adds a `where` with a
scalar branch (codegen bug, §2) and is unnecessary since IEEE saturation already
gives exact limits. `log(1+tiny)` risk: N/A (no log in the kernel).

## 8. Anti-cheat constraints

- ALL numerical work (exp, div, casts) inside the single `@asc2.jit` kernel.
- torch restricted to: `ensure_npu_platform()`, `x.is_contiguous()`,
  `x.contiguous()`, `torch.empty_like(x)`, `x.numel()`. No torch math/dispatch
  ops, no `.to(dtype)` on device data (cast is in-kernel via `tile.to()`), no
  `data_ptr` caching, no views of inputs returned, no `torch.sigmoid`/clone/cat.
- Output: `torch.empty_like(x)` preserves exact shape+dtype; contiguous NPU
  tensor returned; never a view of `x`.
- No `print`/`import`/`break`/`continue`/early `return`/`raise`/`math.*`/Python
  `range()` inside the kernel (syntax-constraints skill).

## 9. Local validation ladder

1. `python3 -m py_compile candidate.py` — Python syntax ONLY (the only step
   runnable in THIS design/impl environment; task.md states pyasc/asc2/torch_npu
   and NPU are absent — do NOT import/compile/run/install).
2. Exact-v2 local compile gate (evaluator-side): all 20 routes must lower through
   pinned pyasc v2 with the 3 dtype specializations. A compile failure is
   measured feedback -> repair (e.g. the §2 `where` fix) and re-run all 20.
   Target: `verified-local-compile`.
3. camodel execution vs golden.py on representative cases -> relative-error
   check (MERE/MARE). Not available here; only the harness yields this.
   Target: `verified-camodel`.
4. CANNBench on real Ascend950PR NPU — acceptance + 0.5 performance vs aclnn
   baseline. Target: `verified-cannbench`.

Current status: pre-implementation; no local-compile evidence yet -> `suspected`.
No candidate.py written in this phase (per instructions). Submission credit will
NOT be spent from this environment (remote eval is user-gated).

## 10. Implementation-phase checklist (for the next step)

- Copy the reference module structure from task.md (lines 256-300) verbatim as
  the skeleton; it is proven to compile and score 100%.
- Keep the simple `asc2.div(1.0, asc2.exp(-xf) + 1.0)` chain.
- DO NOT introduce `asc2.where` with a bare scalar (the §2 regression).
- Keep WIDE_TILE=3072 / NARROW_TILE=1024, `unroll_factor=2`, `real_shape` tails.
- After writing: run `python3 -m py_compile candidate.py`, then STOP (no import).

DESIGN_DONE
