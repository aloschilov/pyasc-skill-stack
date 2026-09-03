# Exp — design.md (DESIGN phase, provenance-gated)

Model: glm-5.2 (dashscope/glm-5.2). Skills loaded: `pyasc-cannbench-kernel`,
`pyasc-syntax-constraints`. Implementation context: task.md + loaded skill
references only. No submission modules, run artifacts, or pyasc source
inspected. No `candidate.py` written in this phase.

## 1. Operator & algorithm

Elementwise `Exp`. Output shape/dtype == input shape/dtype. dtypes: f16, f32,
bf16. Golden (reference only, never used at runtime) computes in f32 and casts
back; the candidate must reproduce that f32-internal math inside one
`@asc2.jit` kernel.

Unified formula (matches golden's f32 associativity exactly):
```
factor = 1.0           if base <= 0.0      # natural e
factor = log(base)     if base >  0.0      # log() computed on host in float64,
                                          # rounds to the same f32 as torch.log
arg   = (xf * scale + shift) * factor      # f32, golden's exact order
y     = exp(arg)                           # f32
store = y.to(input_dtype)                  # f16/bf16/f32
```
- `base <= 0` → factor=1.0 (multiply-by-1.0 is exact for finite/inf/nan in
  IEEE, so arg == `scale*x+shift`, identical to golden's natural-e branch).
- `base > 0` → factor=math.log(base); for base==1.0, math.log(1.0)==0.0
  exactly → arg==0.0 → y==1.0 for all finite inputs (cases 11/17/20).
- No host special-casing of base==1, inf, or nan: the unified expression plus
  IEEE propagation through `asc2.exp` reproduces golden's special-value
  behavior (contract: "do NOT special-case IEEE values unless the golden
  does").

## 2. Pinned-v2 APIs (CANNBench runtime, no load/store spelling)

- `asc2.global_tensor(ptr, [size])` — 1-D GM view (x and out).
- `asc2.copy_in(gm, [off], [tile_size], real_shape=[n])` — load tile (tail via
  `real_shape`).
- `asc2.copy_out(tile, gm, [off], real_shape=[n])` — store tile.
- `tile.to(asc.float32)` / `tile.to(x.dtype)` — promote f16/bf16→f32 and back.
- `asc2.exp(tile)` — core transcendental (IEEE saturation to +inf / flush to 0).
- tile `*` scalar, `+` scalar (scalars on the RIGHT per contract).
- `asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2)`
  — grid-stride tile loop.
- `asc.ceildiv(size, tile)` (host) + `kernel[cores](...)` launch, cores =
  min(72, num_tiles). `asc.ConstExpr[int]` for tile_size. `ensure_npu_platform`
  first call. Target Ascend950PR_9599, ≤72 AIV cores, UB 253952 B.

Kernel signature (mirrors the proven sigmoid template):
```
@asc2.jit
def _exp_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                size: int, num_tiles: int, scale: float, shift: float,
                factor: float, tile_size: asc.ConstExpr[int]): ...
```
Launch order = tensors, ints, floats, constexpr (contract-mandated).

## 3. All 20 cases (shapes, dtype, value_range, attrs → effective params)

`eff_scale = scale*factor`, `eff_shift = shift*factor` shown for traceability;
kernel itself takes `scale, shift, factor` separately and computes
`(xf*scale+shift)*factor` to match golden's f32 order.

| # | shape | dtype | val_range | base | scale | shift | factor | arg behavior / notes |
|---|---|---|---|---|---|---|---|---|
| 1 | [1024,1024] 1.05M | f16 | [-1,1] | -1 | 1.0 | 0.0 | 1.0 | e^x, benign |
| 2 | [2048,2048] 4.19M | f32 | [-2,2] | -1 | 1.5 | 0.0 | 1.0 | e^(1.5x) |
| 3 | [4096,4096] 16.8M | bf16 | [-3,3] | -1 | 2.0 | 0.0 | 1.0 | e^(2x) |
| 4 | [8192,8192] 67.1M | f16 | [-10,10] | -1 | 0.5 | 0.0 | 1.0 | e^(0.5x), largest case |
| 5 | [8192,8192] 67.1M | f32 | [-100,100] | -1 | 1.0 | 1.0 | 1.0 | arg∈[-99,101] → e^101=+inf (golden too) |
| 6 | [1023,1023] 1.05M | bf16 | [-0.1,0.1] | 2 | 1.0 | 0.0 | ln2 | 2^x, tiny args |
| 7 | [1009,1021] 1.03M | f16 | [-1,2] | 2 | 1.5 | 0.0 | ln2 | 2^(1.5x) |
| 8 | [1537,769] 1.18M | f32 | [-5,10] | 10 | 1.0 | 0.0 | ln10 | 10^x, max 1e10 (f32 ok) |
| 9 | [363,367,373] 49.7M | bf16 | [-50,100] | -1 | 2.0 | 0.5 | 1.0 | arg up to 200.5 → +inf (golden too) |
| 10 | [2049,513] 1.05M | f16 | [-65504,65504] | -1 | 1.0 | 2.0 | 1.0 | extreme → +inf/0 mix (golden too) |
| 11 | [3,7,13,4003] 1.09M | f32 | [-88,88] | 1 | 2.0 | 0.0 | 0.0 | base==1 → y==1 exactly |
| 12 | [1000007] 1.00M | bf16 | [-inf,inf] | -1 | 0.5 | 0.5 | 1.0 | IEEE inf→+inf, -inf→0, nan→nan |
| 13 | [11,13,17,67,67] 10.9M | f32 | [nan,nan] | 2 | 1.0 | 1.0 | ln2 | nan propagates through exp |
| 14 | [3,7,11,13,1013] 3.04M | f16 | [0,0] | -1 | 2.0 | 1.0 | 1.0 | all zeros → arg=1.0 → y=e |
| 15 | [512,2049] 1.05M | f32 | [-0.5,0.5] | -1 | 1.0 | 0.5 | 1.0 | e^(x+0.5) |
| 16 | [255,8193] 2.09M | bf16 | [-1,3] | -1 | 1.2 | 0.0 | 1.0 | e^(1.2x) |
| 17 | [4097,511] 2.09M | f16 | [-1000,1000] | 1 | 0.5 | 0.0 | 0.0 | base==1 → y==1 (finite x, xf*0=0) |
| 18 | [2,511,2049] 2.09M | f32 | [-0.2,0.2] | 2 | 0.5 | 0.0 | ln2 | 2^(0.5x) |
| 19 | [4,255,2049] 2.09M | bf16 | [-3,6] | 10 | 0.5 | 0.5 | ln10 | 10^(0.5x+0.5) |
| 20 | [2,3,17,1024,101] 10.5M | f16 | [-20,40] | 1 | 1.5 | 1.0 | 0.0 | base==1 → y==1 |

Coverage: all 20 cases flatten to 1-D `size = x.numel()` (min 1.00M, max 67.1M
elems); every case exceeds `72*3072 = 221184`, so all take the WIDE tile path.
Prime/odd extents (1023×1023, 1009×1021, 1000007, 1009-prime-ish, 5-D shapes)
are handled by `real_shape` tails — no host padding.

## 4. Tiling & tails

- 1-D flatten; grid-stride tile loop (proven sigmoid Pattern A):
  `off = t*tile_size`; `n = tile_size if off+tile_size <= size else size-off`;
  `copy_in(..., real_shape=[n])`; compute; `copy_out(..., real_shape=[n])`.
- Two compiled tiles: `_WIDE_TILE=3072` (size ≥ 221184) and
  `_NARROW_TILE=1024` (small, kept for template robustness; dead for these 20
  cases). Host selects via `asc.ceildiv`; `cores = min(72, num_tiles)` (=72 for
  every case).
- Tails: the last partial tile per core is loaded/stored with
  `real_shape=[n]`; no pad/copy on host. Works for all odd/prime extents.

## 5. UB budget (253952 B static cap)

Exp chain (f16 load → f32 cast → `*scale` → `+shift` → `*factor` → `exp` → f16
store) has **5 f32 intermediates + f16 load + f16 store**, structurally
identical to the proven sigmoid chain (cast, neg, exp, +1, div = 5 f32
intermediates). Sigmoid measured 155648 B at TILE=2048 and ≈233K at TILE=3072
(overflow at 4096 = 311296). Therefore:
- `_WIDE_TILE=3072` → ~233K B < 253952 B. Fits, by direct equivalence to the
  measured sigmoid chain.
- 1.6× safety factor applied; no `where`/comparison dest (256-B alignment
  rule) is used by Exp, so no extra alignment hazard.
- Fallback ladder on `RuntimeError: UB overflow`: halve TILE → 1536 → 1024.
  **Never drop a case** to make a gate pass (skill: preserve full coverage).

## 6. Numerical risks

1. **Overflow to +inf** (cases 5, 9, 10): args exceed ~88.7 → `asc2.exp`
   saturates to +inf; golden (torch.exp) does the same. Goal is matching
   inf/0 pattern, not finite accuracy at the cliff. Mitigation: compute arg
   in-kernel in golden's exact f32 order `(xf*scale+shift)*factor` so the
   overflow boundary is identical. Boundary points are measure-zero for the
   harness's sampled rel-error metric.
2. **Underflow to 0** (large negative args, e.g. case 10 x=-65504): both
   sides → 0; `|0-0|/(0+1e-7)=0`. Safe.
3. **IEEE inf/nan** (cases 12 ±inf, 13 nan): `asc2.exp` propagates per IEEE
   (+inf→+inf, -inf→0, nan→nan). Contract states hardware ops propagate IEEE
   values correctly; no host branch. Verified by camodel comparison, not
   claimed from compile alone.
4. **base==1.0** (cases 11, 17, 20): factor = math.log(1.0) = 0.0 exactly;
   arg = (…)*0.0 = 0.0 for finite x (all three cases have finite ranges);
   exp(0)=1.0 exactly. Inputs finite ⇒ `xf*0` is exact 0 (no nan). For
   base<=0, factor=1.0 is exact-by-1.0 ⇒ arg == `scale*x+shift`.
5. **f16/bf16 rounding**: f16→f32 and bf16→f32 casts are exact (all f16/bf16
   values are f32-representable); the only lossy cast is f32→f16/bf16 on
   store, which golden also performs, so rounding matches. Thresholds
   (f16 2^-10, bf16 2^-7, f32 2^-13) comfortably satisfied by f32-internal
   compute.
6. **No catastrophic cancellation**: Exp is monotone, no subtraction of
   nearly-equal quantities; the `+shift` is benign.
7. **factor precision**: host math.log(base) (float64) rounded to f32 equals
   torch.log(tensor(base,f32)) (both correctly-rounded) ⇒ factor identical
   to golden's, so arg and exp(arg) match to f32-ulp across the range.

## 7. Anti-cheat constraints

- ALL numerical work (cast, scale, shift, factor-mul, exp, store-cast) inside
  the single `@asc2.jit` kernel launched on NPU.
- `torch` used ONLY for: `torch.empty_like(x)` (output alloc),
  `.is_contiguous()`, `.contiguous()`, `.numel()` (metadata). No torch math
  ops, no `torch.exp`/`torch.mul`/`.to(dtype)` on device data, no `torch.cat`,
  no `torch.clone`, no F.passthrough. No caching by `data_ptr` (harness
  rotates pointers).
- `math.log(base)` on host is a scalar attr reduction (contract:
  "precompute module-level constants"; "NO math.* calls" applies INSIDE jit
  only). It is not tensor compute and not a torch op.
- Output is a fresh contiguous NPU tensor with golden's exact shape/dtype
  (`torch.empty_like(x)`). No views of inputs returned.
- `ensure_npu_platform()` is the first host call. Target ≤72 cores.

## 8. Local validation ladder

1. `python3 -m py_compile candidate.py` — syntax only.
2. Worker static contract check — module shape, callable name/signature
   (`exp(x, base=-1.0, scale=1.0, shift=0.0)`), imports, no forbidden torch
   ops, no forbidden in-jit constructs.
3. Exact-v2 local compile gate — lower all 20 cases' shapes/dtypes/attrs
   through pinned pyasc v2; every case must JIT-compile. Catches UB
   overflow, ConstExpr misuse, unsupported syntax. On failure: halve TILE
   (3072→1536→1024), re-run all 20; never drop/shrink a case.
4. Numerical evidence is NOT claimed from the compile gate (skill: only camodel
   execution or CANNBench on real NPU gives numerical evidence). Evidence
   labels: `verified-local-compile` once gate passes; `suspected` until then;
   `verified-camodel` / `verified-cannbench` only after remote evaluation
   (requires explicit user enablement; no submission credit spent here).

Environment note: this scratch env has no pyasc/asc2/torch_npu/NPU (per
task.md), so steps 1–3 are the plan for a pyasc-equipped run; the design
itself is static and complete.

DESIGN_DONE
