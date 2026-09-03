# design.md — Gelu (CANNBench, pyasc asc2)

Provenance: skills `pyasc-cannbench-kernel` + `pyasc-syntax-constraints` loaded; context
= `task.md` (desc/proto/golden/20 cases + kernel contract) only. No submission modules,
run artifacts, or pyasc source inspected. `candidate.py` not yet written.

## 1. Algorithm

Public callable `gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor`.
`approximate` is a **str** → unsupported as a kernel param (type constraints).
Dispatch on the host string and launch one of two specialized `@asc2.jit` kernels.
Both kernels flatten to 1-D (`numel`), promote f16/bf16→f32, compute, cast back.

- **erf mode** (`approximate == "none"`): `y = (x*0.5) * (erf(x*C1) + 1.0)`,
  `C1 = 1/sqrt(2) = 0.7071067811865476`.
- **tanh mode** (`approximate == "tanh"`):
  `y = (x*0.5) * (tanh(C0 * (x + K*x*x*x)) + 1.0)`,
  `C0 = sqrt(2/pi) = 0.7978845608028654`, `K = 0.044715`.
  Compute order: `x2=x*x; x3=x2*x; a=x3*K; b=x+a; c=b*C0; t=tanh(c); one=t+1.0; hx=x*0.5; y=hx*one`.

All constants are module-level floats (no `math.*` inside jit). Scalars always on the
RIGHT of tile arithmetic (`x*C1`, `t+1.0`, `x*0.5`) — `Tile.__rmul__` is absent.

## 2. Pinned v2 APIs (CANNBench contract)

- `from ._pyasc_runtime import ensure_npu_platform`
- `asc2.global_tensor(ptr, [size])` — 1-D GM view (rank-1 throughout; no 2-D mix).
- `asc2.copy_in(x_gm, [off], [tile], real_shape=[n])` — tile load with tail.
- `asc2.copy_out(y_cast, out_gm, [off], real_shape=[n])` — tile store with tail.
- `asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2)` — grid-stride.
- `asc.GlobalAddress` for ptrs; `int` for size/num_tiles; `asc.ConstExpr[int]` for tile size.
- Casts: `x.to(asc.float32)`, `y.to(x.dtype)`. Math: `asc2.erf`, `asc2.tanh`, `+ *`.
- Launch: `kernel[cores](x, out, size, num_tiles, TILE)` with `cores = min(72, num_tiles)`.
- Target: `Ascend950PR_9599`, 72 AIV cores, UB 253952 B.

## 3. All 20 cases (coverage)

1-D flatten + grid-stride + `real_shape` tails covers every rank (1D–5D present;
spec 0–8D) and every odd/non-multiple shape. No host shape branch needed.

| # | shape | dtype | range | mode | elems | path |
|---|---|---|---|---|---|---|
|1|[1024,1024]|f16|[-1,1]|erf|1.05M|erf|
|2|[2048,2048]|f32|[-2,2]|erf|4.19M|erf|
|3|[4096,4096]|bf16|[-3,3]|erf|16.78M|erf|
|4|[8192,8192]|f16|[-10,10]|tanh|67.1M|tanh|
|5|[8192,8192]|f32|[-100,100]|tanh|67.1M|tanh|
|6|[1023,1023]|bf16|[-0.1,0.1]|tanh|1.05M|tanh|
|7|[1009,1021]|f16|[-1,2]|erf|1.03M|erf|
|8|[1537,769]|f32|[-5,10]|tanh|1.18M|tanh|
|9|[363,367,373]|bf16|[-50,100]|erf|49.9M|erf|
|10|[2049,513]|f16|[-65504,65504]|tanh|1.05M|tanh|
|11|[3,7,13,4001]|f32|[-88,88]|erf|1.09M|erf|
|12|[1000003]|bf16|[-inf,inf]|tanh|1.00M|tanh|
|13|[11,13,17,67,67]|f32|[nan,nan]|erf|10.9M|erf|
|14|[3,7,11,13,1009]|f16|[0,0]|tanh|3.03M|tanh|
|15|[512,2049]|f32|[-0.5,0.5]|erf|1.05M|erf|
|16|[255,8193]|bf16|[-1,3]|erf|2.09M|erf|
|17|[4097,511]|f16|[-1000,1000]|tanh|2.09M|tanh|
|18|[2,511,2049]|f32|[-0.2,0.2]|erf|2.09M|erf|
|19|[4,255,2049]|bf16|[-3,6]|tanh|2.09M|tanh|
|20|[2,3,17,1024,101]|f32|[-20,40]|erf|10.5M|erf|

All cases ≥ ~1.0M elems ⇒ all take the wide tile path; narrow path retained only for
spec-completeness on tiny/empty tensors (`size == 0` → return `empty_like`).

## 4. Tiling & tails

Two kernels (separate JIT cache entries, minimal UB per path):
`_gelu_erf_kernel` and `_gelu_tanh_kernel`, each grid-stride:

```
for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):
    off = t * tile
    n = tile if off + tile <= size else size - off     # tail
    x = asc2.copy_in(x_gm, [off], [tile], real_shape=[n])
    xf = x.to(asc.float32)
    ...compute yf in f32...
    asc2.copy_out(yf.to(x.dtype), out_gm, [off], real_shape=[n])
```

Tiles (pinned, see §5):
- `_ERF_TILE  = 2048` (wide), `_NARROW = 1024`.
- `_TANH_TILE = 1024` (wide == narrow; tanh chain is longer).
- `_MAX_CORES = 72`; threshold `_MAX_CORES * _ERF_TILE = 147456`.

Dispatch:
- `size == 0` → `return torch.empty_like(x)`.
- erf: `size >= threshold` → TILE=2048 else 1024; `cores=min(72, ceildiv(size,TILE))`.
- tanh: TILE=1024 always; `cores=min(72, ceildiv(size,1024))`.

## 5. UB budget (253952 B, static alloc)

Formula (measured calibration from the sigmoid reference on this build): 
`real ≈ 1.6 * V * 4 * TILE * unroll(=2)`, where `V` = #live f32 tile values.

- **erf chain** `V ≈ 6` (`xf, s, e, one, hx, y`) + input-dtype load/store.
  erf@2048 u2: `1.6*6*4*2048*2 = 157286 B` ✓ (96 KB headroom; matches the proven
  sigmoid chain length, and gives margin for any hidden erf temporaries).
- **tanh chain** `V ≈ 10` (`xf, x2, x3, a, b, c, t, one, hx, y`).
  tanh@2048 u2: `1.6*10*4*2048*2 = 262144 B` ✗ (8 KB over ⇒ launch-time `UB overflow`).
  tanh@1536 u2: `1.6*10*4*1536*2 = 196608 B` ✓ (57 KB / 23% headroom).
  tanh@1024 u2: `1.6*10*4*1024*2 = 131072 B` ✓ (122 KB headroom).
  **Pinned: tanh TILE = 1024** — chosen for one-shot safety (no local NPU to probe
  hidden temporaries of the `tanh`/`mul` chain); 1536 is a perf fallback if a local
  compile gate were available. UB overflow is a **runtime** launch failure (not caught
  by the exact-v2 compile gate), so conservative is correct here.

256-byte alignment rule: `TILE*4 % 256 == 0` ⇒ `TILE % 64 == 0`; 2048 and 1024 both OK.

## 6. Numerical risks

- **x³ overflow**: max |x| in tanh cases is 65504 (case 10, f16→f32). `65504³ ≈ 2.8e14`,
  well within f32 (max ~3.4e38). case 17 (±1000) → 1e9. No overflow.
- **tanh saturation**: `tanh(large)=±1`; for large |x|, `K*x³` dominates, `x` is lost in
  `x + K*x³` (catastrophic cancellation) **but** tanh has already saturated to ±1, so
  `y = 0.5*x*(±1+1) = x or 0` — the cancellation is benign and matches golden, which
  uses the identical formula. No special-case branch.
- **erf saturation**: `erf(|x|/√2)`→±1 for |x|≳6 (cases 9,11,20). `asc2.erf` is a vector
  op on this build; saturation gives correct limits `y→x` (+∞), `y→0` (−∞).
- **Inf (case 12 bf16 [−inf,inf])**: IEEE propagation — `x=+inf ⇒ x³=inf ⇒ c=inf ⇒
  tanh=1 ⇒ y=+inf`; `x=−inf ⇒ c=−inf ⇒ tanh=−1 ⇒ 0.5*(−inf)*0 = NaN`. Golden
  (`torch.nn.functional.gelu`) computes the same expression and yields the same IEEE
  result ⇒ relative-error match (golden-equal). Documented as a residual `suspected`
  risk if the harness treats `NaN≠NaN`; cannot be special-cased without diverging from
  golden.
- **NaN (case 13 f32 [nan,nan])**: `erf(NaN)=NaN`, `0.5*NaN*(1+NaN)=NaN`. IEEE
  propagation through `asc2.erf`/mul ⇒ matches golden. Same residual NaN-comparison
  caveat as above.
- **Zero (case 14 [0,0] tanh)**: `erf/tanh(0)=0`, `y=0`. Exact.
- **Precision**: f16 thresh `2^-10≈9.7e-4`, bf16 `2^-7≈7.8e-3`, f32 `2^-13≈1.2e-4`.
  All f16/bf16 cases compute in f32 then cast back ⇒ strictly more accurate than the
  golden's f16/bf16 path; f32 cases use the cancellation-free erf/tanh forms with no
  `exp(+arg)` (no overflow) and no `log(1+tiny)` (no flush). `erf`/`tanh` are bounded,
  so no guard-band logic is needed.

## 7. Anti-cheat constraints

- ALL math (`erf`, `tanh`, `*`, `+`, casts) lives inside `@asc2.jit` kernels launched on
  the NPU. Host does only: `ensure_npu_platform()`, `.contiguous()`, `torch.empty_like`,
  `.numel()`, `.dtype`, string dispatch, launch, return.
- No torch compute/math ops anywhere (`torch.mul`, `F.gelu`, `a+b`, `.to(dtype)` on
  device data, `torch.cat/clone/sum`, `x.sigmoid()`…). No `data_ptr` caching.
- `approximate` string is consumed on the host only (never passed into a kernel — str
  is not a supported param type anyway).
- Output is a fresh contiguous NPU tensor (`torch.empty_like(x)`) with golden's exact
  shape/dtype; no views of the input.
- Constants precomputed module-level; no `import`/`math.*` inside jit.

## 8. Local validation ladder

No NPU and no pyasc/asc/asc2/torch_npu installed locally (per task.md). Ladder:

1. **Syntax only** — `python3 -m py_compile candidate.py` (allowed; checks AST, no
   imports execute). Run after writing candidate.
2. **Worker static contract check** — requires the CANNBench worker runtime; not
   runnable here ⇒ treated as `suspected`, deferred to remote.
3. **Exact-v2 local compile gate** (skill `references/local-validation.md`) — requires
   the pinned `pyasc-v2-source`; not installed here ⇒ `suspected`, deferred. UB
   overflow (§5) is a runtime launch failure that this gate would NOT catch anyway;
   the conservative tiles in §4 are chosen to avoid it a priori.
4. **camodel numerical compare vs golden.py** — no local camodel ⇒ `suspected`.
5. **CANNBench on real NPU** — the only acceptance/performance oracle; remote, invoked
   by the official harness.

Evidence label for the design phase: `suspected` (no local compile/numerical evidence
possible). Promotion to `verified-local-compile` / `verified-camodel` requires steps
2–4, which are unavailable in this environment.

## 9. Open implementation notes (for candidate.py, next phase)

- Two kernels: `_gelu_erf_kernel` and `_gelu_tanh_kernel`, identical signature
  `(x: asc.GlobalAddress, out: asc.GlobalAddress, size: int, num_tiles: int,
   tile: asc.ConstExpr[int])`.
- Host `gelu(x, approximate="none")`: `ensure_npu_platform()`; `if not x.is_contiguous():
  x = x.contiguous()`; `out = torch.empty_like(x)`; `size = x.numel()`; empty→return;
  branch on `approximate` (`"none"`→erf, `"tanh"`→tanh; any other value ⇒ assert, since
  spec allows only these two); pick tile per §4; `ceildiv` via `asc.ceildiv`;
  `cores = min(72, num_tiles)`; launch; return `out`.
- `x.is_contiguous()` / `.contiguous()` / `.numel()` / `.dtype` are metadata/views —
  allowed by the anti-cheat whitelist.

DESIGN_DONE
