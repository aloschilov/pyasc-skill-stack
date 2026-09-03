# Design: ForeachNorm

## Skills loaded
- `pyasc-cannbench-kernel`
- `pyasc-syntax-constraints`
- `pyasc-api-patterns` (Pattern R — parallel reduction)

---

## 1. Algorithm by norm order `p`

Per tensor, full reduction to a 0-dim scalar:

| p | Elementwise | Reduction | Final |
|---|------------|-----------|-------|
| 1.0 | `abs(x)` | sum | identity |
| 2.0 | `x * x` | sum | `sqrt(S)` |
| inf | `abs(x)` | max | identity |
| general (incl. -1, 1.5, 2.5, 3, 4, 5) | `exp(log(abs(x)) * p)` | sum | `exp(log(S) / p)` |

**Inf handling**: log(0) = -inf propagates correctly through exp; no clamping needed.

---

## 2. Pinned-v2 APIs

| Layer | API |
|-------|-----|
| Global views | `asc2.global_tensor(ptr, [size])` — always 1-D (flatten) |
| Load/Store | `asc2.copy_in(src, [off], [TILE], real_shape=[n])` / `asc2.copy_out(...)` |
| Cast | `.to(asc.float32)` promote; `.to(dtype)` demote |
| Elem ops | `asc2.abs`, `asc2.exp`, `asc2.log`, `asc2.sqrt`, `asc2.mul`, tile arith with scalar-right |
| Reduction | `asc2.reduce_sum(t)`, `asc2.reduce_max(t)` — returns scalar |
| Cross-core | `asc2.atomic_add(src_tile, dst_gm, [offset])`, `asc2.atomic_max(...)` |
| Broadcast | `asc2.full([N], scalar, dtype=asc.float32)` |
| Loop | `asc2.range(start, stop, step, unroll_factor=2)` — NO `gm_barrier` |

---

## 3. All 20 cases — coverage matrix

| Case | list_len | shape | numel | dtype | p | Special |
|------|----------|-------|-------|-------|---|---------|
| 1 | 2 | 1024×1024 | 1.05M | f16 | 1.0 | |
| 2 | 3 | 2048×2048 | 4.19M | f32 | 1.0 | |
| 3 | 1 | 4096×4096 | 16.78M | bf16 | 1.0 | max single-tensor |
| 4 | 1 | 2048×2048 | 4.19M | f16 | 2.0 | small range |
| 5 | 3 | 2048×4096 | 8.39M×3 | f32 | 3.0 | general p |
| 6 | 1 | 1023×1023 | 1.05M | bf16 | 1.5 | odd dims |
| 7 | 1 | 1009×1021 | 1.03M | f16 | 1.5 | non-power-of-2 |
| 8 | 1 | 1537×769 | 1.18M | f32 | 4.0 | |
| 9 | 2 | 363×367×373 | 49.7M | bf16 | 2.0 | large 3D |
| 10 | 1 | 2049×513 | 1.05M | f16 | 1.0 | full f16 range |
| 11 | 3 | 3×7×13×4001 | 1.09M×3 | f32 | 2.0 | 4D, high range |
| 12 | 1 | 1000003 | ~1M | bf16 | inf | ±inf values |
| 13 | 1 | 11×13×17×67×67 | ~113M | f32 | 5.0 | 5D, large |
| 14 | 1 | 3×7×11×13×1009 | ~3.18M | f16 | 2.0 | all-zero input |
| 15 | 2 | 512×2049 | 1.05M×2 | f32 | 2.0 | |
| 16 | 4 | 255×8193 | 2.09M×4 | bf16 | 1.0 | |
| 17 | 1 | 4097×511 | 2.09M | f16 | -1.0 | negative p |
| 18 | 2 | 2×511×2049 | 2.10M×2 | f32 | 2.0 | |
| 19 | 2 | 4×255×2049 | 2.09M×2 | bf16 | 3.0 | |
| 20 | 4 | 2×3×17×1024×101 | ~10.6M×4 | f32 | 2.5 | 5D |

**Boundary notes:**
- Max numel: case 13 (~113M), case 9 (~49.7M), case 3 (~16.8M)
- Odd dims (1023, 1009×1021, 1537×769, 1000003) — require `real_shape` tail logic
- Case 14: all-zero input under p=2.0 → sum=0, sqrt(0)=0 (safe)
- Case 17: p=-1.0 with elements near 0 → log(0)=-inf, exp(-inf*-1)=exp(inf)=inf → correct propagation
- Case 12: inf/-inf values under p=inf → reduce_max(abs) = inf (correct)

---

## 4. Tiling

**Single flat-1D kernel per tensor. Grid-stride loop over tiles.**

```
for t in asc2.range(block_idx(), num_tiles, block_num(), unroll_factor=2):
    off = t * TILE
    n = TILE if off + TILE <= size else size - off
    x = copy_in(src_gm, [off], [TILE], real_shape=[n])
    ... compute elementwise → partial ...
    acc = acc + reduce_op(partial)
atomic_add(full([8], acc, dtype=...), out_gm, [0])
```

**TILE choice:**

Per p, visible intermediates in f32:

| p path | Visible f32 values | Naive bytes | At TILE=1024 (×1.6) |
|--------|-------------------|-------------|---------------------|
| p=1 | abs(xf) | 4×1024 = 4KB | 6.3 KB |
| p=2 | xf*xf | 4KB | 6.3 KB |
| p=inf | abs(xf) | 4KB | 6.3 KB |
| general | abs(xf), log(abs(xf)), log_abs*p, exp(...) | 4×4KB = 16KB | 25.2 KB |

UB budget: 253952 B. Single-core with unroll_factor=2 doubles live tiles. General path: ~25 KB × 2 = 50 KB + overhead ≈ 80 KB. Well within budget.

**Selected TILE=2048** for p∈{1,2,inf} (light chain, ~12 KB real), **TILE=1024** for general p (heavier chain, ~50 KB real).

Both compile to separate JIT cache entries via `asc.ConstExpr[int]`.

---

## 5. Tail handling

Every `copy_in` / `copy_out` uses `real_shape=[n]` where `n = TILE if off+TILE <= size else size - off`. No host-side padding. The last tile on odd-size tensors just loads fewer elements; UB beyond `n` is uninitialized and never written back (real_shape gates the store width).

---

## 6. UB budget — worst case

Worst: general p path at TILE=1024 with unroll_factor=2.

Intermediates (all f32, 1024 elements each): xf, abs_xf, log_abs, log_abs_p, exp_term → 5 tiles. Plus acc (scalar), full([8],...) for atomic → negligible.

Naive: 5 × 4 × 1024 × 2 (unroll) = 40,960 B
With 1.6× compiler overhead: ~65,536 B. Well under 253,952 B. Safe.

---

## 7. Numerical risks & mitigations

| Risk | Case | Mitigation |
|------|------|------------|
| log(0) = -inf in general path | 14 (zeros), 17 (p=-1) | Let -inf propagate: exp(-inf * p) = 0 for p>0, inf for p<0; exp(log(0)/p) = exp(-inf) = 0 for S=0, giving norm=0. Matches golden. |
| Overflow in x*x for p=2 | 10 (f16 range ±65504), 13 (±10) | Promote to f32 first; x*x in f32 won't overflow for |x|<sqrt(3.4e38) ≈ 1.8e19 |
| Overflow in exp(log_abs*p) | p=5, |x|=88 → log(88)*5 = 22.4, exp(22.4)=5.2e9 | OK in f32 (max ~3.4e38). For p=5, |x|=1000003 → log(1e6)*5≈70, exp(70)=2.5e30, still OK |
| Underflow / zero S for all-zero input | 14 | sum=0, final=sqrt(0)=0 or exp(log(0)/2)=0. Correct. |
| Cancellation in sqrt(sum) | none | sum is always non-negative; no cancellation |
| Large p + moderate x: exp(log_abs*p) overflow | none in 20 cases | max |x|×max p = 1000003×5 → log~14, ×5=70, exp=2.5e30 (OK). Even case 5: |x|=0.1, p=3 → log(0.1)*3=-6.9, exp≈0.001 (fine) |
| p=inf with inf input | 12 | reduce_max(abs(inf)) = inf. Correct. |
| Negative p | 17 (p=-1) | log_abs*p with p<0 and x near 0: abs≈0, log(0)=-inf, -inf*-1=inf, exp(inf)=inf. S=inf, log(S)/p = log(inf)/-1 = inf/-1 = -inf, exp(-inf)=0. Correct per golden. |

**Precision thresholds:**
- f16: MERE < 2^-10 ≈ 9.77e-4, MARE < 2^-9
- bf16: MERE < 2^-7 ≈ 7.81e-3, MARE < 2^-6
- f32: MERE < 2^-13 ≈ 1.22e-4, MARE < 2^-12

All accumulation in f32 → single precision sufficient for all 20 cases within these thresholds (max 113M elements at f32 → ~1e8 × 2^-24 rounding noise ≈ 6, dominated by reduction order, MERE still < 1e-4).

---

## 8. Anti-cheat constraints

- **All numerical compute in `@asc2.jit` kernels** — no torch math ops
- torch allowed ONLY: `torch.empty`, `torch.zeros`, `.numel()`, `.shape`, `.dtype`, `.contiguous()`, `.view()`, `.is_contiguous()`
- **FORBIDDEN**: `torch.norm`, `.to(dtype)` on device tensors (except for output allocation pattern), `torch.sum`, elementwise torch ops
- `ensure_npu_platform()` called at top of public function
- Output tensors must be contiguous NPU tensors, not views of inputs
- Host dispatch is pure Python (list iteration, scalar comparison for branch on p)

---

## 9. Host dispatch structure

```
foreach_norm(x: List[Tensor], scalar: float) -> List[Tensor]:
    ensure_npu_platform()
    for each tensor t in x:
        t = t.contiguous()
        size = t.numel()
        out = torch.empty((), dtype=t.dtype, device=t.device)  # 0-dim
        out8 = torch.zeros([8], dtype=..., device=...)  # for atomic accumulator
        if scalar == inf:  → _norm_max_kernel[num_tiles](...)
        elif scalar == 1.0: → _norm_sum_kernel[tile_sum, num_tiles](...)  # reduce=abs+sum
        elif scalar == 2.0: → _norm_sum_kernel[tile_sq, num_tiles](...)  # reduce=sq+sum
        else:               → _norm_sum_kernel[tile_gen, num_tiles](...)  # reduce=exp(log*abs*p)+sum
        # Second pass: single-core final kernel reads out8[0], applies sqrt/log/exp/power, writes out
        _final_scalar_kernel[1](out8, out, final_op)
        results.append(out.reshape(()))
    return results
```

**Per-tensor kernels** (since each tensor independently reduces to a scalar):
- Core 0 zeros `out8` via `torch.zeros` (pre-launch, allowed)
- All cores grid-stride accumulate into `out8[0..7]` via `atomic_add`
- Single-core final pass: read `out8[0]`, apply post-transform, write `out`

---

## 10. Local validation ladder

1. **Step 1**: `python3 -m py_compile candidate.py` — syntax check only
2. **Step 2**: Worker static contract check (module shape, imports, no torch compute)
3. **Step 3**: Exact-v2 local compile gate — `python3 -m pyasc.tools.compile candidate.py` against all 20 case shapes; record pass/fail per case
4. **Gate label**: `verified-local-compile` only until camodel execution confirms numerical accuracy

No correctness claims without `verified-camodel` or `verified-cannbench` evidence.

---

## 11. Kernel inventory (4 kernels)

| Kernel | Elem chain | Reduce | TILE | Final |
|--------|-----------|--------|------|-------|
| `_k_sum_abs` | abs(xf) | reduce_sum | 2048 | identity |
| `_k_sum_sq` | xf*xf | reduce_sum | 2048 | sqrt |
| `_k_max_abs` | abs(xf) | reduce_max | 2048 | identity |
| `_k_sum_gen` | exp(log(abs(xf)) * p) | reduce_sum | 1024 | exp(log(S)/p) |

Plus one `_k_final` single-core kernel for post-transform (sqrt, or exp(log(S)/p)), or inline for p=1/p=inf cases.

---

## 12. Syntax constraints verified

- ✅ `asc2.range(...)` — only `unroll_factor` kwarg used; NO `gm_barrier` kwarg
- ✅ Plain `acc` (Python scalar) updated via `acc = acc + reduce_sum(...)` inside `asc2.range` loop — matches verified pattern
- ✅ No `break`/`continue`/`print`/`import`/`return-value` inside `@asc2.jit`
- ✅ Scalar-right binary ops: `tile * p`, not `p * tile`
- ✅ Device function returns allowed (for inner helpers if needed), kernel function does not return
- ✅ No `math.log` inside JIT — use `asc2.log`; no `math.sqrt` — use `asc2.sqrt`
- ✅ `asc.ConstExpr[int]` for TILE (compile-time)

---

DESIGN_DONE
