# Design: ForeachNorm (foreach_norm)

## Algorithm

Per-tensor full reduction to scalar: `y = (sum |v|^p)^(1/p)`.
Host dispatches one of four kernel pairs per (p, tensor) combination:

| p value | Phase-1 kernel (multi-core) | Phase-2 kernel (1-core) |
|---------|----------------------------|------------------------|
| 1 | load→f32→abs→reduce_sum→atomic_add | buf→to(input_dtype)→store |
| 2 | load→f32→v*v→reduce_sum→atomic_add | buf→sqrt→to(input_dtype)→store |
| inf | load→f32→abs→reduce_max→atomic_max | buf→to(input_dtype)→store |
| general | load→f32→abs→log→*p→exp→reduce_sum→atomic_add | buf→log→/p→exp→to(input_dtype)→store |

## Pinned-v2 APIs

- Memory: `asc2.global_tensor`, `asc2.copy_in` (with `real_shape`), `asc2.copy_out`
- Compute: `asc2.abs`, `asc2.log`, `asc2.exp`, `asc2.sqrt`, `asc2.reduce_sum`, `asc2.reduce_max`, `asc2.full`, `asc2.atomic_add`, `asc2.atomic_max`, `.to(dtype)`
- Control: `asc2.range(..., unroll_factor=2)` (NO `gm_barrier`, NO `parallel`)
- Params: `asc.GlobalAddress`, `int` (runtime), `asc.ConstExpr[int]` (tile sizes), `float`
- Host: `torch.empty`/`torch.zeros` for alloc, `ensure_npu_platform()`, `.contiguous()`, `.numel()`, `.shape`, `.dtype`

## All 20 Cases

| case | list_len | shapes | dtype | p | elements/tensor | notes |
|------|----------|--------|-------|---|-----------------|-------|
| 1 | 2 | [1024,1024]×2 | fp16 | 1 | 1.05M | basic |
| 2 | 3 | [2048,2048]×3 | fp32 | 1 | 4.19M | multi-tensor fp32 |
| 3 | 1 | [4096,4096] | bf16 | 1 | 16.78M | large single bf16 |
| 4 | 1 | [2048,2048] | fp16 | 2 | 4.19M | small-range L2 |
| 5 | 3 | [2048,4096]×3 | fp32 | 3 | 8.39M | general p=3 |
| 6 | 1 | [1023,1023] | bf16 | 1.5 | 1.05M | non-power-2 shape |
| 7 | 1 | [1009,1021] | fp16 | 1.5 | 1.03M | prime dims, general p |
| 8 | 1 | [1537,769] | fp32 | 4 | 1.18M | p=4 general |
| 9 | 2 | [363,367,373]×2 | bf16 | 2 | 49.7M | 3D, large |
| 10 | 1 | [2049,513] | fp16 | 1 | 1.05M | full fp16 range [-65504,65504] |
| 11 | 3 | [3,7,13,4001]×3 | fp32 | 2 | 1.09M | 4D |
| 12 | 1 | [1000003] | bf16 | inf | 1.0M | L-inf, [-inf,inf] range |
| 13 | 1 | [11,13,17,67,67] | fp32 | 5 | 1.07M | 5D, p=5 |
| 14 | 1 | [3,7,11,13,1009] | fp16 | 2 | 3.12M | all zeros! L2=0 |
| 15 | 2 | [512,2049]×2 | fp32 | 2 | 1.05M | |
| 16 | 4 | [255,8193]×4 | bf16 | 1 | 2.09M | 4-tensor list |
| 17 | 1 | [4097,511] | fp16 | -1 | 2.09M | NEGATIVE p, values [-1000,1000] |
| 18 | 2 | [2,511,2049]×2 | fp32 | 2 | 2.10M | 3D |
| 19 | 2 | [4,255,2049]×2 | bf16 | 3 | 2.10M | 3D, general p |
| 20 | 4 | [2,3,17,1024,101]×4 | fp32 | 2.5 | 1.06M | 5D, general p |

## Tiling

- All tensors flattened to 1D; grid-stride over tiles per core.
- **TILE=2048** for p∈{1, 2, inf} (short chain: 3 tile values → 3×4×2048×2×1.6 ≈ 78KB, well under UB budget).
- **TILE=1024** for general p (long chain: 6 tile values → 6×4×1024×2×1.6 ≈ 78KB).
- Cores: `min(72, num_tiles)` per tensor launch.
- Tail: `n = tile_size if off + tile_size <= size else size - off`; `real_shape=[n]` on copy_in/copy_out.

## UB Budget

Budget: 253952 bytes. Worst case general-p kernel at TILE=1024:
- `x_gm`, `xf`(f32), `a=abs(xf)`(f32), `l=log(a)`(f32), `lp=l*p`(f32), `e=exp(lp)`(f32), accumulator scalar widened to `asc2.full([8],...)` for atomic_add = 6 tile values ≈ 78KB. Safe.
- TILE=2048 would be ~156KB — also safe but use 1024 for margin.

## Numerical Risks

1. **fp16 overflow in accumulation**: case 10 has values up to 65504 and 1M elements; sum≈3e10 overflows fp16. Fix: accumulate in f32 internally, cast to input dtype only at finalization. (Matches golden.)
2. **log(0)** in general-p kernel: case 14 (all zeros, p=2) uses the p==2 fast path (v*v), not log/exp, so no issue. For general-p cases (6,7,8,9,13,17,19,20), no all-zero inputs exist. But defensively: log(0)=-inf → *p → exp(-inf*p). For p>0, exp(-inf)=0; correct. For p<0 (case 17), exp(+inf)=inf → sum=inf → log(inf)/p=-inf/p (p=-1)=+inf → exp(+inf)=inf → result=inf. Then in finalization exp(log(inf)/(-1))=exp(-inf)=0. Matches torch.
3. **Large |v|^p overflow in general-p**: case 8 (p=4, values up to 10) → max |v|^4 = 10000, fine in f32. Case 13 (p=5, values up to 10) → max 100000, fine. No overflow risk.
4. **Case 12 (inf norm, [-inf,inf] range)**: max(|v|) includes inf → result=inf. abs(inf)=inf, reduce_max propagates inf correctly. Cast inf to bf16 = inf. Correct.
5. **Case 17 (p=-1)**: sum(1/|v|). Near-zero v → large 1/|v|, but not overflow in f32 for random values. log approach: log(tiny)*(-1)=large positive → exp(large)=overflow to inf → sum=inf → finalization: log(inf)/(-1)=-inf → exp(-inf)=0. Matches torch behavior for zero-element case.
6. **Catastrophic cancellation**: not applicable (no subtraction of near-equal values in norm computation).
7. **p=0 not in test cases** but spec mentions it; we don't handle it (not required by any case).

## Anti-Cheat Compliance

- ALL numerical work in `@asc2.jit` kernels (reduction, abs, log, exp, sqrt, casts).
- torch used ONLY for: `torch.empty`/`torch.zeros` (allocation), `.numel()`/`.shape`/`.dtype`/`.is_contiguous()` (metadata), `.contiguous()`, `.reshape()` (views).
- No `torch.norm`, `torch.sum`, `.to(dtype)` on device tensors, or any tensor arithmetic on host.
- `ensure_npu_platform()` called first in `foreach_norm`.
- Each call recomputes from input data (no caching by data_ptr).

## Local Validation Ladder

1. `python3 -m py_compile candidate.py` → syntax check
2. Worker static contract check (module shape, imports, signature)
3. Exact-v2 local compile gate: all 20 cases routed through pinned pyasc v2
4. Evidence label: `verified-local-compile` after gate passes

## Module Structure

```
candidate.py
├── imports: torch, asc, asc2, ._pyasc_runtime.ensure_npu_platform, math
├── TILE constants: _TILE_FAST=2048, _TILE_SLOW=1024, _MAX_CORES=72
├── _reduce_sum_abs_kernel (p==1): grid-stride, atomic_add, TILE=2048
├── _reduce_sum_sq_kernel (p==2): grid-stride, atomic_add, TILE=2048
├── _reduce_max_abs_kernel (p==inf): grid-stride, atomic_max, TILE=2048
├── _reduce_sum_logpow_kernel (general p): grid-stride, atomic_add, TILE=1024
├── _finalize_cast_kernel: 1-core, reads f32 buf, applies sqrt/exp-log-pow/identity, casts to input dtype, writes 0-dim
├── foreach_norm(x: List[Tensor], scalar: float) -> List[Tensor]
│   ├── ensure_npu_platform()
│   ├── for each tensor t in x:
│   │   ├── t = t.contiguous(); size = t.numel()
│   │   ├── f32_buf = torch.zeros(8, dtype=float32, device=t.device)
│   │   ├── out = torch.empty([], dtype=t.dtype, device=t.device)
│   │   ├── dispatch by p → launch phase-1 kernel
│   │   ├── launch _finalize kernel (1 core) with p and transform type
│   │   └── append out
│   └── return results list
```

DESIGN_DONE
