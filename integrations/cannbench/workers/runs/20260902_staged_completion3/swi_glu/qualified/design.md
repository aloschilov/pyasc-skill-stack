# SwiGlu Design

## Algorithm

output = SiLU(x0) * x1 where (x0, x1) = chunk(input, 2, dim).
SiLU(v) = v * sigmoid(v). All compute promoted to f32, cast back to input dtype.

## Pinned-v2 APIs

- `asc2.global_tensor(ptr, [dims])` — 1-D or 2-D GM views
- `asc2.copy_in(gm, [off], [tile], real_shape=[n])` — load with tail
- `asc2.copy_out(tile, gm, [off], real_shape=[n])` — store with tail
- `asc2.range(start, stop, step, unroll_factor=2)` — grid-stride tile loop (NO `gm_barrier`, NO `parallel`)
- Tile math: `+ - * /`, `asc2.exp`, `.to(dtype)`, comparisons, `asc2.where`
- Scalar-on-right rule: `x * 0.5`, never `0.5 * x`
- `asc.ConstExpr[int]` for tile_size (compile-time)
- `asc2.jit` (bare) for kernel, no `always_compile`

## Case Coverage (all 20)

| Group | Cases | Key traits |
|-------|-------|------------|
| f16 aligned 2-D, dim=-1 | 1,7 | outer=rows, half_cols=C/2, standard 2-D path |
| f32 aligned 2-D, dim=-1 | 2,5,15 | same path, wider value range (case 5: ±100) |
| bf16 aligned 2-D, dim=-1 | 3 | same path |
| f16 dim=0 | 4,17 | outer=1, C=shape[0], inner=prod(shape[1:]); 2-D [1, C*inner] |
| bf16 dim=0 | 6,16 | same |
| f32 dim=0 | 8 | outer=1, [1538,1537] dim=0 -> C=1538, inner=1537, half=769*1537 |
| bf16 3-D dim=-1 | 9 | outer=363*367, C=14, half_cols=7 |
| f16 extreme range | 10 | [-65504,65504], dim=1, shape [2049,1024] -> outer=2049, C=1024, half=512 |
| f32 4-D dim=-1 | 11 | outer=3*7*13, C=1018, half=509 |
| **bf16 degenerate** | **12** | **[1000003,2] dim=1 -> half_cols=1, 2B < 32 -> 1-D fallback** |
| f32 NaN input | 13 | [11,13,16,67] dim=2 -> outer=11*13, C=16, inner=67, half=8*67=536 |
| f16 zeros | 14 | [3,7,11,13,1012] dim=-1, value=0, trivially correct |
| f16 2-D dim=0 | 17 | [4096,1023] outer=1, C=4096, inner=1023, half=2048*1023 |
| f32 3-D dim=-1 | 18 | [2,1023,4096] outer=2*1023, C=4096, half=2048 |
| bf16 3-D dim=1 | 19 | [4,510,4097] outer=4, C=510, inner=4097, half=255*4097 |
| f32 5-D dim=-1 | 20 | [2,3,17,512,100] outer=2*3*17*512, C=100, half=50 |

## Tiling

**Primary path (2-D zero-copy)**:
- Host computes: outer = prod(shape[:dim]), C = shape[dim], inner = prod(shape[dim+1:]), half_cols = (C//2)*inner, full_cols = C*inner
- View input as contiguous 1-D, kernel uses `global_tensor(ptr, [outer, full_cols])`
- Single kernel with grid-stride over linearized tiles: total_elements = outer * half_cols
- TILE = 2048 (f32), grid-stride over flat element index
- For element index `i`: row = i // half_cols, col_in_half = i % half_cols
  - x0 at [row, col_in_half], x1 at [row, half_cols + col_in_half]
- Output stored at flat index i in [outer * half_cols] output buffer

**Fallback 1-D path** (case 12 and any half_cols*elem_size < 32):
- Host: `x0 = input.narrow(dim, 0, C//2).contiguous()`, same for x1
- Flat 1-D elementwise kernel over size = x0.numel()
- Two copy-in kernels for x0/x1 contiguity (perf cost only on case 12)

**TILE selection**: TILE=2048, two tiers: wide (2048) when total >= 72*2048, narrow (1024) otherwise.

## Tails

- All grid-stride loops compute `n = tile_size if off + tile_size <= size else size - off`
- `real_shape=[n]` on every `copy_in`/`copy_out`
- No host-side padding

## UB Budget

Visible f32 values per tile (2-D path):
1. x0_tile (f16/bf16 load) — not f32
2. x1_tile (f16/bf16 load) — not f32
3. x0f = x0.to(f32) — f32
4. x1f = x1.to(f32) — f32
5. neg_x0f = -x0f — f32
6. exp_neg = exp(neg_x0f) — f32
7. denom = 1 + exp_neg — f32 (or fused)
8. sig = x0f * stable_sigmoid — f32
9. silu_x0 = sig (reused) — f32
10. result = silu_x0 * x1f — f32
11. out = result.to(input_dtype) — not f32

~7 visible f32 tiles × 4 × 2048 = 57344 B visible. With 1.6× overhead: ~91750 B. Well within 253952 B.
With unroll_factor=2: ~183500 B. Still within budget. **TILE=2048 is safe.**

For exp extreme: stable sigmoid uses `exp(min(s,0))` and `exp(-|s|)`, which may introduce an extra tile for abs/min. Budget: ~9 f32 tiles × 4 × 2048 × 1.6 × 2 ≈ 236000 B. Tight but within. If UB overflows, fall back to TILE=1024.

## Numerical Risks

1. **Stable sigmoid**: `sig(s) = exp(min(s, 0)) / (1 + exp(-abs(s)))` — avoids exp overflow for large positive s.
2. **Case 10** (f16 ±65504): sigmoid(65504) saturates to 1.0, silu(65504) ≈ 65504. In f32 promotion this is exact. Cast back to f16 is correct.
3. **Case 12** (bf16 ±inf): sigmoid(±inf) → 1 or 0. silu(inf) = inf*1 = inf. silu(-inf) = -inf*0 → NaN (IEEE 0×inf). Golden also produces NaN here. IEEE propagation is correct.
4. **Case 13** (NaN input): NaN propagates through all ops. Golden also produces NaN. Correct.
5. **Case 5** (f32 ±100): silu(100) = 100·sig(100) ≈ 100. Well within f32 range.
6. **Precision**: all compute in f32, cast back. Meets f16 threshold 2^-10, bf16 2^-7, f32 2^-13.

## Anti-Cheat Constraints

- ALL numerical work in `@asc2.jit` kernels on NPU
- torch used only for: `ensure_npu_platform()`, `.contiguous()`, `.numel()`, `.shape`, `.dtype`, `.element_size()`, `.narrow()` (metadata/views), `torch.empty()`/`torch.empty_like()` (allocation)
- No torch math ops, no `.to(dtype)` on device tensors, no `torch.cat`, no `torch.sum`
- Output is freshly allocated contiguous NPU tensor, correct shape/dtype
- No caching, no data_ptr tricks

## Local Validation Ladder

1. `python3 -m py_compile candidate.py` — syntax gate
2. Static contract check — worker harness
3. exact-v2 local compile gate — all 20 cases must compile through pinned pyasc v2
4. camodel numerical verification (different model reviews and compares to golden)
5. CANNBench on real NPU (submission, remote only)

Evidence labels: `verified-local-compile` after step 3, `verified-camodel` after step 4, `verified-cannbench` after step 5.

## Module Structure

```
imports (torch, asc, asc2, ensure_npu_platform)
constants (_WIDE_TILE=2048, _NARROW_TILE=1024, _MAX_CORES=72)

@asc2.jit _swiglu_2d_kernel(x_ptr, out_ptr, outer, full_cols, half_cols, total, num_tiles, tile_size: ConstExpr)
  — grid-stride, 2-D global_tensor, load x0/x1 by computed offset, f32 compute, cast back

@asc2.jit _swiglu_1d_kernel(x0_ptr, x1_ptr, out_ptr, size, num_tiles, tile_size: ConstExpr)
  — flat 1-D fallback for degenerate layouts

def swi_glu(input, dim=-1):
  — ensure_npu_platform, contiguous, compute outer/C/inner/half_cols
  — branch: degenerate → .narrow().contiguous() + 1-D kernel
  — else: 2-D zero-copy kernel
  — return output
```

DESIGN_DONE
