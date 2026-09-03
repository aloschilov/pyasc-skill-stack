# RmsNorm — Design

## 1. Algorithm

**y = x / sqrt(mean(x²) + ε) · γ** over the last dimension D.

Two-pass streaming per row (avoids UB overflow that killed the prior tile_d=2048 single-pass at 328 608 B):

- **Pass 1 (reduce):** stream D in TILE_D chunks → accumulate Σx² in f32 scalar `acc`.
- **Scalar compute:** `mean_sq = acc · inv_D`; `inv_rms = rsqrt(mean_sq + epsilon)`.
- **Pass 2 (apply):** re-stream D chunks, reload x and γ, emit `y = x · inv_rms · γ`, cast back to input dtype, store.

Reload-from-GM in pass 2 trades memory bandwidth for UB headroom; bandwidth is not the scoring bottleneck.

## 2. Pinned-v2 APIs

| Purpose | Call |
|---|---|
| Global views | `asc2.global_tensor(ptr, [size])` — 1-D for x/out (flattened S·D), 1-D for γ `[D]` |
| Load | `asc2.copy_in(gm, [offset], [TILE_D], real_shape=[n])` |
| Store | `asc2.copy_out(tile, gm, [offset], real_shape=[n])` |
| Cast | `tile.to(asc.float32)` / `tile.to(x.dtype)` |
| Reduce | `asc2.reduce_sum(tile)` → scalar |
| Scalar seed | `acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))` |
| Scalar fold | `acc = acc + asc2.reduce_sum(xf * xf)` inside `asc2.range(n)` |
| Inv-sqrt | `asc2.rsqrt(scalar)` |
| Ops | `xf * xf`, `xf * inv_rms`, `norm * gf` (scalar on **right** for `· inv_rms`, tile·tile otherwise) |
| Iteration | `asc2.range(...)` — never Python `range` over runtime values |

No `asc2.range(gm_barrier=...)` (confirmed absent). No `break`/`continue`/early `return`.

## 3. Case Matrix (all 20)

| # | S (rows) | D | dtype | ε | value range | key stress |
|---|---|---|---|---|---|---|
| 1 | 4096 | 768 | f16 | 1e-6 | [-1,1] | standard |
| 2 | 4096 | 1024 | f32 | 1e-6 | [-2,2] | D = tile |
| 3 | 4096 | 2048 | bf16 | 1e-6 | [-3,3] | 2-chunk D |
| 4 | 4096 | 4096 | f16 | 1e-6 | [-10,10] | 4-chunk D |
| 5 | 4096 | 8192 | f32 | 1e-6 | [-100,100] | 8 chunks, large x² |
| 6 | 4092 | 4097 | bf16 | 1e-5 | [-5,5] | prime D tail |
| 7 | 4221 | 1023 | f16 | 1e-8 | [-0.1,0.1] | tiny x, small ε |
| 8 | 8176 | 2049 | f32 | 1e-4 | [-1,1] | prime D tail |
| 9 | 8168 | 4099 | bf16 | 1e-12 | [-0.5,0.5] | extreme ε |
| 10 | 4191 | 769 | f16 | 1e-6 | [-1,2] | prime D |
| 11 | 4019 | 2049 | f32 | 1e-6 | [-50,100] | large range |
| 12 | 4335 | 4097 | bf16 | 1e-6 | [-3,6] | prime D |
| 13 | 7063 | 1021 | f16 | 1e-7 | [-1,1] | prime D |
| 14 | 4037 | 373 | f32 | 1e-5 | [-10,10] | short D |
| 15 | **1000003** | **2** | bf16 | 1e-6 | [None,None] | massive S, tiny D |
| 16 | 2431 | 67 | f16 | 1e-8 | [None,None] | tiny D, 4-D input |
| 17 | 231 | 4096 | f32 | 1e-4 | [0,0] | **all-zero rows** |
| 18 | 1022 | 8192 | bf16 | 1e-6 | [-0.2,0.2] | small S, big D |
| 19 | 1020 | 4096 | f16 | 1e-3 | **[-65504,65504]** | fp16 extrema |
| 20 | 102 | 128 | f32 | 1e-6 | [-20,40] | 5-D, small S |

S ranges 102…1 000 003; D ranges 2…8192; ranks 2–5.

## 4. Tiling

- **Outer:** grid-stride `for r in asc2.range(asc2.block_idx(), S, asc2.block_num())` — each core grabs non-contiguous rows.
- **Inner:** `for dt in asc2.range(num_d_tiles)` — sequential D chunks per row (no `unroll_factor`; keeps UB single-buffered).
- **TILE_D = 1024** (ConstExpr). Host selects: `tile_d = 1024` always. For D=2, n=2 real elements in a 1024-slot tile (wastes bandwidth, not UB).

### Tail handling
```
off_d = dt * tile_d
n = tile_d if off_d + tile_d <= D else D - off_d
```
`real_shape=[n]` on both `copy_in` and `copy_out`. No host padding, no `gm_barrier`.

## 5. UB Budget (TILE_D=1024, unroll=1)

Measured calibration factor from prior iteration: **1.6×** over naive visible-byte count.

| Tile variable | dtype | naive bytes |
|---|---|---|
| x (pass 1) | f16 | 1024·2 = 2048 |
| xf (pass 1 cast) | f32 | 1024·4 = 4096 |
| x2 = xf·xf | f32 | 1024·4 = 4096 |
| x (pass 2 reload) | f16 | 2048 |
| γ | f16 | 2048 |
| xf (pass 2) | f32 | 4096 |
| γf | f32 | 4096 |
| y = xf·inv·γf | f32 | 4096 |
| y_out (cast back) | f16 | 2048 |

Sum naive ≈ **26 624 B** (f16/bf16 worst case); ×1.6 = **~42 600 B**. f32 path: 7 f32 tiles × 4 096 ≈ 28 672 B ×1.6 = **~45 875 B**. **Well under 253 952 B.** Headroom allows TILE_D=2048 as a perf-tuning option later if verified.

For comparison, the **failed iteration** at TILE_D=2048 unroll=2 consumed 328 608 B — this design's single-buffered streaming cuts usage by ≈ 6×.

## 6. Numerical Risks & Mitigations

| Risk | Case(s) | Mitigation |
|---|---|---|
| x² overflows f16 (65504² ≈ 4.3e9) | 19 | Promote to f32 **before** squaring: `xf = x.to(asc.float32)` |
| Σx² overflows f32 | 5 (100²·8192 = 8.2e7) | Fits f32 (max 3.4e38); no issue |
| All-zero row → rsqrt(ε) | 17 | ε > 0 always; rsqrt well-defined; 0·inv = 0 ✓ |
| ε = 1e-12 + tiny mean_sq | 9 | f32 subnormal min ≈ 1.4e-45 ≪ 1e-12; no flush-to-zero |
| Inf/NaN propagation | all | IEEE: inf² = inf; rsqrt(inf) = 0; inf·0 = NaN — matches golden's torch.nn.functional.rms_norm |
| Catastrophic cancellation | n/a | No subtraction in this algorithm |
| Precision threshold (f16: 2⁻¹⁰ ≈ 9.8e-4) | all | Full f32 internal compute; only store casts down |

`inv_D = 1.0 / D` computed on host as `float`, passed as kernel float param — avoids int-division inside JIT.

## 7. Anti-Cheat Compliance

- ✅ All math in `@asc2.jit` kernel(s) on NPU
- ✅ torch used only for: `torch.empty_like`, `.shape`, `.numel()`, `.dtype`, `.is_contiguous()`, `.contiguous()`, `.view`/`.reshape`
- ✅ No `torch.nn.functional.rms_norm`, no tensor arithmetic, no `.to(dtype)` on device data
- ✅ Output is freshly allocated contiguous tensor
- ✅ `ensure_npu_platform()` called first
- ✅ No caching by `data_ptr`

## 8. Kernel Architecture

```
host rms_norm(x, gamma, epsilon=1e-6):
    ensure_npu_platform()
    x = x.contiguous() if needed
    D = x.shape[-1]; S = x.numel() // D
    out = torch.empty_like(x)
    tile_d = _TILE_D = 1024
    num_d_tiles = (D + tile_d - 1) // tile_d
    inv_D = 1.0 / float(D)
    cores = min(72, S)
    _rms_norm_kernel[cores](x, gamma, out, S, D, num_d_tiles, epsilon, inv_D, tile_d)
    return out

@asc2.jit
_rms_norm_kernel(x_ptr, gamma_ptr, out_ptr,
                 S: int, D: int, num_d_tiles: int,
                 epsilon: float, inv_D: float,
                 tile_d: asc.ConstExpr[int]):
    x_gm  = asc2.global_tensor(x_ptr,     [S * D])
    g_gm  = asc2.global_tensor(gamma_ptr, [D])
    o_gm  = asc2.global_tensor(out_ptr,   [S * D])

    for r in asc2.range(asc2.block_idx(), S, asc2.block_num()):
        row = r * D
        # ---- pass 1: reduce ----
        acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))
        for dt in asc2.range(num_d_tiles):
            od = dt * tile_d
            n  = tile_d if od + tile_d <= D else D - od
            x  = asc2.copy_in(x_gm, [row + od], [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            acc = acc + asc2.reduce_sum(xf * xf)
        inv_rms = asc2.rsqrt(acc * inv_D + epsilon)

        # ---- pass 2: apply ----
        for dt in asc2.range(num_d_tiles):
            od = dt * tile_d
            n  = tile_d if od + tile_d <= D else D - od
            x  = asc2.copy_in(x_gm,  [row + od], [tile_d], real_shape=[n])
            g  = asc2.copy_in(g_gm,  [od],      [tile_d], real_shape=[n])
            xf = x.to(asc.float32)
            gf = g.to(asc.float32)
            y  = xf * inv_rms * gf
            asc2.copy_out(y.to(x.dtype), o_gm, [row + od], real_shape=[n])
```

One kernel, one JIT specialization at TILE_D=1024. Host handles all rank/dtype dispatch via metadata only.

## 9. Local Validation Ladder

| Gate | Command | Evidence label |
|---|---|---|
| 1. Syntax | `python3 -m py_compile candidate.py` | — |
| 2. Static contract | worker contract check | — |
| 3. Exact-v2 compile | local compile gate × 20 cases | `verified-local-compile` |
| 4. Camodel numerical | (requires hardware) | `verified-camodel` |
| 5. CANNBench | official harness | `verified-cannbench` |

Gate 3 is the critical fix from iteration 1 (all 20 failed with UB overflow). This design targets ~43–46 KB UB usage → should clear 253 952 B on every case.

DESIGN_DONE
