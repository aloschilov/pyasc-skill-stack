## Review comments addressed + all-core tiling extended + full CANN coverage

Updated `combined-target-ops` (head `28276777`) to address the outstanding review comments on the merged target-op MRs and to re-measure everything on real NPU (Ascend950PR_9599). **one_hot has been dropped from this MR** and is tracked separately (it needs a fundamental algorithm remaster, extended cases already live in #359).

### What changed (per review)

- **Renames**: `reciprocal_kernel_1D` -> `reciprocal`, `addcdiv_kernel_1D` -> `addcdiv`, `reduce_max_last_axis` -> `reduce_max_d_last_axis`.
- **JIT decorator aligned to current v2 defaults**: dropped `static_alloc` (defaults to `True` on C310) and `parallel` (no longer an `asc2.range` argument in v2 - loop overlap is driven by `unroll_factor`); kept `reuse_alloc=1` (default is `0`, so it is a real setting).
- **No host padding / no last-block tail branch**: all three ops now follow the canonical `test_vadd.py` pattern with an in-kernel bounds-aware loop. This also fixes the last-core out-of-bounds concern raised on reduce_max - `copy_in(pad_value=...)` + `copy_out` clamping instead of host row-padding.
- **All-core / UB-budget tiling** extended from reduce_max to `reciprocal` and `addcdiv`: `block_num = min(72, ceildiv(n, tile))`, tiles sized to the 192 KB UB with double buffering and live-tensor count, STATIC + DYNAMIC variants.
- **dtype coverage**: added representative fp16 cases to reciprocal/addcdiv. Note: `asc2.div` does not accept `bfloat16` on this target, so bf16 division cases are intentionally omitted.
- **reduce_max middle-axis reduction** added (new capability + reviewer cases). `inner==1` middle-axis cases are routed to the last-axis kernel (drops `[1024,100,2,1]` from ~612 us to ~11 us).
- `yapf` applied to all three files.

### Measurements (real NPU, `--profile --runs 10`, 100% golden PASS)

pyasc us = STATIC variant median; ratio = CANN CST / pyasc (>1 means pyasc is faster). **All previously-N/A CANN cells are now filled** via on-HW TTK CST (tbetoolkits, `-c=true --core-type=VectorCore --run=10`).

### reciprocal (44/44 PASS)

| input shape | dtype | pyasc us | CANN CST us | ratio |
|---|---|---|---|---|
| [1024] | f32 | 1.605 | 1.683 | 1.049 |
| [2400] | f32 | 1.679 | 1.742 | 1.038 |
| [16, 5, 1, 64] | f32 | 1.808 | 1.793 | 0.992 |
| [16, 256] | f32 | 1.749 | 1.850 | 1.058 |
| [16, 320] | f32 | 1.809 | 1.832 | 1.013 |
| [16, 24, 768] | f32 | 3.212 | 3.298 | 1.027 |
| [128, 1, 2304] | f32 | 3.214 | 3.314 | 1.031 |
| [2500] | f32 | 1.689 | 1.783 | 1.056 |
| [1200] | f32 | 1.687 | 1.622 | 0.961 |
| [2048] | f32 | 1.659 | 1.776 | 1.071 |
| [1500] | f32 | 1.724 | 1.672 | 0.970 |
| [1024, 1, 20] | f32 | 2.839 | 1.859 | 0.655 |
| [1024, 1, 50] | f32 | 2.897 | 2.087 | 0.720 |
| [1024, 1, 1000] | f32 | 4.921 | 7.956 | 1.617 |
| [256, 1] | f32 | 1.366 | 1.498 | 1.097 |
| [100, 14, 10] | f32 | 2.449 | 1.938 | 0.791 |
| [2048, 1] | f32 | 1.654 | 1.800 | 1.088 |
| [1024, 6144] | f32 | 19.926 | 27.726 | 1.391 |
| [8192, 1024] | f32 | 26.972 | 35.814 | 1.328 |
| [2048, 8192] | f32 | 61.247 | 65.548 | 1.070 |
| [128, 2, 512] | f16 | 2.889 | 2.463 | 0.853 |
| [1024, 6144] | f16 | 12.089 | 26.112 | 2.160 |

geomean ratio (n=22): **1.057**

### addcdiv (50/50 PASS)

| input shape | dtype | pyasc us | CANN CST us | ratio |
|---|---|---|---|---|
| [11734, 16] | f32 | 3.264 | 4.534 | 1.389 |
| [152] | f32 | 1.286 | 1.431 | 1.113 |
| [152, 456] | f32 | 2.762 | 2.571 | 0.931 |
| [1, 168] | f32 | 1.295 | 1.652 | 1.276 |
| [7, 10] | f32 | 1.153 | 1.184 | 1.027 |
| [8] | f32 | 1.141 | 1.580 | 1.385 |
| [80] | f32 | 1.149 | 1.475 | 1.284 |
| [98166, 16] | f32 | 13.589 | 15.149 | 1.115 |
| [1024] | f32 | 1.585 | 1.652 | 1.042 |
| [1, 14, 1] | f32 | 1.101 | 1.160 | 1.054 |
| [1024, 152] | f32 | 2.957 | 3.992 | 1.350 |
| [421] | f32 | 1.596 | 1.522 | 0.954 |
| [256, 320] | f32 | 2.703 | 2.947 | 1.090 |
| [8, 64] | f32 | 1.590 | 1.635 | 1.028 |
| [1, 40] | f32 | 1.159 | 1.172 | 1.011 |
| [64, 121] | f32 | 1.899 | 1.804 | 0.950 |
| [48] | f32 | 1.155 | 1.158 | 1.003 |
| [1024, 1024] | f32 | 9.832 | 11.309 | 1.150 |
| [64, 225, 1] | f32 | 2.367 | 1.976 | 0.835 |
| [16, 16, 1] | f32 | 1.352 | 1.411 | 1.044 |
| [1820039, 16] | f32 | 297.318 | 299.062 | 1.006 |
| [315511, 16] | f32 | 41.495 | 41.332 | 0.996 |
| [98166, 128] | f32 | 115.403 | 116.080 | 1.006 |
| [1024, 1024] | f16 | 5.659 | 9.688 | 1.712 |
| [98166, 128] | f16 | 51.618 | 64.606 | 1.252 |

geomean ratio (n=25): **1.106**

### reduce_max (30/30 PASS)

| input shape | axis | dtype | pyasc us | CANN CST us | ratio |
|---|---|---|---|---|---|
| [200, 10] | -1 | f32 | 2.002 | 2.010 | 1.004 |
| [13, 2048, 32] | -1 | f32 | 4.040 | 4.430 | 1.097 |
| [10, 2048, 64] | -1 | f32 | 5.038 | 5.853 | 1.162 |
| [45, 2048, 4] | -1 | f32 | 8.184 | 3.971 | 0.485 |
| [64, 2048, 8] | -1 | f32 | 4.360 | 4.614 | 1.058 |
| [70, 2048, 16] | -1 | f32 | 8.662 | 8.133 | 0.939 |
| [2048, 83, 18] | -1 | f32 | 16.176 | 15.656 | 0.968 |
| [1500, 1, 61] | -1 | f32 | 2.463 | 2.357 | 0.957 |
| [3072, 113, 24] | -1 | f32 | 23.644 | 22.535 | 0.953 |
| [4608, 115, 12] | -1 | f32 | 40.492 | 39.086 | 0.965 |
| [1500, 61, 61] | -1 | f32 | 21.648 | 16.926 | 0.782 |
| [1, 128, 144] | 1 | f32 | 2.188 | 2.374 | 1.085 |
| [1024, 100, 2, 1] | 2 | f32 | 11.062 | 3.807 | 0.344 |
| [64, 32, 48] | 1 | f32 | 2.134 | 2.360 | 1.106 |
| [8, 4, 2, 64] | 2 | f32 | 1.586 | 1.822 | 1.149 |

geomean ratio (n=15): **0.897** (middle-axis reduction is new functional coverage; the small `[1024,100,2,1]` and `[45,2048,4]` cases remain the outliers vs the hand-tuned CANN kernel).

All measurements on Ascend950PR_9599; pyasc and CANN measured on the same box.
