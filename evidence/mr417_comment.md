## NPU hardware run — all #417 target-operator tests on `Ascend950PR_9599`

Ran the full #417 test set on **real NPU hardware** (not the camodel simulator) and compared per-shape latency against CANN reference values.

### Environment

- **Device:** `Ascend950PR_9599` (real NPU, `/dev/davinci0`)
- **asc2 build:** current `v2` (branch `v2`, with the `reuse_alloc` JIT API), used unmodified — the #417 tests run as-committed (`reuse_alloc=1`, no downgrade)
- **CANN toolkit:** `cann-9.1.0`
- **pyasc measurement:** `pytest --backend NPU --platform Ascend950PR_9599 --profile --runs 10` (median of 10 launches, µs)
- **reduce_max tiling:** the branch was updated to the 192 KB / 72-core v2 host-side tiling selector before measurement (replaces the earlier degenerate tiling)

### Result summary

**86 / 86 cases PASS** (golden 100% on every case):

| operator | cases | golden | geomean ratio (CANN / pyasc) |
|---|---|---|---|
| reciprocal | 20 | PASS | 0.686 |
| reduce_max | 11 | PASS | **1.033** (≈ CANN parity) |
| addcdiv | 23 | PASS | 0.750 |
| one_hot | 32 (16 shapes × static/dynamic) | PASS | 0.020 |

`ratio = CANN_CST_µs / pyasc_µs` — **> 1 means pyasc is faster than the CANN reference**, < 1 means slower. Geomeans are over the cases that have a CANN reference.

### CANN reference sources

- **reciprocal / reduce_max / addcdiv** — CANN static (CST) references taken from the merged source MRs' `*_selected_representative_with_perf.xlsx` workbooks (#352 / #353 / #354; reduce_max also cross-checked against the inline table in #353).
- **one_hot** — #359 carried no CANN reference, so it was generated here with **TTK (tbetoolkits)** for the CANN `OneHot` AICORE op: `run.sh <manifest> -c=true -d=false -s=false -b=release --core-type=VectorCore --run=10` (CST mode, real device, `axis=-1`/depth-innermost to match the kernel's actual output layout). `CST_GOLD` is `GOLDEN_FAILURE` because TTK feeds random on/off scalars, but `CST_PERF` is the real on-device CANN kernel time, which is what the reference needs.

### N/A cells

The 3 big (numel ≥ 5e6) reciprocal cases (`[1024,6144]`, `[8192,1024]`, `[2048,8192]`) and the 3 big reduce_max cases (`[3072,113,24]`, `[4608,115,12]`, `[1500,61,61]`) were only ever added "for HW measurement" in the source MRs and have no CANN CST reference there, so their ratio is N/A (pyasc HW numbers are still reported below). addcdiv's 3 big cases do have CANN references (measured in #354) and are included.

---

## Per-operator comparison (pyasc NPU median vs CANN CST)
### reciprocal

| input shape | pyasc µs (NPU) | CANN CST µs | ratio (CANN/pyasc) |
|---|---|---|---|
| [1024] | 1.429 | 1.683 | 1.178 |
| [2400] | 2.411 | 1.742 | 0.723 |
| [16, 5, 1, 64] | 2.477 | 1.793 | 0.724 |
| [16, 256] | 1.921 | 1.850 | 0.963 |
| [16, 320] | 2.479 | 1.832 | 0.739 |
| [16, 24, 768] | 4.774 | 3.298 | 0.691 |
| [128, 1, 2304] | 4.780 | 3.314 | 0.693 |
| [2500] | 2.433 | 1.783 | 0.733 |
| [1200] | 1.878 | 1.622 | 0.864 |
| [2048] | 1.435 | 1.776 | 1.238 |
| [1500] | 1.878 | 1.672 | 0.890 |
| [1024, 1, 20] | 5.842 | 1.859 | 0.318 |
| [1024, 1, 50] | 13.055 | 2.087 | 0.160 |
| [1024, 1, 1000] | 12.927 | 7.956 | 0.615 |
| [256, 1] | 1.304 | 1.498 | 1.149 |
| [100, 14, 10] | 7.698 | 1.938 | 0.252 |
| [2048, 1] | 1.439 | 1.800 | 1.251 |
| [1024, 6144] | 70.856 | N/A | N/A |
| [8192, 1024] | 95.612 | N/A | N/A |
| [2048, 8192] | 195.582 | N/A | N/A |

geomean ratio (n=17): **0.686**

### reduce_max (192 KB / 72-core v2 tiling)

| input shape | pyasc µs (NPU) | CANN CST µs | ratio (CANN/pyasc) |
|---|---|---|---|
| [200, 10] | 2.979 | 2.010 | 0.675 |
| [13, 2048, 32] | 4.021 | 4.430 | 1.102 |
| [10, 2048, 64] | 5.035 | 5.853 | 1.162 |
| [45, 2048, 4] | 3.542 | 3.971 | 1.121 |
| [64, 2048, 8] | 4.432 | 4.614 | 1.041 |
| [70, 2048, 16] | 8.521 | 8.133 | 0.954 |
| [2048, 83, 18] | 12.377 | 15.656 | 1.265 |
| [1500, 1, 61] | 2.206 | 2.357 | 1.068 |
| [3072, 113, 24] | 23.485 | N/A | N/A |
| [4608, 115, 12] | 25.502 | N/A | N/A |
| [1500, 61, 61] | 16.576 | N/A | N/A |

geomean ratio (n=8): **1.033** — the new tiling brings the last-axis reduction to ≈ CANN parity (the earlier degenerate tiling was ~100× slower).

### addcdiv

| input shape | pyasc µs (NPU) | CANN CST µs | ratio (CANN/pyasc) |
|---|---|---|---|
| [11734, 16] | 5.680 | 4.534 | 0.798 |
| [152] | 1.349 | 1.431 | 1.061 |
| [152, 456] | 3.363 | 2.571 | 0.764 |
| [1, 168] | 1.370 | 1.652 | 1.206 |
| [7, 10] | 1.377 | 1.184 | 0.860 |
| [8] | 1.368 | 1.580 | 1.155 |
| [80] | 1.381 | 1.475 | 1.068 |
| [98166, 16] | 36.842 | 15.149 | 0.411 |
| [1024] | 1.355 | 1.652 | 1.219 |
| [1, 14, 1] | 1.370 | 1.160 | 0.847 |
| [1024, 152] | 4.935 | 3.992 | 0.809 |
| [421] | 1.368 | 1.522 | 1.113 |
| [256, 320] | 3.394 | 2.947 | 0.868 |
| [8, 64] | 1.364 | 1.635 | 1.199 |
| [1, 40] | 1.373 | 1.172 | 0.854 |
| [64, 121] | 3.120 | 1.804 | 0.578 |
| [48] | 1.377 | 1.158 | 0.841 |
| [1024, 1024] | 25.690 | 11.309 | 0.440 |
| [64, 225, 1] | 5.223 | 1.976 | 0.378 |
| [16, 16, 1] | 1.357 | 1.411 | 1.040 |
| [1820039, 16] | 716.206 | 299.062 | 0.418 |
| [315511, 16] | 117.810 | 41.332 | 0.351 |
| [98166, 128] | 300.395 | 116.080 | 0.386 |

geomean ratio (n=23): **0.750**

### one_hot (pyasc = `static` median; `dynamic` within ~0.1%)

| input shape | depth | pyasc µs (NPU) | CANN CST µs | ratio (CANN/pyasc) |
|---|---|---|---|---|
| [1, 1, 593, 1, 1] | 31 | 142.489 | 3.976 | 0.028 |
| [1, 997] | 64 | 187.914 | 4.445 | 0.024 |
| [1, 1, 1, 1, 1, 3712, 1] | 3511 | 462.621 | 16.364 | 0.035 |
| [1, 9216] | 2 | 153.404 | 2.755 | 0.018 |
| [9600] | 2 | 144.271 | 2.697 | 0.019 |
| [1, 1024, 2, 4, 6] | 4 | 155.674 | 3.982 | 0.026 |
| [1, 65536] | 2 | 174.250 | 3.203 | 0.018 |
| [1, 1, 1, 1, 1, 4793, 28] | 184 | 695.171 | 38.008 | 0.055 |
| [2328, 1, 1, 1, 1, 101, 1] | 1 | 556.479 | 4.700 | 0.008 |
| [2, 16, 256, 256] | 2 | 5341.452 | 27.905 | 0.005 |
| [359, 167, 1, 1, 163] | 1 | 22854.275 | 86.958 | 0.004 |
| [42767, 7, 16, 16] (fp16) | 2 | 190064.309 | 1139.620 | 0.006 |
| [1259, 1, 192, 2, 127] | 3 | 160749.300 | 1332.027 | 0.008 |
| [800, 1] | 2 | 109.376 | 2.397 | 0.022 |
| [1, 1] | 7 | 1.387 | 2.212 | 1.595 |
| [65536] | 2 | 174.261 | 2.864 | 0.016 |

geomean ratio (n=16): **0.020** — the current one_hot kernel emits depth-sized tiles inside a per-element scalar loop (`unroll_factor=1`), so it is ~50× slower than CANN's `OneHot`. Clear perf-tuning target; functional golden is correct on all cases.

---

**Takeaways:** reciprocal and addcdiv are within ~0.7–0.75× of CANN (competitive, some small shapes already faster); **reduce_max reaches CANN parity** with the new 192 KB/72-core tiling; **one_hot is the main perf-tuning target** (~50× off, per-element scalar loop). All 86 cases are functionally correct (golden PASS).
