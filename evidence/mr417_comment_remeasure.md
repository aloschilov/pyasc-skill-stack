## NPU remeasurement — all #417 target-operator tests on `Ascend950PR_9599`

Second independent NPU run (~5 h after the first) to confirm stability. **86 / 86 cases PASS**, golden 100%. Per-shape medians are within ~1–3 % of the first run and all geomeans are unchanged, so the numbers below are the clean, confirmed results.

### Environment

- **Device:** `Ascend950PR_9599` (real NPU, `/dev/davinci0`)
- **asc2 build:** current `v2` (with the `reuse_alloc` JIT API) — #417 tests run as-committed (`reuse_alloc=1`, no downgrade)
- **CANN toolkit:** `cann-9.1.0`
- **pyasc measurement:** `pytest --backend NPU --platform Ascend950PR_9599 --profile --runs 10` (median of 10 launches, µs)
- **reduce_max tiling:** 192 KB / 72-core v2 host-side tiling selector

### Result summary

| operator | cases | golden | geomean ratio (CANN / pyasc) | run-1 → run-2 |
|---|---|---|---|---|
| reciprocal | 20 | PASS | 0.656 | 0.686 → 0.656 |
| reduce_max | 11 | PASS | **1.032** (≈ CANN parity) | 1.033 → 1.032 |
| addcdiv | 23 | PASS | 0.761 | 0.750 → 0.761 |
| one_hot | 32 (16 shapes × static/dynamic) | PASS | 0.020 | 0.020 → 0.020 |

`ratio = CANN_CST_µs / pyasc_µs` — **> 1 means pyasc is faster than the CANN reference**. Geomeans are over the cases that have a CANN reference.

### CANN reference sources

- **reciprocal / reduce_max / addcdiv** — CANN static (CST) references from the merged source MRs' `*_selected_representative_with_perf.xlsx` workbooks (#352 / #353 / #354).
- **one_hot** — #359 carried no CANN reference, so it was generated with **TTK (tbetoolkits)** for the CANN `OneHot` AICORE op (`-c=true -d=false -s=false -b=release --core-type=VectorCore --run=10`, CST mode, real device, `axis=-1`). `CST_PERF` is the on-device CANN kernel time.

### N/A cells

The 3 big (numel ≥ 5e6) reciprocal and 3 big reduce_max cases were only added "for HW measurement" in the source MRs and have no CANN CST reference there (ratio N/A; pyasc HW numbers still reported). addcdiv's 3 big cases do have CANN references and are included.

---

## Per-operator comparison (pyasc NPU median vs CANN CST)
### reciprocal

| input shape | pyasc µs (NPU) | CANN CST µs | ratio (CANN/pyasc) |
|---|---|---|---|
| [1024] | 1.544 | 1.683 | 1.090 |
| [2400] | 2.516 | 1.742 | 0.692 |
| [16, 5, 1, 64] | 2.661 | 1.793 | 0.674 |
| [16, 256] | 2.044 | 1.850 | 0.905 |
| [16, 320] | 2.668 | 1.832 | 0.687 |
| [16, 24, 768] | 4.832 | 3.298 | 0.683 |
| [128, 1, 2304] | 4.829 | 3.314 | 0.686 |
| [2500] | 2.529 | 1.783 | 0.705 |
| [1200] | 1.850 | 1.622 | 0.877 |
| [2048] | 1.643 | 1.776 | 1.081 |
| [1500] | 1.891 | 1.672 | 0.884 |
| [1024, 1, 20] | 5.988 | 1.859 | 0.310 |
| [1024, 1, 50] | 13.261 | 2.087 | 0.157 |
| [1024, 1, 1000] | 13.615 | 7.956 | 0.584 |
| [256, 1] | 1.342 | 1.498 | 1.116 |
| [100, 14, 10] | 7.786 | 1.938 | 0.249 |
| [2048, 1] | 1.629 | 1.800 | 1.105 |
| [1024, 6144] | 71.492 | N/A | N/A |
| [8192, 1024] | 96.171 | N/A | N/A |
| [2048, 8192] | 193.980 | N/A | N/A |

geomean ratio (n=17): **0.656**

### reduce_max (192 KB / 72-core v2 tiling)

| input shape | pyasc µs (NPU) | CANN CST µs | ratio (CANN/pyasc) |
|---|---|---|---|
| [200, 10] | 3.102 | 2.010 | 0.648 |
| [13, 2048, 32] | 3.999 | 4.430 | 1.108 |
| [10, 2048, 64] | 4.998 | 5.853 | 1.171 |
| [45, 2048, 4] | 3.527 | 3.971 | 1.126 |
| [64, 2048, 8] | 4.452 | 4.614 | 1.036 |
| [70, 2048, 16] | 8.382 | 8.133 | 0.970 |
| [2048, 83, 18] | 12.431 | 15.656 | 1.259 |
| [1500, 1, 61] | 2.193 | 2.357 | 1.075 |
| [3072, 113, 24] | 22.922 | N/A | N/A |
| [4608, 115, 12] | 24.937 | N/A | N/A |
| [1500, 61, 61] | 16.945 | N/A | N/A |

geomean ratio (n=8): **1.032** — the last-axis reduction is at ≈ CANN parity with the new tiling.

### addcdiv

| input shape | pyasc µs (NPU) | CANN CST µs | ratio (CANN/pyasc) |
|---|---|---|---|
| [11734, 16] | 5.954 | 4.534 | 0.762 |
| [152] | 1.334 | 1.431 | 1.073 |
| [152, 456] | 3.311 | 2.571 | 0.777 |
| [1, 168] | 1.335 | 1.652 | 1.237 |
| [7, 10] | 1.353 | 1.184 | 0.875 |
| [8] | 1.339 | 1.580 | 1.180 |
| [80] | 1.351 | 1.475 | 1.092 |
| [98166, 16] | 37.219 | 15.149 | 0.407 |
| [1024] | 1.348 | 1.652 | 1.226 |
| [1, 14, 1] | 1.335 | 1.160 | 0.869 |
| [1024, 152] | 4.826 | 3.992 | 0.827 |
| [421] | 1.323 | 1.522 | 1.150 |
| [256, 320] | 3.384 | 2.947 | 0.871 |
| [8, 64] | 1.339 | 1.635 | 1.221 |
| [1, 40] | 1.356 | 1.172 | 0.864 |
| [64, 121] | 2.901 | 1.804 | 0.622 |
| [48] | 1.335 | 1.158 | 0.867 |
| [1024, 1024] | 25.239 | 11.309 | 0.448 |
| [64, 225, 1] | 4.993 | 1.976 | 0.396 |
| [16, 16, 1] | 1.333 | 1.411 | 1.059 |
| [1820039, 16] | 708.623 | 299.062 | 0.422 |
| [315511, 16] | 118.900 | 41.332 | 0.348 |
| [98166, 128] | 307.071 | 116.080 | 0.378 |

geomean ratio (n=23): **0.761**

### one_hot (pyasc = `static` median; `dynamic` within ~0.1%)

| input shape | depth | pyasc µs (NPU) | CANN CST µs | ratio (CANN/pyasc) |
|---|---|---|---|---|
| [1, 1, 593, 1, 1] | 31 | 142.256 | 3.976 | 0.028 |
| [1, 997] | 64 | 187.964 | 4.445 | 0.024 |
| [1, 1, 1, 1, 1, 3712, 1] | 3511 | 461.369 | 16.364 | 0.035 |
| [1, 9216] | 2 | 154.783 | 2.755 | 0.018 |
| [9600] | 2 | 144.669 | 2.697 | 0.019 |
| [1, 1024, 2, 4, 6] | 4 | 156.040 | 3.982 | 0.026 |
| [1, 65536] | 2 | 176.951 | 3.203 | 0.018 |
| [1, 1, 1, 1, 1, 4793, 28] | 184 | 697.951 | 38.008 | 0.054 |
| [2328, 1, 1, 1, 1, 101, 1] | 1 | 558.034 | 4.700 | 0.008 |
| [2, 16, 256, 256] | 2 | 5435.770 | 27.905 | 0.005 |
| [359, 167, 1, 1, 163] | 1 | 22915.508 | 86.958 | 0.004 |
| [42767, 7, 16, 16] (fp16) | 2 | 193145.819 | 1139.620 | 0.006 |
| [1259, 1, 192, 2, 127] | 3 | 162678.231 | 1332.027 | 0.008 |
| [800, 1] | 2 | 111.415 | 2.397 | 0.022 |
| [1, 1] | 7 | 1.407 | 2.212 | 1.572 |
| [65536] | 2 | 177.458 | 2.864 | 0.016 |

geomean ratio (n=16): **0.020** — the current one_hot kernel (per-element scalar loop, `unroll_factor=1`) is ~50× slower than CANN's `OneHot`; functional golden is correct on all cases. Clear perf-tuning target.

---

**Takeaways (confirmed across two runs):** reciprocal and addcdiv sit at ~0.66–0.76× of CANN (competitive, several small shapes already faster); **reduce_max is at CANN parity** with the 192 KB/72-core tiling; **one_hot is the main perf-tuning target** (~50× off). All 86 cases functionally correct (golden PASS).
