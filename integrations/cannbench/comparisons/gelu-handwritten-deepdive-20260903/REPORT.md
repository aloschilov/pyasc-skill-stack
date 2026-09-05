# Handwritten GeLU deep dive on CANNBench

Status: iteration 03 completed successfully on CANNBench: 20/20 cases passed,
with no anti-cheat failures. Correctness and launch coverage are complete; the
measured 0.4748x geometric-mean speedup remains below the >=1x target.

## Scope and provenance

- Upstream source: [`compiler-team/pyasc`, branch `v2`](https://gitcode.com/compiler-team/pyasc/tree/v2)
- Evaluated commit: `0a631f70968c3cb7c33ce45330a85768dd5a6f06`
- Hardware reported by CANNBench: `Ascend950PR_9589 x 1`
- Runtime: self-contained CPython 3.12/x86_64 wheel built from that commit
- CANNBench task: one `Gelu` operator, 20 cases spanning FP16, BF16, FP32,
  `approximate="none"|"tanh"`, aligned and prime tails, NaN, Inf, zero, and
  ranges through the dtype boundaries

The upstream handwritten target is an implementation reference, not a full
CANNBench solution. It has one FP32 shape, implements only a sigmoid restatement
of tanh GeLU, swaps the linear and cubic coefficients while testing against the
same swapped formula, and assumes a full-tile shape. The adapter therefore had
to repair the operator contract before performance tuning was meaningful.

## Official runs

| Iteration | CANNBench job | Result | What it established |
|---|---|---:|---|
| 01: native FP16 exact plus low-level erfc/tanh | [`job_24726123eebe`](https://cannbench.com/workspace/jobs/job_24726123eebe) | 2/20, score 6.2844, passed-case geomean 0.4269x | FP16 exact cases 1 and 7 are numerically correct. The other 18 routes never launched because the base `asc.Compiler` emitted an FFTS argument and the launcher failed in `GetC2cCtrlAddrWrapper` with 207000. |
| 02: C310 ABI repair plus VF/reuse tanh | [`job_bcf486ff6371`](https://cannbench.com/workspace/jobs/job_bcf486ff6371) | 3/20, score 8.5228, passed-case geomean 0.2897x | The ABI repair worked: exact cases 1–3 launched and passed. Tanh case 4 caused vector-core timeout 507034; cases 7–20 were cascade-skipped after the device became unrecoverable. |
| 03: safe low-level tanh | [`job_a375a6e244ca`](https://cannbench.com/workspace/jobs/job_a375a6e244ca) | 20/20, score 62.0711, geomean 0.4748x | The cancellation-free low-level tanh path eliminates the hardware timeout and the C310 ABI repair launches every specialization. All 20 cases pass, but none reaches the CANN baseline. |

The reported geomeans for failed jobs cover only cases that reached a valid
performance measurement. They are not full 20-case GeLU performance scores.
All three official jobs reported zero anti-cheat failures.

## Complete case-level result from iteration 03

All candidate times and speedups below are the official CANNBench hardware
measurements. `vs HW` compares against CANNBench's hardware-limit time rather
than its operator baseline.

| Case | Route | Candidate us | Baseline us | Speedup | vs HW |
|---:|---|---:|---:|---:|---:|
| 1 | `_gelu_exact_native` | 10.85 | 4.490 | 0.4138x | 0.1207x |
| 2 | `_gelu_exact_erfc_f32` | 51.03 | 15.370 | 0.3012x | 0.2056x |
| 3 | `_gelu_exact_erfc_bf16` | 206.00 | 30.140 | 0.1463x | 0.1018x |
| 4 | `_gelu_tanh_promoted_lowlevel` | 253.08 | 172.930 | 0.6833x | 0.3315x |
| 5 | `_gelu_tanh_f32` | 456.39 | 387.265 | 0.8485x | 0.3676x |
| 6 | `_gelu_tanh_promoted_lowlevel` | 6.30 | 4.460 | 0.7079x | 0.2079x |
| 7 | `_gelu_exact_native` | 10.98 | 4.450 | 0.4053x | 0.1175x |
| 8 | `_gelu_tanh_f32` | 9.04 | 6.110 | 0.6759x | 0.3263x |
| 9 | `_gelu_exact_erfc_bf16` | 602.00 | 117.530 | 0.1952x | 0.1032x |
| 10 | `_gelu_tanh_promoted_lowlevel` | 6.44 | 4.560 | 0.7081x | 0.2034x |
| 11 | `_gelu_exact_erfc_f32` | 15.98 | 6.050 | 0.3786x | 0.1708x |
| 12 | `_gelu_tanh_promoted_lowlevel` | 6.44 | 4.380 | 0.6801x | 0.1941x |
| 13 | `_gelu_exact_erfc_f32` | 131.86 | 37.455 | 0.2841x | 0.2069x |
| 14 | `_gelu_tanh_promoted_lowlevel` | 13.02 | 7.780 | 0.5975x | 0.2911x |
| 15 | `_gelu_exact_erfc_f32` | 15.73 | 5.980 | 0.3802x | 0.1666x |
| 16 | `_gelu_exact_erfc_bf16` | 27.54 | 6.230 | 0.2262x | 0.0948x |
| 17 | `_gelu_tanh_promoted_lowlevel` | 9.84 | 6.310 | 0.6413x | 0.2663x |
| 18 | `_gelu_exact_erfc_f32` | 27.87 | 8.860 | 0.3179x | 0.1880x |
| 19 | `_gelu_tanh_promoted_lowlevel` | 9.98 | 6.260 | 0.6273x | 0.2615x |
| 20 | `_gelu_exact_erfc_f32` | 130.29 | 36.195 | 0.2778x | 0.2024x |

The official aggregate is score `62.0711356991`, 20/20 accuracy, zero
anti-cheat failures, geometric-mean speedup `0.4748251724x`, and
geometric-mean speedup versus the hardware limit `0.1912868786x`. Per-route
geomeans are 0.4095x for native FP16 exact, 0.3207x for FP32 erfc exact,
0.1863x for promoted BF16 erfc exact, 0.6625x for promoted low-level tanh,
and 0.7573x for native FP32 tanh.

## Case-level result from iteration 02

| Case | Route | Candidate | Baseline | Speedup | Accuracy |
|---:|---|---:|---:|---:|---|
| 1 | FP16 exact, native `asctile.erf` | 10.54 us | 4.49 us | 0.4260x | pass |
| 2 | FP32 exact, stable low-level `erfc` | 51.79 us | 15.37 us | 0.2968x | pass |
| 3 | BF16 exact, FP32 low-level `erfc` | 206.09 us | 30.14 us | 0.1462x | pass |
| 4 | FP16 tanh, FP32 VF+reuse | device timeout | 172.93 us | 0x | no output |
| 5–6 | after case 4 | device error | — | 0x | no output |
| 7–20 | after device loss | cascade-skipped | — | 0x | not run |

## Root causes and repairs

### C310 low-level launch ABI

The base v2 `asc.Compiler._schedule_postprocessing` always calls
`add_legalize_kernel_args(..., set_ffts_addr=True)`. This adds a hidden FFTS
argument even for C310. On the CANNBench 950PR runtime the launcher then calls
`c2c_ctrl_addr()` and fails with 207000. The local adapter preserves the base
low-level compiler pipeline and changes only that condition to
`set_ffts_addr=(arch != C310)`. Iteration 02 proved the repair on hardware:
both low-level exact routes launched and passed. The compile evidence also
records `has_ffts_arg=false` for every specialization.

### AscTile JIT options

The v2 AscTile compiler defines `reuse_alloc`, `static_alloc`, and `vf_fusion`,
but its JIT class inherits option discovery and call-time extraction that use
the base `asc.CompileOptions`. Directly following the documented
`@asctile.jit(reuse_alloc=..., vf_fusion=...)` form therefore fails before
codegen. The local adapter selects `self.compiler.options_cls` consistently at
construction and call time. This made the tested allocation/VF settings real,
rather than labels in a prompt.

### Tanh vector-core timeout

Iteration 02 used 72 cores, a 13,824-element physical tile,
`vf_fusion=True`, and `reuse_alloc=1`. The compile gate accepted all three
dtypes and reported 221,184 bytes of UB, below the 253,952-byte limit. The
67M-element FP16 case nevertheless timed out on the vector core. The exact
root inside compiler/runtime code is not yet isolated; what is verified is
that compile/UB success was insufficient for this long fused loop. Iteration
03 removes that combination rather than shrinking benchmark coverage.

## Tiling and AI Core conclusions

- `72` cores are used for every benchmark case because all inputs contain
  enough 13,824-element tiles. This matches the upstream 950PR target launch
  and maximizes independent vector work; the observed shortfall is not caused
  by accidentally launching 16 or 32 cores.
- A 13,824-element tile keeps at least 72 tiles even for the smallest ~1M
  inputs, while substantially reducing loop/DMA setup relative to 4,096.
- For FP16 exact, `vf_fusion=True, reuse_alloc=1` reduced reported UB from
  248,832 bytes at tile 6,912 to 110,592 bytes at tile 13,824. Hardware time,
  however, remained about 10.5 us; the baseline is 4.49 us. Wider tiling alone
  therefore did not close the gap.
- Stable low-level `erfc` is much faster than the previous nine-coefficient
  AscTile approximation (for FP32 case 2, 51.79 us versus the earlier
  136.06 us), but still only 0.2968x of the CANNBench baseline. BF16 promotion
  and conversion are more expensive still.
- Iteration 03 confirms these conclusions over the full hardware matrix. Tanh
  is the strongest route (0.5975x--0.8485x), while exact FP32 is
  0.2778x--0.3802x and promoted exact BF16 is 0.1463x--0.2262x. The best
  individual case is still below 1x, so the >=1x expectation was not met.
- The remaining gap is not a coverage, tail, or under-subscription failure:
  every case launches 72 cores with the 13,824-element tiling and passes.
  Current pyasc v2 must compose GeLU from several vector operations and must
  promote BF16 for unsupported arithmetic/transcendentals. It cannot bind the
  fused `AscendC::Gelu` primitive exposed by the installed CANN headers. This
  fused-GeLU binding gap and BF16 promotion are therefore the measured
  performance limitations to address next.

## Offline native-primitive probe

After iteration 03 was frozen, a skills-driven OpenCode review identified
`asc.adv.tanh` as the only untested low-level transcendental likely to shorten
the tanh path. A separate compile-only probe exercised native-dtype
`asc.adv.erfc` and `asc.adv.tanh` across the same 20-case dispatch matrix. FP16
and FP32 lowered through the repaired C310 ABI, producing compact 2.1--3.2 KB
AscendC sources. BF16 failed in the first basic vector multiplication because
current v2 accepts only FP16/FP32/int16/int32 there; the existing FP32
promotion is therefore required rather than accidental overhead.

The native FP16 tanh formula was also checked against Torch over every FP16
benchmark range. It violates the CANNBench MARE bound on `[-10, 10]` because
`1 + tanh(u)` cancels in the negative tail. A native FP16 erfc variant showed
the same risk near tiny exact outputs. Neither route replaces the queued
cancellation-free iteration 03. The probe is evidence about API feasibility,
not NPU correctness or speed.

The installed CANN headers expose a fused `AscendC::Gelu` implementation, but
the pinned pyasc v2 dialect/Python API has no corresponding operation. That
missing binding is now the leading explanation for the remaining gap to the
single-kernel CANN baseline: the available pyasc routes must compose several
vector operations and, for BF16, two conversions. This is a documented v2
performance blocker, not yet a measured causal proof.

## Next run

Iteration 03 is immutable at the hash recorded in `MANIFEST.json`; its local
20/20 compile/ABI evidence and official 20/20 CANNBench result now agree. The
GeLU-first gate is complete. The queue can proceed to the missing four-operator
`with_skills` arm, followed by the full-current-v2 six-operator campaign and
the five-operator remainder.
