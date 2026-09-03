# RmsNorm — Design

## Algorithm

`y = x * gamma * rsqrt(mean(x^2) + eps)` along the last dimension D.

Split into **two passes per row** inside one kernel:
1. **Reduction pass** — stream D in `TILE_D` chunks, upcast to f32, accumulate
   `ss += reduce_sum(x_f32 * x_f32)`.
2. **Normalize pass** — re-stream D, load gamma, compute
   `y = x_f32 * gamma_f32 * broadcast(rsqrt(ss/D + eps))`, cast back, store.

`ss` is a loop-carried scalar seeded with
`asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))` (proven pattern).

## Pinned-v2 APIs

| Purpose | API |
|---------|-----|
| Global view | `asc2.global_tensor(ptr, [size])` — 1-D, rank-1 throughout |
| Load | `asc2.copy_in(gm, [off], [TILE_D], real_shape=[n])` |
| Store | `asc2.copy_out(tile, gm, [off], real_shape=[n])` |
| Grid stride | `asc2.range(block_idx(), S, block_num(), unroll_factor=2)` (outer row loop) |
| Inner D loop | `asc2.range(num_d_tiles)` (no unroll — reduction, not pipelined) |
| Reduce | `asc2.reduce_sum(tile)` → scalar |
| Math | `asc2.rsqrt(tile)`, `.to(asc.float32)`, `.to(dt)`, `* + /` |
| Broadcast | `asc2.full([TILE_D], scalar, dtype=asc.float32)` |

No `gm_barrier` kwarg on `asc2.range` (raises TypeError — verified on this build).
No `break`/`continue`/early `return`/`print`/imports inside JIT.
No Python `range()` over runtime values — use `asc2.range`.

## All 20 cases

| # | shape | dtype | D | S | value range | eps | Notes |
|---|-------|-------|---|---|-------------|-----|-------|
| 1 | [32,128,768] | f16 | 768 | 4096 | [-1,1] | 1e-6 | D<TILE, 1 chunk |
| 2 | [32,128,1024] | f32 | 1024 | 4096 | [-2,2] | 1e-6 | D<TILE, 1 chunk |
| 3 | [32,128,2048] | bf16 | 2048 | 4096 | [-3,3] | 1e-6 | D=TILE, 1 chunk |
| 4 | [16,256,4096] | f16 | 4096 | 4096 | [-10,10] | 1e-6 | 2 chunks |
| 5 | [8,512,8192] | f32 | 8192 | 4096 | [-100,100] | 1e-6 | 4 chunks; x^2 peak 1e4, sum ~3.3e7, f32 OK |
| 6 | [4,1023,4097] | bf16 | 4097 | 4092 | [-5,5] | 1e-5 | 3 chunks, tail=1 |
| 7 | [63,67,1023] | f16 | 1023 | 4221 | [-0.1,0.1] | 1e-8 | 1 chunk, very small eps |
| 8 | [16,511,2049] | f32 | 2049 | 8176 | [-1,1] | 1e-4 | 2 chunks, tail=1 |
| 9 | [8,1021,4099] | bf16 | 4099 | 8168 | [-0.5,0.5] | 1e-12 | 3 chunks; near-zero eps (still >0) |
| 10 | [33,127,769] | f16 | 769 | 4191 | [-1,2] | 1e-6 | 1 chunk |
| 11 | [31,129,2049] | f32 | 2049 | 3999 | [-50,100] | 1e-6 | 2 chunks; x^2 peak 1e4 |
| 12 | [17,255,4097] | bf16 | 4097 | 4335 | [-3,6] | 1e-6 | 3 chunks |
| 13 | [7,1009,1021] | f16 | 1021 | 7063 | [-1,1] | 1e-7 | 1 chunk |
| 14 | [11,367,373] | f32 | 373 | 4037 | [-10,10] | 1e-5 | 1 chunk |
| 15 | [1000003,2] | bf16 | 2 | 1000003 | randn | 1e-6 | S massive, D tiny; tail=2 in 2048 tile |
| 16 | [11,13,17,67] | f16 | 67 | 2431 | randn | 1e-8 | 4-D input, 1 chunk |
| 17 | [3,7,11,4096] | f32 | 4096 | 231 | [0,0]! | 1e-4 | all-zero input; y=0 (0*gamma*100=0); 2 chunks |
| 18 | [2,511,8192] | bf16 | 8192 | 1022 | [-0.2,0.2] | 1e-6 | 4 chunks |
| 19 | [4,255,4096] | f16 | 4096 | 1020 | **[-65504,65504]** | 1e-3 | x^2 up to 4.3e9 — overflows f16, must f32-promote; 2 chunks |
| 20 | [2,3,17,1024,128] | f32 | 128 | 104448 | [-20,40] | 1e-6 | 5-D input; 1 chunk |

## Tiling strategy

- **TILE_D = 2048** (ConstExpr). D=2..8192 maps to 1–4 chunks.
- Grid dimension = **S** (flattened leading dims). Grid-stride rows across cores.
- Inner dimension = D in `num_d_tiles = ceildiv(D, 2048)` chunks.
- gamma loaded at `gamma_gm[d * TILE_D]` (gamma is 1-D, shared across rows).

## Tail handling

- Outer row loop: no tail — grid-stride absorbs non-multiple-of-72 S values (all S ≥ 231 > 72, so full 72 cores always).
- Inner D loop: last chunk uses `real_shape=[n_d]` where `n_d = min(TILE_D, D - off_d)`.
  - copy_in with real_shape loads n_d elements; remaining tile positions are padded.
  - **For reduction pass**: padding elements in x_sq must be 0. Assumption: copy_in zero-pads beyond real_shape (standard DMA behavior). If not verified at build time, fall back to splitting the tail chunk into a separate code path with exact-sized tile.
  - For normalize pass: garbage in padding positions is harmless — copy_out with real_shape only writes valid elements.

## UB budget (TILE_D=2048, f32 compute)

| Phase | Visible tiles | Bytes (naive) | ×1.6 factor |
|-------|--------------|---------------|-------------|
| Pass 1 (per chunk) | x(f16=2B) + xf(4B) + sq(4B) = 10B/elem | 20480 | ~32768 |
| Pass 2 (per chunk) | x(2) + xf(4) + g(2) + gf(4) + inv(4) + yf(4) = 20B/elem | 40960 | ~65536 |
| inv_rms broadcast tile | 4B/elem | 8192 | ~13107 |
| **Total per row (pass 2, worst)** | | | **~78643** |
| **With unroll_factor=2** | doubled | | **~157286** |

157286 << 253952 B budget. **No UB overflow risk.**

## Numerical risks and mitigations

| Risk | Case(s) | Mitigation |
|------|---------|------------|
| f16 overflow in x^2 (65504^2 = 4.3e9) | 19 | Always cast to f32 before squaring — f32 range is 3.4e38 |
| f32 overflow in sum(x^2) | 5, 11, 19 | Max: 8192 × 4.3e9 ≈ 3.5e13, well within f32 (3.4e38) |
| Near-zero input (all zeros) | 17 | eps=1e-4 → rsqrt(1e-4)=100, y=0×gamma×100=0. Correct. |
| Tiny epsilon (1e-12) | 9 | eps > 0 always. rsqrt(tiny + 1e-12) is fine in f32. |
| rsqrt precision | all | `asc2.rsqrt` is hardware-accelerated; relative error well within f16 threshold (2^-10) |
| f16/bf16 accumulation | 1,4,7,10,13,16,19 | Always accumulate sum-of-squares in f32 (required by spec: golden uses f32 internal compute) |
| Catastrophic cancellation | none | RmsNorm is sum-of-squares (all positive), no subtraction |

**Precision thresholds**: f16 → MERE < 2^-10 ≈ 9.8e-4; f32 → MERE < 2^-13 ≈ 1.2e-4; bf16 → MERE < 2^-7 ≈ 7.8e-3. f32 internal compute with hardware rsqrt comfortably meets all thresholds.

## Anti-cheat constraints

- Math: ALL in `@asc2.jit` kernel. No torch math, no `torch.nn.functional.rms_norm`.
- torch: only `torch.empty_like`, `.contiguous()`, `.shape`, `.numel()`, `.dtype`, `.device`, `.is_contiguous()`.
- `ensure_npu_platform()` called first.
- Output: contiguous NPU tensor, same shape/dtype as golden. Not a view of input.
- No `torch.cat`, `torch.clone`, `torch.sum`, tensor arithmetic, `.to(dtype)` on device data.

## Kernel signature

```python
@asc2.jit
def _rms_norm_kernel(
    x_ptr: asc.GlobalAddress,
    gamma_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    S: int, D: int, num_d_tiles: int,
    epsilon: float,
    tile_d: asc.ConstExpr[int]
):
```

Host flattens `x.shape[:-1]` into S, passes S, D, num_d_tiles, epsilon, and ConstExpr TILE_D=2048.
Launch: `_rms_norm_kernel[72](x, gamma, out, S, D, num_d_tiles, epsilon, 2048)`.

## Host dispatch

```python
def rms_norm(x, gamma, epsilon=1e-6):
    ensure_npu_platform()
    if not x.is_contiguous(): x = x.contiguous()
    if not gamma.is_contiguous(): gamma = gamma.contiguous()
    out = torch.empty_like(x)
    D = x.shape[-1]
    S = x.numel() // D
    TILE_D = 2048
    num_d_tiles = (D + TILE_D - 1) // TILE_D
    _rms_norm_kernel[72](x, gamma, out, S, D, num_d_tiles, epsilon, TILE_D)
    return out
```

## Local validation ladder

1. `python3 -m py_compile candidate.py` — syntax check.
2. Exact-v2 compile gate: all 20 cases route through pinned pyasc v2.
3. **Do not claim correctness** from local compile alone.
4. Numerical evidence requires camodel execution (not available locally).
5. Acceptance = CANNBench on real NPU.

Evidence label for this deliverable: `verified-local-compile` (pending).

DESIGN_DONE
