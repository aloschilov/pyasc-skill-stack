# Design: ForeachAddcdivScalar

## Algorithm

Fused elementwise composite over tensor lists:
```
y_i[j] = x1_i[j] + (x2_i[j] / x3_i[j]) * scalar
```
where `i` indexes the list (1..L), `j` indexes elements within tensor `i`.

**Host dispatch**: Loop over list length `L`, launch one kernel per (x1_i, x2_i, x3_i) triple. Allocate output per triple with `torch.empty_like(x1_i)`.

**Kernel**: Single `@asc2.jit` function, grid-stride tiling over flattened 1-D element count.

## Pinned-v2 APIs

- `asc2.global_tensor(ptr, [size])` — 1-D global memory view
- `asc2.copy_in(gm, [offset], [tile_size], real_shape=[n])` — load with tail support
- `asc2.copy_out(tile, gm, [offset], real_shape=[n])` — store with tail support
- `asc2.range(start, stop, step, unroll_factor=2)` — grid-stride loop
- `asc2.block_idx()`, `asc2.block_num()` — core indexing
- `tile.to(dtype)` — dtype cast
- Tile arithmetic: `+ - * /` (tile-tile, tile-scalar)
- NO `asc2.parallel`, NO `gm_barrier` on `asc2.range`

## All 20 Cases Coverage

| # | List len | Shape per tensor | DType | Scalar | Special |
|---|----------|------------------|-------|--------|---------|
| 1 | 2 | 2D (1024x1024) | f32 | 1.0 | — |
| 2 | 3 | 2D (2048x2048) | f16 | 1.0 | — |
| 3 | 1 | 2D (4096x4096) | bf16 | 1.0 | — |
| 4 | 1 | 2D (8192x8192) | f32 | 0.5 | — |
| 5 | 2 | 2D (8192x4096) | f16 | 2.0 | — |
| 6 | 1 | 2D (1023x1023) | bf16 | -1.0 | prime dims |
| 7 | 1 | 2D (1009x1021) | f32 | 1.5 | prime dims |
| 8 | 1 | 2D (1537x769) | f16 | 1.0 | prime dims |
| 9 | 2 | 3D (363x367x373) | bf16 | 1.0 | 3D |
| 10 | 1 | 2D (2049x513) | f32 | 1.0 | large range |
| 11 | 2 | 4D (3x7x13x4001) | f16 | 1.0 | 4D |
| 12 | 1 | 1D (1000003) | f32 | inf | inf/nan values |
| 13 | 1 | 5D (11x13x17x67x67) | bf16 | nan | nan values |
| 14 | 1 | 5D (3x7x11x13x1013) | f32 | 1.0 | zeros, tiny denom |
| 15 | 2 | 2D (512x2049) | f32 | 1.0 | — |
| 16 | 4 | 2D (255x8193) | bf16 | 1.0 | — |
| 17 | 1 | 2D (4097x511) | f16 | 0.0 | scalar=0 |
| 18 | 2 | 3D (2x511x2049) | f32 | 2.0 | 3D |
| 19 | 1 | 3D (4x255x2049) | bf16 | -0.5 | 3D |
| 20 | 1 | 5D (2x3x17x1024x101) | f32 | 1.5 | 5D |

**Key observations**:
- List lengths: 1, 2, 3, 4
- Ranks: 1D to 5D
- Element counts: ~1M to ~67M
- DTypes: f32 (10 cases), f16 (5), bf16 (5)
- Scalars: 0.0, ±0.5, ±1.0, 1.5, 2.0, inf, nan
- Special values: inf/nan in cases 12-13, zeros in case 14, tiny denominators in cases 6, 8, 14

## Tiling Strategy

**Tile size**: 1024 elements (f32 = 4KB per tile, with unroll_factor=2 = 8KB per iteration).

**Rationale**:
- Op chain: load x1/x2/x3 (3 tiles), cast to f32 (3 tiles), div (1), mul scalar (1), add (1), cast back (1), store (1) = ~12 tile live values
- Naive UB: 12 * 4 * 1024 * 2 = 98304 bytes
- With 1.6x overhead: ~157KB < 253KB budget ✓
- TILE=2048 would be ~314KB (overflow) ✗

**Grid-stride**: Each core processes multiple tiles, striding by `block_num()`.

**Core count**: `cores = min(72, num_tiles)` to maximize utilization.

## Tail Handling

```python
n = tile_size if off + tile_size <= size else size - off
x1_tile = asc2.copy_in(x1_gm, [off], [tile_size], real_shape=[n])
```
No host-side padding. `real_shape` tells the hardware how many elements are valid in the last tile.

## UB Budget

- Available: 253952 bytes
- Tile size: 1024 elements
- Live tiles per iteration: ~12 (3 inputs f16/bf16, 3 cast f32, div, mul, add, cast back, output)
- Usage estimate: 12 * 4B * 1024 * 2 (unroll) * 1.6 (overhead) ≈ 157KB ✓
- Headroom: ~96KB for compiler temporaries

**Fallback**: If UB overflow at runtime, halve tile to 512.

## Numerical Risks

1. **Division by zero / tiny denominators** (cases 6, 8, 14): x3 values in [0.001, 0.1]. IEEE division produces large values; f32 compute avoids intermediate overflow.

2. **Inf/nan propagation** (cases 12-13): scalar=inf or nan, input values include inf/nan. IEEE arithmetic propagates correctly through `+`, `/`, `*`. NO special-casing in kernel.

3. **Precision for f16/bf16**: Golden promotes to f32 before compute. We do the same:
   ```python
   x1f = x1_tile.to(asc.float32)
   x2f = x2_tile.to(asc.float32)
   x3f = x3_tile.to(asc.float32)
   div_tile = x2f / x3f
   scaled = div_tile * scalar
   result_f32 = x1f + scaled
   result = result_f32.to(input_dtype)
   ```

4. **Scalar range**: -1024 to 1024, plus inf/nan. Passed as runtime `float` argument. IEEE handles all.

5. **Cancellation**: None expected. Addition is not subtractive; division/multiplication are stable.

## Anti-Cheat Constraints

✓ **Allowed**:
- `torch.empty_like(x1_i)` for output allocation
- `x1_i.is_contiguous()`, `x1_i.contiguous()` for metadata/contiguity
- `x1_i.numel()` for size
- `x1_i.shape`, `x1_i.dtype` for metadata

✗ **Forbidden**:
- Any `torch.*` math ops
- Tensor arithmetic outside kernel (`x1 + x2`)
- `.to(dtype)` on device tensors (cast inside kernel only)
- `torch.cat`, `torch.sum`, `torch.clone`
- Caching outputs by `data_ptr`

**All compute in kernel**: Division, multiplication, addition, dtype casts happen inside `@asc2.jit`.

## Local Validation Ladder

1. **Syntax check**: `python3 -m py_compile candidate.py`
2. **Static contract check**: Worker harness validates module shape, imports, callable signature
3. **Exact-v2 compile gate**: `python3 -m cannbench.worker.exact_v2_compile --op foreach_addcdiv_scalar`
   - Compiles all 20 cases through pinned pyasc v2
   - Catches UB overflow, unsupported syntax, API misuse
   - Evidence label: `verified-local-compile`
4. **Camodel execution** (if available): Numerical comparison against golden
   - Evidence label: `verified-camodel`
5. **CANNBench on real NPU**: Acceptance oracle
   - Evidence label: `verified-cannbench`

**This design phase stops at step 3**. Numerical correctness and performance require camodel or real hardware.

## Implementation Notes

- **Module-level imports**: `import torch`, `import asc`, `import asc2`, `from ._pyasc_runtime import ensure_npu_platform`
- **Kernel signature**: `_addcdiv_kernel(x1_ptr, x2_ptr, x3_ptr, out_ptr, size, num_tiles, scalar, tile_size: asc.ConstExpr[int])`
- **Host signature**: `foreach_addcdiv_scalar(x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalar: float) -> List[torch.Tensor]`
- **List loop**: `for i in range(len(x1)):` on host, one kernel launch per triple
- **No early exit**: Even if scalar=0, launch kernel (result = x1). IEEE handles all.

---

**DESIGN_DONE**
