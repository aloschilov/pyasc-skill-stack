# GELU Operator Design — pyasc asc2 JIT Kernel

## Overview

Implement `gelu(x, approximate="none")` as a pyasc asc2 kernel module covering all 20 evaluation cases. Two separate `@asc2.jit` kernels — one per mode — following the proven grid-stride tile-loop pattern from the sigmoid reference.

## Mode Decomposition

| Mode | Trigger | Math | Kernel |
|------|---------|------|--------|
| exact | `approximate="none"` | `y = x * 0.5 * (1 + erf(x / sqrt(2)))` | `_gelu_exact_kernel` |
| tanh | `approximate="tanh"` | `y = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))` | `_gelu_tanh_kernel` |

## Tile Sizes & UB Budget

| Mode | Wide Tile | Narrow Tile | Rationale |
|------|-----------|-------------|-----------|
| exact | 2048 | 1024 | ~5 live f32 tiles at peak; 5×4×2048×2×1.6 ≈ 131 KB < 254 KB |
| tanh | 1024 | 512 | ~7 live f32 tiles at peak (incl. fix-up); 7×4×1024×2×1.6 ≈ 90 KB < 254 KB |

Tile selection: WIDE when `numel >= 72 * WIDE_TILE`, else NARROW. All 20 cases have ≥1M elements, so WIDE will always be selected, but NARROW is included as safety.

## Case Coverage (all 20)

| Case | Shape | dtype | Range | Mode | Elements | Notes |
|------|-------|-------|-------|------|----------|-------|
| 1 | 1024×1024 | f16 | [-1,1] | exact | 1M | Standard |
| 2 | 2048×2048 | f32 | [-2,2] | exact | 4M | Standard |
| 3 | 4096×4096 | bf16 | [-3,3] | exact | 16M | Standard |
| 4 | 8192×8192 | f16 | [-10,10] | tanh | 64M | Large |
| 5 | 8192×8192 | f32 | [-100,100] | tanh | 64M | Large; tanh saturates |
| 6 | 1023×1023 | bf16 | [-0.1,0.1] | tanh | ~1M | Small range |
| 7 | 1009×1021 | f16 | [-1,2] | exact | ~1M | Non-power-of-2 |
| 8 | 1537×769 | f32 | [-5,10] | tanh | ~1.2M | Non-square |
| 9 | 363×367×373 | bf16 | [-50,100] | exact | ~50M | 3D |
| 10 | 2049×513 | f16 | [-65504,65504] | tanh | ~1M | Full f16 range; x² in f32 prevents overflow |
| 11 | 3×7×13×4001 | f32 | [-88,88] | exact | ~1.1M | 4D; erf(62)→1.0, no overflow |
| 12 | 1000003 | bf16 | [-inf,inf] | tanh | 1M | Needs -inf fix-up |
| 13 | 11×13×17×67×67 | f32 | [nan,nan] | exact | ~10.7M | NaN propagates naturally |
| 14 | 3×7×11×13×1009 | f16 | [0,0] | tanh | ~3.2M | All zeros; gelu(0)=0 |
| 15 | 512×2049 | f32 | [-0.5,0.5] | exact | ~1M | Standard |
| 16 | 255×8193 | bf16 | [-1,3] | exact | ~2.1M | Standard |
| 17 | 4097×511 | f16 | [-1000,1000] | tanh | ~2.1M | tanh saturates |
| 18 | 2×511×2049 | f32 | [-0.2,0.2] | exact | ~2.1M | 3D |
| 19 | 4×255×2049 | bf16 | [-3,6] | tanh | ~2.1M | 3D |
| 20 | 2×3×17×1024×101 | f32 | [-20,40] | exact | ~10.5M | 5D |

## Numerical Stability

### Exact mode: `y = x * 0.5 * (1 + erf(x / sqrt(2)))`

- No overflow risk: erf clamps to [-1, 1], so cdf = 0.5*(1+erf) ∈ [0, 1]. Multiplying by x yields a value with |y| ≤ |x|.
- Case 11 (x ∈ [-88,88], f32): erf(88/√2) = erf(62.2) → 1.0 in f32, so y = 88*1.0 = 88. For x=-88: erf(-62.2) → -1.0, cdf → 0.0, y = -88 * 0.0 = -0.0. No NaN since -88 is finite.
- Case 13 (NaN input): NaN/sqrt(2) = NaN, erf(NaN) = NaN, result = NaN. Matches golden.

### Tanh mode: `y = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x²)))`

- Rewritten as `x * (1 + 0.044715 * x²)` to avoid a separate x³ term (saves one f32 tile).
- Case 10 (f16, [-65504, 65504]): x² = 4.29e9 in f32 (fits f32). 0.044715 * 4.29e9 = 1.92e8. x * (1 + 1.92e8) ≈ 1.26e13. sqrt(2/pi) * 1.26e13 = 1.0e13. tanh(1e13) = 1.0. y = 0.5 * 65504 * 2 = 65504. Correct.
- Case 5 (f32, [-100,100]): x² = 10000. inner = 100*(1 + 0.044715*10000) = 100*448.15 = 44815. tanh(0.798*44815) = tanh(35740) = 1.0. Correct saturation.
- **Case 12 (-inf fix)**: For x = -inf: x² = +inf, inner = -inf * inf = -inf, tanh(-inf) = -1, 1+(-1) = 0, y = -inf * 0.5 * 0 = NaN. **Bug.** Fix: `asc2.where(xf < -3.4e38, 0.0, result)`. Only -inf satisfies `f32 < -3.4e38` (since max finite f32 magnitude is ~3.4e38). Correct because gelu(-inf) = 0.
- For x = +inf: x² = +inf, inner = +inf, tanh(+inf) = 1, y = inf * 0.5 * 2 = inf. Correct, no fix needed.
- Case 14 (all zeros): tanh(0) = 0, y = 0 * 0.5 * 1 = 0. Correct.

## Kernel Pseudocode

### Exact Kernel

```
@asc2.jit
_gelu_exact_kernel(x_ptr, out_ptr, size, num_tiles, tile_size: ConstExpr):
    x_gm = global_tensor(x_ptr, [size])
    out_gm = global_tensor(out_ptr, [size])
    for t in range(block_idx(), num_tiles, block_num(), unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(float32)
        z = xf * 0.7071067811865476          # x / sqrt(2)
        erf_z = asc2.erf(z)
        cdf = (erf_z + 1.0) * 0.5
        result = xf * cdf
        copy_out(result.to(x.dtype), out_gm, [off], real_shape=[n])
```

Peak live f32 tiles: xf, z, erf_z, cdf, result = 5. UB ≈ 131 KB at TILE=2048.

### Tanh Kernel

```
@asc2.jit
_gelu_tanh_kernel(x_ptr, out_ptr, size, num_tiles, tile_size: ConstExpr):
    x_gm = global_tensor(x_ptr, [size])
    out_gm = global_tensor(out_ptr, [size])
    for t in range(block_idx(), num_tiles, block_num(), unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(float32)
        x_sq = xf * xf
        inner = xf * (1.0 + 0.044715 * x_sq)     # x + 0.044715*x^3
        tanh_arg = 0.7978845608028654 * inner     # sqrt(2/pi) * inner
        tanh_val = asc2.tanh(tanh_arg)
        result = xf * 0.5 * (1.0 + tanh_val)
        # Fix -inf: gelu(-inf) = 0
        neg_large = asc2.full([tile_size], -3.4e38, dtype=asc.float32)
        is_neg_inf = asc2.less(xf, neg_large)
        zero_tile = asc2.full([tile_size], 0.0, dtype=asc.float32)
        result = asc2.where(is_neg_inf, zero_tile, result)
        copy_out(result.to(x.dtype), out_gm, [off], real_shape=[n])
```

Peak live f32 tiles during main compute: xf, x_sq, inner, tanh_arg, tanh_val, result = 6. During fix-up: xf, result, neg_large, is_neg_inf, zero_tile, new_result = 6. UB ≈ 90 KB at TILE=1024.

## Host Function

```python
def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    ensure_npu_platform()
    x = x.contiguous() if not x.is_contiguous() else x
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    if approximate == "none":
        tile = _EXACT_WIDE if size >= _MAX_CORES * _EXACT_WIDE else _EXACT_NARROW
    else:
        tile = _TANH_WIDE if size >= _MAX_CORES * _TANH_WIDE else _TANH_NARROW
    num_tiles = asc.ceildiv(size, tile)
    cores = min(_MAX_CORES, num_tiles)
    if approximate == "none":
        _gelu_exact_kernel[cores](x, out, size, num_tiles, tile)
    else:
        _gelu_tanh_kernel[cores](x, out, size, num_tiles, tile)
    return out
```

## Constants

| Name | Value | Where Used |
|------|-------|------------|
| `_EXACT_WIDE` | 2048 | Host |
| `_EXACT_NARROW` | 1024 | Host |
| `_TANH_WIDE` | 1024 | Host |
| `_TANH_NARROW` | 512 | Host |
| `_MAX_CORES` | 72 | Host |
| `1/√2` | 0.7071067811865476 | Exact kernel (literal) |
| `√(2/π)` | 0.7978845608028654 | Tanh kernel (literal) |
| cubic coeff | 0.044715 | Tanh kernel (literal) |
| -3.4e38 | -3.4e38 | Tanh kernel (full tile) |

## Module Structure

```
candidate.py
├── imports: torch, asc, asc2, _pyasc_runtime
├── module-level tile/core constants
├── @asc2.jit _gelu_exact_kernel
├── @asc2.jit _gelu_tanh_kernel
└── def gelu(x, approximate="none") -> Tensor
```

## Accuracy Thresholds

| dtype | MERE threshold | MARE threshold | Our expected error |
|-------|---------------|----------------|-------------------|
| float16 | 2^-10 ≈ 9.8e-4 | 2^-7 ≈ 7.8e-3 | < 1e-4 (f32 compute) |
| float32 | 2^-13 ≈ 1.2e-4 | 2^-10 ≈ 9.8e-4 | < 1e-5 (f32 native) |
| bfloat16 | 2^-7 ≈ 7.8e-3 | 2^-4 ≈ 6.3e-2 | < 1e-3 (f32 compute) |

All cases use f32 internal compute (promote from f16/bf16, demote on output). This gives f32-precision results that comfortably pass all thresholds.

## Risk Mitigations

1. **UB overflow**: Tanh kernel uses TILE=1024 (conservative). If overflow occurs at runtime, halve to 512.
2. **-inf in tanh mode**: Explicit `asc2.where` fix-up with `asc2.less` comparison.
3. **NaN propagation**: Natural through IEEE arithmetic; no special handling needed.
4. **Non-power-of-2 shapes**: Grid-stride loop with `real_shape=[n]` handles all tails.
5. **Anti-cheat compliance**: All compute in `@asc2.jit`; torch used only for allocation/metadata.
