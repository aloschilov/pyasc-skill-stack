"""CANN Bench Gelu interface implemented as pyasc asc2 kernels.

Two modes per the spec (proto.yaml attr ``approximate``):
  - "none": y = x * 0.5 * (1 + erf(x / sqrt(2)))
  - "tanh": y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))

Both naive forms cancel catastrophically for x << 0: ``1 + erf(v)`` and
``1 + tanh(u)`` round to 0 in f32 once x < ~-5.5, while the true output
is small but nonzero, failing the harness f32 relative-error check on
wide value ranges (e.g. [-88, 88]).

Stable reformulations used here:
  - erf mode: 1 + erf(v) = 2 - erfc(|v|) for v >= 0 and erfc(|v|) for
    v < 0, with erfc computed by the Numerical Recipes rational-
    Chebyshev fit  erfc(z) = t * exp(-z^2 + P(t)), t = 1/(1 + z/2)
    (fractional error < 1.2e-7 for all z >= 0). No cancellation in
    either branch.  Output is computed as
      y = where(x >= 0, x - x*0.5*erfc(z), x*0.5*erfc(z))
    which is cancellation-free (the subtraction x - x*0.5*erfc is
    safe because 0.5*erfc in [0, 0.5] so the result is in [0.5x, x]).
  - tanh mode: 1 + tanh(u) = 2*sigmoid(2u), and sigmoid(s) is computed
    in the branch-free stable form e^min(s,0) / (1 + e^-|s|), so e^()
    never sees a positive argument.

Performance optimisations:
  - Two compiled tile variants per mode (wide / safe).  The host tries
    the wide variant first and falls back to the safe one on UB
    overflow, caching the working tile size for subsequent calls.
  - erf wide variant uses unroll_factor=1 (halves UB allocation,
    enabling TILE=1024); safe variant keeps unroll_factor=2 at TILE=512.
  - erf kernel eliminates the intermediate v = x/sqrt(2) (z = |x|/sqrt(2)
    computed directly) and uses the algebraic output form above
    (saves 2 tile temporaries vs the original 2-erfc branch).
  - tanh kernel factors the cubic polynomial: s = (x^2 * (_GELU_C *
    _TWO_SQRT_2_OVER_PI) + _TWO_SQRT_2_OVER_PI) * x, saving one
    multiply per element.

One or more @asc2.jit kernels per mode (the mode is a compile-time
choice, not a runtime value).  Kernel design follows pyasc-api-patterns
Pattern A (1-D flatten, grid-stride tile loop, real_shape tails, f32
internal compute).  Tiles are always the LEFT operand of arithmetic
(asc2 Tile has no __rmul__).
"""

import math

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

# ---------------------------------------------------------------------------
# Module-level precomputed constants (no math.* inside @asc2.jit).
# ---------------------------------------------------------------------------
_MAX_CORES = 72
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_TWO_SQRT_2_OVER_PI = 2.0 * math.sqrt(2.0 / math.pi)
_GELU_C = 0.044715
# Product _GELU_C * _TWO_SQRT_2_OVER_PI, precomputed so the tanh kernel
# can factor the cubic  x^3*c + x  into  (x^2 * gc_t + k) * x  saving
# one tile-scalar multiply per element.
_GC_T = _GELU_C * _TWO_SQRT_2_OVER_PI

# Tile sizes — wide variants are tried first and fall back to safe.
_TILE_ERF_SAFE = 512      # unroll_factor=2 — always fits
_TILE_ERF_WIDE = 1024     # unroll_factor=1 — fits when UB allows
_TILE_TANH_SAFE = 1024    # unroll_factor=2 — always fits
_TILE_TANH_WIDE = 1536    # unroll_factor=2 — fits when UB allows

# Cached tile-selection state (populated on first call per mode).
_erf_wide_ok = None       # None = untested, True/False = tested
_tanh_tile = None          # None = untested, int = chosen tile


# ---------------------------------------------------------------------------
# erf-mode kernels
# ---------------------------------------------------------------------------

@asc2.jit
def _gelu_erf_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                     size: int, num_tiles: int,
                     tile_size: asc.ConstExpr[int]):
    """erf-mode GELU, grid-stride tile loop with unroll_factor=2."""
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        z = asc2.abs(xf) * _INV_SQRT2
        den = z * 0.5 + 1.0
        tt = asc2.full([tile_size], 1.0, dtype=asc.float32) / den
        # Numerical Recipes erfc Chebyshev fit, Horner form in tt.
        p = tt * 0.17087277 - 0.82215223
        p = p * tt + 1.48851587
        p = p * tt - 1.13520398
        p = p * tt + 0.27886807
        p = p * tt - 0.18628806
        p = p * tt + 0.09678418
        p = p * tt + 0.37409196
        p = p * tt + 1.00002368
        p = p * tt - 1.26551223
        erfc_z = tt * asc2.exp(p - z * z)          # erfc(|v|), rel err ~1e-7
        # Algebraic output — cancellation-free:
        #   x >= 0: y = x - x * 0.5 * erfc  (safe: 0.5*erfc in [0,0.5])
        #   x <  0: y = x * 0.5 * erfc
        half_erfc = erfc_z * 0.5
        y_neg = xf * half_erfc
        y_pos = xf - y_neg
        y = asc2.where(xf >= 0.0, y_pos, y_neg)
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


@asc2.jit
def _gelu_erf_kernel_u1(x_ptr: asc.GlobalAddress,
                        out_ptr: asc.GlobalAddress,
                        size: int, num_tiles: int,
                        tile_size: asc.ConstExpr[int]):
    """erf-mode GELU, grid-stride tile loop with unroll_factor=1.

    Unroll 1 halves UB allocation vs unroll 2, enabling TILE=1024
    (each f32 tile temporary costs 4*TILE*1 bytes instead of 4*TILE*2).
    """
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=1):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        z = asc2.abs(xf) * _INV_SQRT2
        den = z * 0.5 + 1.0
        tt = asc2.full([tile_size], 1.0, dtype=asc.float32) / den
        p = tt * 0.17087277 - 0.82215223
        p = p * tt + 1.48851587
        p = p * tt - 1.13520398
        p = p * tt + 0.27886807
        p = p * tt - 0.18628806
        p = p * tt + 0.09678418
        p = p * tt + 0.37409196
        p = p * tt + 1.00002368
        p = p * tt - 1.26551223
        erfc_z = tt * asc2.exp(p - z * z)
        half_erfc = erfc_z * 0.5
        y_neg = xf * half_erfc
        y_pos = xf - y_neg
        y = asc2.where(xf >= 0.0, y_pos, y_neg)
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


# ---------------------------------------------------------------------------
# tanh-mode kernel
# ---------------------------------------------------------------------------

@asc2.jit
def _gelu_tanh_kernel(x_ptr: asc.GlobalAddress,
                     out_ptr: asc.GlobalAddress,
                     size: int, num_tiles: int,
                     tile_size: asc.ConstExpr[int]):
    """tanh-mode GELU with stable sigmoid, grid-stride tile loop."""
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        # Factored cubic: s = 2u = (x^2 * gc_t + k) * x
        #   where gc_t = _GELU_C * _TWO_SQRT_2_OVER_PI, k = _TWO_SQRT_2_OVER_PI
        x2 = xf * xf
        s = (x2 * _GC_T + _TWO_SQRT_2_OVER_PI) * xf
        # y = x * sigmoid(s), stable: e^min(s,0) / (1 + e^-|s|)
        sig = asc2.exp(asc2.minimum(s, 0.0)) / (asc2.exp(-asc2.abs(s)) + 1.0)
        y = xf * sig
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


# ---------------------------------------------------------------------------
# Public callable
# ---------------------------------------------------------------------------

def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Element-wise GELU of an NPU tensor via pyasc asc2 kernels."""
    ensure_npu_platform()
    if approximate not in ("none", "tanh"):
        raise ValueError(
            f"approximate must be 'none' or 'tanh', got {approximate!r}")
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out

    if approximate == "none":
        # ---------------------------------------------------------------
        # erf mode: try wide tile (TILE=1024, unroll=1) first, fall back
        # to safe tile (TILE=512, unroll=2) on UB overflow.
        # ---------------------------------------------------------------
        global _erf_wide_ok
        if _erf_wide_ok is None:
            tile = _TILE_ERF_WIDE
            num_t = asc.ceildiv(size, tile)
            cores = min(_MAX_CORES, num_t)
            try:
                _gelu_erf_kernel_u1[cores](x, out, size, num_t, tile)
                _erf_wide_ok = True
            except RuntimeError:
                _erf_wide_ok = False
                tile = _TILE_ERF_SAFE
                num_t = asc.ceildiv(size, tile)
                cores = min(_MAX_CORES, num_t)
                _gelu_erf_kernel[cores](x, out, size, num_t, tile)
        elif _erf_wide_ok:
            tile = _TILE_ERF_WIDE
            num_t = asc.ceildiv(size, tile)
            cores = min(_MAX_CORES, num_t)
            _gelu_erf_kernel_u1[cores](x, out, size, num_t, tile)
        else:
            tile = _TILE_ERF_SAFE
            num_t = asc.ceildiv(size, tile)
            cores = min(_MAX_CORES, num_t)
            _gelu_erf_kernel[cores](x, out, size, num_t, tile)
    else:
        # ---------------------------------------------------------------
        # tanh mode: try wide tile (TILE=1536, unroll=2) first, fall back
        # to safe tile (TILE=1024, unroll=2) on UB overflow.
        # ---------------------------------------------------------------
        global _tanh_tile
        if _tanh_tile is None:
            _tanh_tile = _TILE_TANH_WIDE
        tile = _tanh_tile
        num_t = asc.ceildiv(size, tile)
        cores = min(_MAX_CORES, num_t)
        try:
            _gelu_tanh_kernel[cores](x, out, size, num_t, tile)
        except RuntimeError:
            _tanh_tile = _TILE_TANH_SAFE
            tile = _tanh_tile
            num_t = asc.ceildiv(size, tile)
            cores = min(_MAX_CORES, num_t)
            _gelu_tanh_kernel[cores](x, out, size, num_t, tile)

    return out
