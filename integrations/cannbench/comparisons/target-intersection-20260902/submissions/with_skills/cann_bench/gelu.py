"""CANN Bench Gelu interface implemented as a pyasc asc2 kernel.

Kernel design (grid-stride variant):
  - 1-D flatten; each block strides over tiles so any element count / rank works.
  - Tail tiles handled with ``real_shape`` loads/stores (no host padding).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision (the
    spec's precision standard expects f32 internal compute), then cast back on
    store.
  - Host dispatches on the ``approximate`` string (unsupported as a kernel
    param) to one of two specialized ``@asc2.jit`` kernels:
      * erf mode  (approximate == "none"):
            y = (x * 0.5) * (erf(x / sqrt(2)) + 1.0)
      * tanh mode (approximate == "tanh"):
            y = (x * 0.5) * (tanh(sqrt(2/pi) * (x + 0.044715 * x^3)) + 1.0)
  - All constants are module-level floats (no math.* inside jit); scalars stay
    on the RIGHT of tile arithmetic (Tile has no __rmul__).
  - Tiles: erf chain (V~6) uses 2048 wide / 1024 narrow; tanh chain (V~10) uses
    1024 (UB-budget safe under 253952 B with unroll_factor=2).
  - erf/tanh are bounded ops, so no overflow guard is needed; IEEE special
    values (Inf/NaN) propagate through the vector ops and match the golden's
    identical-form evaluation.
"""

import math

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_C1_ERF = 1.0 / math.sqrt(2.0)
_C0_TANH = math.sqrt(2.0 / math.pi)
_K_TANH = 0.044715

_ERF_TILE = 2048
_NARROW_TILE = 1024
_TANH_TILE = 1024
_MAX_CORES = 72
_ERF_THRESHOLD = _MAX_CORES * _ERF_TILE


@asc2.jit
def _gelu_erf_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                     size: int, num_tiles: int,
                     tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        s = xf * _C1_ERF
        e = asc2.erf(s)
        one = e + 1.0
        hx = xf * 0.5
        yf = hx * one
        asc2.copy_out(yf.to(x.dtype), out_gm, [off], real_shape=[n])


@asc2.jit
def _gelu_tanh_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      size: int, num_tiles: int,
                      tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        x2 = xf * xf
        x3 = x2 * xf
        a = x3 * _K_TANH
        b = xf + a
        c = b * _C0_TANH
        th = asc2.tanh(c)
        one = th + 1.0
        hx = xf * 0.5
        yf = hx * one
        asc2.copy_out(yf.to(x.dtype), out_gm, [off], real_shape=[n])


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Element-wise GELU of an NPU tensor via a pyasc asc2 kernel."""
    ensure_npu_platform()
    assert approximate in ("none", "tanh"), (
        f"approximate must be 'none' or 'tanh', got {approximate!r}")
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    if approximate == "none":
        if size >= _ERF_THRESHOLD:
            tile = _ERF_TILE
        else:
            tile = _NARROW_TILE
        num_tiles = asc.ceildiv(size, tile)
        cores = min(_MAX_CORES, num_tiles)
        _gelu_erf_kernel[cores](x, out, size, num_tiles, tile)
    else:
        tile = _TANH_TILE
        num_tiles = asc.ceildiv(size, tile)
        cores = min(_MAX_CORES, num_tiles)
        _gelu_tanh_kernel[cores](x, out, size, num_tiles, tile)
    return out
