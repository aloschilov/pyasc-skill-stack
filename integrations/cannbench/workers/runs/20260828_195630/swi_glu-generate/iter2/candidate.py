"""CANN Bench SwiGlu interface implemented as a pyasc asc2 kernel.

Kernel design (pyasc-api-patterns grid-stride variant, 2-D tiled):
  - The contiguous input is viewed as [outer, C*inner] where
    outer = prod(shape[:dim]), C = shape[dim], inner = prod(shape[dim+1:]).
    x0 occupies columns [0, half_cols), x1 columns [half_cols, 2*half_cols)
    of every row, with half_cols = (C // 2) * inner. Output is [outer, half_cols]
    which exactly matches the golden's contiguous output memory layout, so no
    on-device data copies are needed to materialize x0 / x1.
  - Work is distributed over (row, column-chunk) tiles ([1, TILE]) via a
    grid-stride loop indexed ``t`` decomposed as ``row = t // num_col_tiles`` /
    ``col_tile = t % num_col_tiles`` so small-outer shapes still saturate cores.
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision (the
    spec's precision standard expects f32 internal compute).
  - silu(v) = v * sigmoid(v). sigmoid uses the cancellation-free stable form
    ``sig = exp(min(s,0)) / (1 + exp(-|s|))``; exp(min(s,0)) and exp(-|s|) both
    have non-positive arguments so neither can overflow. To match the golden
    reference's ``1 / (1 + exp(-s))`` bit-for-bit in the saturated tail
    (where ``exp(-s)`` overflows f32 for s < -log(F32_MAX)), sigmoid is clamped
    to 0 there with an ``asc2.where`` blend.
  - UB budget: the ~12 f32 tile temporaries cost ~12 * 4 * 1024 * 2(unroll)
    * 1.6(measured) ~ 188 KB, under the ~253952 byte static limit; TILE=1024
    with unroll_factor=2 is the safe choice.
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_TILE = 1024
_MAX_CORES = 72
# f32 exp() saturates to +inf for args > log(F32_MAX) ~ 88.7228.
_EXP_SAT_THRESH = -88.7228


@asc2.jit
def _swiglu_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                   outer: int, half_cols: int, total_cols: int,
                   num_col_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [outer, total_cols])
    out_gm = asc2.global_tensor(out_ptr, [outer, half_cols])
    for t in asc2.range(asc2.block_idx(), outer * num_col_tiles,
                        asc2.block_num(), unroll_factor=2):
        row = t // num_col_tiles
        col_tile = t % num_col_tiles
        col_off = col_tile * tile_size
        n = tile_size if col_off + tile_size <= half_cols else half_cols - col_off
        x0 = asc2.copy_in(x_gm, [row, col_off], [1, tile_size],
                          real_shape=[1, n])
        x1 = asc2.copy_in(x_gm, [row, col_off + half_cols], [1, tile_size],
                          real_shape=[1, n])
        s = x0.to(asc.float32)
        x1f = x1.to(asc.float32)
        abs_s = asc2.abs(s)
        neg_abs = -abs_s
        exp_neg_abs = asc2.exp(neg_abs)
        denom = exp_neg_abs + 1.0
        min_s = (s - abs_s) * 0.5
        numer = asc2.exp(min_s)
        sig = numer / denom
        zero_tile = s * 0.0
        cond = s >= -88.7228
        sig_clamped = asc2.where(cond, sig, zero_tile)
        silu_v = s * sig_clamped
        out = silu_v * x1f
        asc2.copy_out(out.to(x0.dtype), out_gm, [row, col_off],
                      real_shape=[1, n])


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """SwiGLU activation of an NPU tensor via a pyasc asc2 kernel.

    output = silu(x0) * x1 where (x0, x1) = chunk(input, 2, dim) and
    silu(v) = v * sigmoid(v).
    """
    ensure_npu_platform()
    if not input.is_contiguous():
        input = input.contiguous()
    shape = tuple(input.shape)
    rank = len(shape)
    total = input.numel()
    if rank == 0 or total == 0:
        return torch.empty(shape, dtype=input.dtype, device=input.device)
    d = dim if dim >= 0 else dim + rank
    if d < 0 or d >= rank:
        raise ValueError(
            "SwiGlu: dim {} out of range for rank {}".format(dim, rank))
    c_dim = shape[d]
    if c_dim % 2 != 0:
        raise ValueError(
            "SwiGlu: size along dim {} must be even, got {}".format(dim, c_dim))
    outer = 1
    for i in range(d):
        outer *= shape[i]
    inner = 1
    for i in range(d + 1, rank):
        inner *= shape[i]
    half_cols = (c_dim // 2) * inner
    total_cols = c_dim * inner
    out_shape = list(shape)
    out_shape[d] = c_dim // 2
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    tile_size = _TILE
    num_col_tiles = asc.ceildiv(half_cols, tile_size)
    total_tiles = outer * num_col_tiles
    cores = min(_MAX_CORES, total_tiles)
    _swiglu_kernel[cores](input, out, outer, half_cols, total_cols,
                          num_col_tiles, tile_size)
    return out
