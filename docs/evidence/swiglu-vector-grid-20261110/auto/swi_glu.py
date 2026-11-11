"""SwiGlu (SiLU-gated linear unit) on public pyasc v2 asctile APIs.

swi_glu(input, dim=-1) -> output = silu(x0) * x1 = (x0 * sigmoid(x0)) * x1,
where (x0, x1) = input.chunk(2, dim=dim).

Repair round 2. The previous attempt timed out during lowering. Its kernel
derived each tile's logical width at runtime (`valid = cols - c0` with an
`if valid > tile: valid = tile` reassignment) inside nested tile loops and fed
that runtime scalar into `real_shape`. This version keeps every logical bound
compile-time via host-classified full/tail launches:

- Memory is viewed as 2D global tensors [rows, 2*cols] / [rows, cols], where
  rows = prod(shape[:d]) and cols = (shape[d]//2) * prod(shape[d+1:]) for the
  normalized split dimension d. x0 is at column c, x1 at column cols + c.
- Every tile is a physically aligned padded block whose width is a multiple
  of 16 elements, so the final physical dimension is 32B/64B aligned for
  f16/bf16/f32, with explicit real_shape for logical bounds and pad_value=0.0
  (neutral: silu(0)*0 = 0, no Inf/NaN from padded lanes). Row-block routes
  use [16, ct] tiles with ct in {16, 32, 64}; the wide-row route uses
  [rows, w_ct] tiles shaped for the live rows, at most 1024 elements per
  tile, with w_ct capped so eight blocks still receive full column tiles.
- Full row blocks and the row tail are separate host launches; full column
  tiles and the column tail are separate in-kernel blocks. All valid widths
  (rt, ct, r_valid, c_tail) are ConstExpr. The tail tile starts at physical
  column offset n_c_full*ct (after the completed full tiles, never again at
  column zero); the row-tail block starts at (rows//16)*16.
- Sigmoid is computed stepwise in f32 as 1/(1+exp(-x)) so IEEE specials match
  the golden's formula evaluation: silu(-Inf) = -Inf*0 = NaN, silu(+Inf)=+Inf,
  NaN propagates, 0*Inf = NaN.
- f16/bf16 inputs are cast once to f32 on load; all arithmetic stays in f32;
  the result is cast back once on store.

Torch is used only for allocation and metadata. Tensors are passed directly
to public JIT launches (never data_ptr()). Cores are capped at 8, bounded by
useful blocks. When all rows fit in a single 16-row tile and there are at
least eight full column tiles, distribute those columns across the same cap,
tiling for the live rows: height rows (no padded rows) and the widest
16-aligned width that keeps the tile at or below 1024 elements while still
leaving eight full column tiles, so every launched block stays useful.
"""

import torch
from grid_template import vector_blocks

try:
    from asc.experimental import asctile
except ImportError:
    import asctile

from ._pyasc_runtime import ensure_npu_platform

_RT = 16
_MAX_CORES = 8
_MAX_TILE_ELEMS = 1024


@asctile.jit
def _swiglu_tile(in_ptr: asctile.GlobalAddress,
                 out_ptr: asctile.GlobalAddress,
                 rows: int,
                 cols: int,
                 r0: int,
                 c0: int,
                 rt: asctile.ConstExpr[int],
                 ct: asctile.ConstExpr[int],
                 r_valid: asctile.ConstExpr[int],
                 c_valid: asctile.ConstExpr[int],
                 out_mode: asctile.ConstExpr[int]):
    in_gm = asctile.global_tensor(in_ptr, [rows, 2 * cols])
    out_gm = asctile.global_tensor(out_ptr, [rows, cols])
    x0 = asctile.copy_in(in_gm, [r0, c0], [rt, ct],
                         real_shape=[r_valid, c_valid], pad_value=0.0)
    x1 = asctile.copy_in(in_gm, [r0, cols + c0], [rt, ct],
                         real_shape=[r_valid, c_valid], pad_value=0.0)
    x0f = asctile.cast(x0, asctile.float32)
    x1f = asctile.cast(x1, asctile.float32)
    e = asctile.exp(asctile.mul(x0f, -1.0))
    s = asctile.div(1.0, asctile.add(e, 1.0))
    res = asctile.mul(asctile.mul(x0f, s), x1f)
    if out_mode == 1:
        out_t = asctile.cast(res, asctile.float16)
    elif out_mode == 2:
        out_t = asctile.cast(res, asctile.bfloat16)
    else:
        out_t = res
    asctile.copy_out(out_t, out_gm, [r0, c0], real_shape=[r_valid, c_valid])


@asctile.jit
def _swiglu_impl(in_ptr: asctile.GlobalAddress,
                 out_ptr: asctile.GlobalAddress,
                 rows: int,
                 cols: int,
                 r_base: int,
                 n_rb: int,
                 n_c_full: int,
                 rt: asctile.ConstExpr[int],
                 ct: asctile.ConstExpr[int],
                 r_valid: asctile.ConstExpr[int],
                 c_tail: asctile.ConstExpr[int],
                 out_mode: asctile.ConstExpr[int]):
    for rb in asctile.range(asctile.block_idx(), n_rb, asctile.block_num()):
        r0 = r_base + rb * rt
        for c in asctile.range(0, n_c_full):
            _swiglu_tile(in_ptr, out_ptr, rows, cols, r0, c * ct,
                         rt, ct, r_valid, ct, out_mode)
        if c_tail > 0:
            _swiglu_tile(in_ptr, out_ptr, rows, cols, r0, n_c_full * ct,
                         rt, ct, r_valid, c_tail, out_mode)


@asctile.jit
def _swiglu_wide_row(in_ptr: asctile.GlobalAddress,
                     out_ptr: asctile.GlobalAddress,
                     rows: int,
                     cols: int,
                     n_c_full: int,
                     rt: asctile.ConstExpr[int],
                     ct: asctile.ConstExpr[int],
                     r_valid: asctile.ConstExpr[int],
                     c_tail: asctile.ConstExpr[int],
                     out_mode: asctile.ConstExpr[int]):
    for c in asctile.range(asctile.block_idx(), n_c_full, asctile.block_num()):
        _swiglu_tile(in_ptr, out_ptr, rows, cols, 0, c * ct,
                     rt, ct, r_valid, ct, out_mode)
    if c_tail > 0:
        if asctile.block_idx() == n_c_full % asctile.block_num():
            _swiglu_tile(in_ptr, out_ptr, rows, cols, 0, n_c_full * ct,
                         rt, ct, r_valid, c_tail, out_mode)


def _col_tile(cols):
    if cols <= 16:
        return 16
    if cols <= 32:
        return 32
    return 64


def _wide_col_tile(rows, cols):
    ct = _MAX_TILE_ELEMS // rows
    ct -= ct % 16
    cap = cols // _MAX_CORES
    cap -= cap % 16
    if cap < ct:
        ct = cap
    if ct < 16:
        ct = 16
    return ct


def swi_glu(input, dim=-1):
    ensure_npu_platform()
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    dtype = input.dtype
    if dtype == torch.float32:
        out_mode = 0
    elif dtype == torch.float16:
        out_mode = 1
    elif dtype == torch.bfloat16:
        out_mode = 2
    else:
        raise TypeError("swi_glu supports float16, float32 and bfloat16 inputs")
    rank = input.dim()
    if rank == 0:
        raise ValueError("input must have at least one dimension")
    d = int(dim)
    if d < 0:
        d += rank
    if d < 0 or d >= rank:
        raise ValueError("dim is out of range for the input rank")
    shape = list(input.shape)
    size = shape[d]
    if size % 2 != 0:
        raise ValueError("size of the split dimension must be even")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    half = size // 2
    rows = 1
    for i in range(d):
        rows *= shape[i]
    inner = 1
    for i in range(d + 1, rank):
        inner *= shape[i]
    cols = half * inner
    out_shape = shape[:d] + [half] + shape[d + 1:]
    out = torch.empty(out_shape, dtype=dtype, device=input.device)
    if rows == 0 or cols == 0:
        return out
    ct = _col_tile(cols)
    n_c_full = cols // ct
    c_tail = cols % ct
    n_rb_full = rows // _RT
    r_tail = rows % _RT
    if rows <= _RT and n_c_full >= _MAX_CORES:
        w_ct = _wide_col_tile(rows, cols)
        n_w_full = cols // w_ct
        w_tail = cols % w_ct
        _swiglu_wide_row[vector_blocks(n_w_full + bool(w_tail))](input, out, rows, cols, n_w_full,
                                     rows, w_ct, rows, w_tail, out_mode)
        return out
    if n_rb_full > 0:
        cores = vector_blocks(n_rb_full)
        _swiglu_impl[cores](input, out, rows, cols, 0, n_rb_full, n_c_full,
                            _RT, ct, _RT, c_tail, out_mode)
    if r_tail > 0:
        _swiglu_impl[1](input, out, rows, cols, n_rb_full * _RT, 1, n_c_full,
                        _RT, ct, r_tail, c_tail, out_mode)
    return out
