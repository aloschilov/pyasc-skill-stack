"""SwiGLU operator for CANNBench on the public archive-bundled AscTile runtime
(Ascend950PR_9599, local compiler/UB budget 253952 B).

    output = silu(x0) * x1 = (x0 * sigmoid(x0)) * x1

The single input is split into two equal halves along ``dim``.

Layout decomposition
--------------------
With ``outer = prod(shape[:dim])``, ``N = shape[dim]``, ``H = N // 2`` and
``inner = prod(shape[dim+1:])``, output element (o, h, i) reads gate
``x[o, h, i]`` and value ``x[o, H + h, i]``.  Flattening each o-block gives
three contiguous runs per block: gate at ``2*L*o``, value at ``2*L*o + L`` and
output at ``L*o`` with ``L = H * inner``.  The kernel therefore views the
input as a dense ``[outer, 2*L]`` row-major descriptor and the output as
``[outer, L]``: a strided 2-D tile at (row, col) loads the gate plane at
column ``col`` and the value plane at column ``L + col`` and stores the output
at column ``col``.  This is uniform over every split axis, rank, dtype and
tail; no host-side permutation or data copy is ever materialized.

Tile / unroll / grid selection (host, after ``ensure_npu_platform()``)
----------------------------------------------------------------------
* physical tile width is a multiple of the 32-byte DMA quantum
  (16 lanes for fp16/bf16, 8 for fp32), bounded by an 8192-lane column
  target and by the per-lane UB cost of this kernel family
  (fp16 22 B/lane, fp32 28 B/lane, bf16 32 B/lane, unroll 1);
* tile height is derived from the detected UB capacity, the row count and
  useful work per AIV, rounded down to a power of two;
* partial rows/columns are bounded with ``real_shape`` plus an
  operation-neutral zero pad (silu(0) = 0, 0 * pad = 0, no Inf/NaN);
* the grid is ``vector_blocks(total_tiles)``; unroll 2 is selected only when
  at least two tiles per block exist and the doubled footprint fits UB.

Numerics: gate and value are widened to fp32, ``sig = g / (1 + exp(-g))``
matches silu (including -inf -> nan, +inf -> inf, NaN propagation, and the
fp16 boundary where exp overflows to inf giving -0), then one cast to the
input dtype before the store.  The kernel uses ``@asctile.jit(reuse_alloc=1)``
(the recommended allocation pass); synchronization stays ON via defaults.
"""

import torch
from asc.experimental import asctile

try:
    from ._pyasc_runtime import (
        ensure_npu_platform,
        query_aivs,
        query_ub_bytes,
        vector_blocks,
    )
except ImportError:  # direct import when the module is on sys.path
    from _pyasc_runtime import (
        ensure_npu_platform,
        query_aivs,
        query_ub_bytes,
        vector_blocks,
    )

__all__ = ["swi_glu"]

_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_ITEMSIZE = {torch.float16: 2, torch.float32: 4, torch.bfloat16: 2}
# Conservative compiled live-buffer cost in bytes per physical lane per
# pipeline stage for this kernel family (inputs widened to fp32, fp32
# temporaries, one output cast), consistent with the measured bounds for the
# strided SwiGLU row kernel on the archive-bundled 950PR build.
_BYTES_PER_LANE = {torch.float16: 22, torch.float32: 28, torch.bfloat16: 32}
_DEFAULT_UB_BYTES = 253952
_DEFAULT_AIVS = 48
_COLUMN_TARGET = 8192
_MIN_TILE_ROWS = 4


@asctile.jit(reuse_alloc=1)
def _swiglu_tiles(
    x_ptr: asctile.GlobalAddress,
    y_ptr: asctile.GlobalAddress,
    outer: int,
    half_len: int,
    n_tiles: int,
    tile_rows: asctile.ConstExpr[int],
    tile_cols: asctile.ConstExpr[int],
    col_chunks: asctile.ConstExpr[int],
    unroll: asctile.ConstExpr[int],
):
    """Grid-stride 2-D tiled SwiGLU over the [outer, L] / [outer, 2L] views.

    Tile ordinal t maps to row block ``t // col_chunks`` and column chunk
    ``t % col_chunks``.  Every logical output element has exactly one writer;
    partial rows and columns are bounded with real_shape and zero padding.
    """
    x_gm = asctile.global_tensor(x_ptr, [outer, 2 * half_len])
    y_gm = asctile.global_tensor(y_ptr, [outer, half_len])
    for t in asctile.range(
        asctile.block_idx(), n_tiles, asctile.block_num(), unroll_factor=unroll
    ):
        row0 = (t // col_chunks) * tile_rows
        col0 = (t % col_chunks) * tile_cols
        valid_rows = min(outer - row0, tile_rows)
        valid_cols = min(half_len - col0, tile_cols)
        gate = asctile.copy_in(
            x_gm,
            [row0, col0],
            [tile_rows, tile_cols],
            real_shape=[valid_rows, valid_cols],
            pad_value=0.0,
        )
        value = asctile.copy_in(
            x_gm,
            [row0, col0 + half_len],
            [tile_rows, tile_cols],
            real_shape=[valid_rows, valid_cols],
            pad_value=0.0,
        )
        gate_f = gate.to(asctile.float32)
        value_f = value.to(asctile.float32)
        sig = gate_f / (1.0 + asctile.exp(-gate_f))
        out = (sig * value_f).to(gate.dtype)
        asctile.copy_out(
            out, y_gm, [row0, col0], real_shape=[valid_rows, valid_cols]
        )


def _align_up(value, align):
    return ((value + align - 1) // align) * align


def _pow2_floor(value):
    result = 1
    while result * 2 <= value:
        result *= 2
    return result


def _plan_tiles(outer, half_len, dtype, ub_bytes, aivs):
    """Select (tile_rows, tile_cols, col_chunks) from layout and resources."""
    itemsize = _ITEMSIZE[dtype]
    coef = _BYTES_PER_LANE[dtype]
    align = max(1, 32 // itemsize)

    if half_len <= _COLUMN_TARGET:
        width = _align_up(half_len, align)
    else:
        width = _COLUMN_TARGET

    # A single tile's live buffers must fit the UB budget.
    while width > align and coef * width > ub_bytes:
        width = max(align, width // 2)

    rows = max(1, ub_bytes // (coef * width))
    rows = max(1, min(rows, outer))

    # Trade column width for tile height so short rows keep parallel work,
    # but only when the row count can actually use the extra height.
    if rows < _MIN_TILE_ROWS and outer >= _MIN_TILE_ROWS:
        while width > align and rows < _MIN_TILE_ROWS:
            width = max(align, width // 2)
            rows = max(1, min(outer, ub_bytes // (coef * width)))

    chunks = (half_len + width - 1) // width

    # With a single column chunk, bound the height by useful rows per AIV so
    # small inputs still spread over the vector cores.
    if chunks == 1:
        rows = min(rows, max(1, (outer + aivs - 1) // aivs))

    rows = _pow2_floor(max(1, rows))
    return rows, width, chunks


def swi_glu(input, dim=-1):
    """SwiGlu: split ``input`` in half along ``dim`` and return
    ``silu(x0) * x1`` with the split dimension halved."""
    if not isinstance(input, torch.Tensor):
        raise TypeError("swi_glu: input must be a torch.Tensor")
    if input.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "swi_glu: unsupported dtype %s (expected float16/float32/bfloat16)"
            % (input.dtype,)
        )
    rank = input.dim()
    if rank == 0:
        raise ValueError("swi_glu: input must have at least one dimension")
    dim = int(dim)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise IndexError(
            "swi_glu: dim %d out of range for rank %d" % (dim, rank)
        )
    shape = list(input.shape)
    split_size = shape[dim]
    if split_size % 2 != 0:
        raise ValueError(
            "swi_glu: split dimension size %d must be even" % (split_size,)
        )

    ensure_npu_platform()
    ub_bytes = query_ub_bytes() or _DEFAULT_UB_BYTES
    aivs = query_aivs() or _DEFAULT_AIVS

    out_shape = shape[:dim] + [split_size // 2] + shape[dim + 1:]
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    if out.numel() == 0:
        return out

    # The kernel addresses the flat row-major layout; guarantee contiguity
    # without ever moving data on the host beyond this metadata-driven view.
    x = input if input.is_contiguous() else input.contiguous()

    outer = 1
    for d in range(dim):
        outer *= shape[d]
    inner = 1
    for d in range(dim + 1, rank):
        inner *= shape[d]
    half_len = (split_size // 2) * inner

    tile_rows, tile_cols, col_chunks = _plan_tiles(
        outer, half_len, input.dtype, ub_bytes, aivs
    )
    row_tiles = (outer + tile_rows - 1) // tile_rows
    n_tiles = row_tiles * col_chunks
    blocks = max(1, int(vector_blocks(n_tiles)))

    unroll = 1
    if (
        n_tiles >= 2 * blocks
        and 2 * _BYTES_PER_LANE[input.dtype] * tile_rows * tile_cols
        <= ub_bytes
    ):
        unroll = 2

    _swiglu_tiles[blocks](
        x,
        out,
        outer,
        half_len,
        n_tiles,
        tile_rows,
        tile_cols,
        col_chunks,
        unroll,
    )
    return out
