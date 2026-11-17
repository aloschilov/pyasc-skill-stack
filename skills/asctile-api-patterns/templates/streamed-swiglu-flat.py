"""Process [R, 2H] in full contiguous chunks and one partial chunk per row."""

import ctypes
from math import prod

import torch
from asc.experimental import asctile


@asctile.jit(always_compile=False)
def swiglu_flat(
    x: asctile.GlobalAddress,
    y: asctile.GlobalAddress,
    R: asctile.ConstExpr[int],
    H: asctile.ConstExpr[int],
    T: asctile.ConstExpr[int],
    U: asctile.ConstExpr[int],
    cache_input: asctile.ConstExpr[bool] = True,
):
    gx = asctile.global_tensor(x, [2 * R * H], l2_cache=cache_input)
    gy = asctile.global_tensor(y, [R * H])
    whole = H // T
    nfull = R * whole
    blocks = asctile.block_num()
    block = asctile.block_idx()
    q = nfull // blocks
    rem = nfull % blocks
    begin = q * block + min(block, rem)
    end = q * (block + 1) + min(block + 1, rem)
    for t in asctile.range(begin, end, unroll_factor=U):
        row = t // max(1, whole)
        col = (t % max(1, whole)) * T
        gate_tile = asctile.copy_in(
            gx, [row * 2 * H + col], [T]
        )
        value_tile = asctile.copy_in(
            gx, [row * 2 * H + H + col], [T]
        )

        gate_f = gate_tile.to(asctile.float32)
        value_f = value_tile.to(asctile.float32)

        sig = gate_f / (1.0 + asctile.exp(-gate_f))
        out_f = sig * value_f

        out_dtype = gate_tile.dtype
        out_tile = out_f.to(out_dtype)

        asctile.copy_out(out_tile, gy, [row * H + col])
    if H % T:
        for row in asctile.range(block, R, blocks):
            col = whole * T
            valid = H % T
            gate_tile = asctile.copy_in(
                gx, [row * 2 * H + col], [T], real_shape=[valid], pad_value=0.0
            )
            value_tile = asctile.copy_in(
                gx, [row * 2 * H + H + col], [T], real_shape=[valid], pad_value=0.0
            )

            gate_f = gate_tile.to(asctile.float32)
            value_f = value_tile.to(asctile.float32)

            sig = gate_f / (1.0 + asctile.exp(-gate_f))
            out_f = sig * value_f

            out_dtype = gate_tile.dtype
            out_tile = out_f.to(out_dtype)

            asctile.copy_out(out_tile, gy, [row * H + col], real_shape=[valid])


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Bounded, single-block baseline for an even split along any dimension."""
    rank = input.ndim
    if not -rank <= dim < rank:
        raise ValueError(f"Invalid split dimension {dim} for rank {rank}")
    axis = dim % rank
    shape = list(input.shape)
    if shape[axis] % 2:
        raise ValueError("SwiGLU needs two equal halves along the split dimension")

    rows = prod(shape[:axis])
    shape[axis] //= 2
    half = prod(shape[axis:])
    out = input.new_empty(shape)
    if out.numel() == 0:
        return out

    # Keep this starting point small enough for a single unrolled stage.
    alignment = 32 // input.element_size()
    tile = min(256, (half + alignment - 1) // alignment * alignment)
    grid = 1
    if input.device.type == "npu":
        asctile.set_platform(asctile.Backend.NPU, asctile.runtime.current_platform())
        stream = torch.npu.current_stream(input.device).npu_stream
        grid = (1, ctypes.c_void_p(stream))
    swiglu_flat[grid](input.contiguous(), out, rows, half, tile, 1)
    return out
