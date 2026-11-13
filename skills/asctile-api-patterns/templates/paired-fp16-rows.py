"""Two aligned FP16 row tiles per block with a small tail in the second wave.

Use after the shape, alignment, payload and enlarged-tile UB checks in
references/strided-row-tiles.md. N is the actual launch grid.
"""
import torch
from asc.experimental import asctile


def paired_fp16_height(rows, half, dtype, blocks, usable_ub):
    """Return zero when either half-row DMA or the enlarged UB tile cannot fit."""
    if dtype != torch.float16 or min(rows, half, blocks, usable_ub) <= 0:
        return 0
    if (half * 2) % 32:
        return 0
    height, tail = divmod(rows, 2 * blocks)
    if height < 2 or not 0 < tail <= height // 4:
        return 0
    if not 8192 <= height * half * 2 <= 16384:
        return 0
    if 20 * (height + tail) * half > usable_ub:
        return 0
    return height


@asctile.jit(reuse_alloc=2)
def paired_rows(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                R: asctile.ConstExpr[int], H: asctile.ConstExpr[int],
                T: asctile.ConstExpr[int], N: asctile.ConstExpr[int]):
    gx = asctile.global_tensor(x, [R, 2 * H])
    gy = asctile.global_tensor(y, [R, H])
    b = asctile.block_idx()
    tail = R - 2 * N * T
    r = b * T
    gate = asctile.copy_in(gx, [r, 0], [T, H])
    value = asctile.copy_in(gx, [r, H], [T, H])
    gate_f = gate.to(asctile.float32)
    out = (gate_f / (1.0 + asctile.exp(-gate_f))) * value.to(asctile.float32)
    asctile.copy_out(out.to(gate.dtype), gy, [r, 0])
    if b == N - 1:
        r_last = (N + b) * T
        gate_last = asctile.copy_in(gx, [r_last, 0], [T + tail, H])
        value_last = asctile.copy_in(gx, [r_last, H], [T + tail, H])
        gate_f_last = gate_last.to(asctile.float32)
        out_last = (gate_f_last / (1.0 + asctile.exp(-gate_f_last))) * value_last.to(asctile.float32)
        asctile.copy_out(out_last.to(gate_last.dtype), gy, [r_last, 0])
    else:
        r_other = (N + b) * T
        gate_other = asctile.copy_in(gx, [r_other, 0], [T, H])
        value_other = asctile.copy_in(gx, [r_other, H], [T, H])
        gate_f_other = gate_other.to(asctile.float32)
        out_other = (gate_f_other / (1.0 + asctile.exp(-gate_f_other))) * value_other.to(asctile.float32)
        asctile.copy_out(out_other.to(gate_other.dtype), gy, [r_other, 0])
