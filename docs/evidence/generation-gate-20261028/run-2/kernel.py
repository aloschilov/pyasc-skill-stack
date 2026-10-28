#!/usr/bin/env python3.10
"""
pyasc kernel: vector add

Operation: out = x + y for two contiguous float32 tensors of size 1024.
Usage:
    python3.10 kernel.py
"""

import torch

# The published pyasc-dev wheel exposes asctile as asc.experimental.asctile;
# older source builds expose it as a top-level module.
try:
    from asc.experimental import asctile
except ImportError:  # pragma: no cover - legacy source build
    import asctile

SIZE = 1024
TILE_SIZE = 128
CORE_NUM = 72  # Ascend950PR_9599


@asctile.jit
def vector_add(x_ptr: asctile.GlobalAddress,
               y_ptr: asctile.GlobalAddress,
               out_ptr: asctile.GlobalAddress,
               size: asctile.ConstExpr[int],
               tile_size: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    num_tiles = asctile.ceildiv(size, tile_size)
    for tile_id in asctile.range(
        asctile.block_idx(), num_tiles, asctile.block_num(), unroll_factor=1
    ):
        tile_offset = tile_id * tile_size
        valid = size - tile_offset
        if valid > tile_size:
            valid = tile_size
        x = asctile.copy_in(
            x_gm, [tile_offset], [tile_size], real_shape=[valid], pad_value=0
        )
        y = asctile.copy_in(
            y_gm, [tile_offset], [tile_size], real_shape=[valid], pad_value=0
        )
        out = x + y
        asctile.copy_out(out, out_gm, [tile_offset], real_shape=[valid])


def vector_add_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    size = x.numel()
    num_tiles = (size + TILE_SIZE - 1) // TILE_SIZE
    core_num = min(CORE_NUM, num_tiles)
    vector_add[core_num](x, y, out, size, TILE_SIZE)
    return out


if __name__ == "__main__":
    torch.manual_seed(2026)
    x = torch.randn(SIZE, dtype=torch.float32)
    y = torch.randn(SIZE, dtype=torch.float32)
    out = vector_add_launch(x, y)
    expected = x + y
    if not torch.allclose(out, expected):
        raise AssertionError("vector add verification failed")
    print("[PASS] vector add verified with torch.allclose")
