#!/usr/bin/env python3.10
"""
pyasc kernel: vector_add (float32, size 1024)

Operation: out = x + y, elementwise over a flat float32 vector.

Usage:
    python3.10 kernel.py -r Model -v Ascend950PR_9599   # Run with simulator
    python3.10 kernel.py -r NPU                         # Run with NPU hardware
"""

import logging

import torch

import asc

# The published pyasc-dev wheel exposes asctile as asc.experimental.asctile;
# older source builds expose it as a top-level module.
try:
    from asc.experimental import asctile
except ImportError:  # pragma: no cover - legacy source build
    import asctile

SIZE = 1024
TILE_SIZE = 256
CORE_NUM = 72  # Ascend950PR_9599; derive from target metadata in reusable code.

logging.basicConfig(level=logging.INFO)


@asctile.jit(always_compile=True)
def add_kernel(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress,
               out_ptr: asc.GlobalAddress, size: asc.ConstExpr[int],
               tile_size: asc.ConstExpr[int]):
    """Elementwise vector add: out[i] = x[i] + y[i]."""
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    num_tiles = asctile.ceildiv(size, tile_size)
    for tile_id in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num()):
        tile_offset = tile_id * tile_size
        valid = min(size - tile_offset, tile_size)
        x = asctile.copy_in(
            x_gm, [tile_offset], [tile_size], real_shape=[valid], pad_value=0
        )
        y = asctile.copy_in(
            y_gm, [tile_offset], [tile_size], real_shape=[valid], pad_value=0
        )
        out = x + y
        asctile.copy_out(out, out_gm, [tile_offset], real_shape=[valid])


def add_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Host wrapper: allocates the output and launches the JIT kernel."""
    out = torch.empty_like(x)
    size = out.numel()
    num_tiles = (size + TILE_SIZE - 1) // TILE_SIZE
    core_num = min(CORE_NUM, num_tiles)
    add_kernel[core_num](x, y, out, size, TILE_SIZE)
    return out


def verify() -> None:
    """torch.allclose verification of the kernel against the host reference."""
    torch.manual_seed(0)
    x = torch.randn(SIZE, dtype=torch.float32)
    y = torch.randn(SIZE, dtype=torch.float32)
    out = add_launch(x, y)
    if not torch.allclose(out, x + y, atol=1e-5, rtol=1e-5):
        raise ValueError("vector_add verification failed: output differs from x + y")
    logging.info("[PASS] vector_add verified for size %d.", SIZE)


if __name__ == "__main__":
    verify()
