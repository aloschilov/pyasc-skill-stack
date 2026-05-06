#!/usr/bin/env python3.10
"""
pyasc kernel: vector add (vadd) — asc2 API

Operation: out = x + y (element-wise)

This kernel demonstrates the complete asc2 development pattern:
- @asc2.jit decoration with always_compile=True
- asc2.tensor for global memory wrapping
- asc2.load / asc2.store for tile-based memory access
- NumPy-like arithmetic (x + y on tiles)
- asc2.range for tile iteration
- Multi-core launch with kernel[core_num](...)
- np.testing.assert_allclose verification

Based on: ~/workspace/pyasc/python/test/kernels/asc2/test_vadd.py

Usage:
    python3.10 kernel.py -r Model -v Ascend950PR_9599   # Run with simulator
    python3.10 kernel.py -r NPU                         # Run with NPU hardware
    pytest kernel.py --backend Model --platform Ascend950PR_9599
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

TILE_SIZE = 128
CORE_NUM = 16

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def vadd_kernel(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                size: int, tile_size: asc.ConstExpr[int], tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    y_gm = asc2.tensor(y_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        y = asc2.load(y_gm, [tile_size], offsets=[tile_offset])
        out = x + y
        asc2.store(out, out_gm, offsets=[tile_offset])


def vadd_launch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    vadd_kernel[CORE_NUM](x, y, out, size, TILE_SIZE, asc.ceildiv(num_tiles, CORE_NUM))
    return out


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    rng = np.random.default_rng(seed=2026)
    size = 8192
    x = rng.random(size, dtype=np.float32) * 10
    y = rng.random(size, dtype=np.float32) * 10
    out = vadd_launch(x, y)
    np.testing.assert_allclose(out, x + y, atol=1e-5, rtol=1e-5)
    logging.info("[PASS] Kernel output verified: out = x + y")


def test_vadd(backend: config.Backend, platform: config.Platform):
    """pytest entry point."""
    run_kernel(backend, platform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend: Model or NPU")
    parser.add_argument("-v", type=str, default=None, help="platform/SoC version")
    args = parser.parse_args()
    backend = args.r
    platform = args.v
    if backend not in config.Backend.__members__:
        raise ValueError(f"Unsupported Backend! Supported: {list(config.Backend.__members__.keys())}")
    backend = config.Backend(backend)
    if platform is not None:
        platform_values = [p.value for p in config.Platform]
        if platform not in platform_values:
            raise ValueError(f"Unsupported Platform! Supported: {platform_values}")
        platform = config.Platform(platform)
    logging.info("[INFO] Running vadd kernel.")
    run_kernel(backend, platform)
    logging.info("[INFO] vadd kernel run complete.")
