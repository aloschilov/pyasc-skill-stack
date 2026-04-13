#!/usr/bin/env python3.10
"""
pyasc kernel template: {kernel_name}

Operation: {describe operation}
Usage:
    python3.10 kernel.py -r Model -v Ascend910B1   # Run with simulator
    python3.10 kernel.py -r NPU                    # Run with NPU hardware
    pytest kernel.py --backend Model --platform Ascend910B1
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
def kernel_func(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                size: int, tile_size: asc.ConstExpr[int], tile_per_block: asc.ConstExpr[int]):
    """Replace this with your kernel implementation."""
    x_gm = asc2.tensor(x_ptr, [size])
    y_gm = asc2.tensor(y_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        y = asc2.load(y_gm, [tile_size], offsets=[tile_offset])
        # TODO: Replace with your operation
        out = x + y
        asc2.store(out, out_gm, offsets=[tile_offset])


def kernel_launch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    kernel_func[CORE_NUM](x, y, out, size, TILE_SIZE, asc.ceildiv(num_tiles, CORE_NUM))
    return out


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    # TODO: Update sizes, dtype, and expected result for your operation
    test_sizes = [8192, 131072]
    rng = np.random.default_rng(seed=2026)
    for size in test_sizes:
        x = rng.random(size, dtype=np.float32) * 10
        y = rng.random(size, dtype=np.float32) * 10
        out = kernel_launch(x, y)
        np.testing.assert_allclose(out, x + y, atol=1e-5, rtol=1e-5)
        logging.info(f"[PASS] Kernel output verified for size {size}.")


def test_kernel(backend: config.Backend, platform: config.Platform):
    """pytest entry point — uses conftest.py fixtures for backend/platform."""
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
    logging.info(f"[INFO] Running kernel with backend={backend}, platform={platform}")
    run_kernel(backend, platform)
    logging.info("[INFO] Kernel run complete.")
