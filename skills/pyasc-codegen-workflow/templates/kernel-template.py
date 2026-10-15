#!/usr/bin/env python3.10
"""
pyasc kernel template: {kernel_name}

Operation: {describe operation}
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
import asctile

TILE_SIZE = 128
CORE_NUM = 72  # Ascend950PR_9599; derive from target metadata in reusable code.

logging.basicConfig(level=logging.INFO)


@asctile.jit(always_compile=True)
def kernel_func(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                size: asc.ConstExpr[int], tile_size: asc.ConstExpr[int]):
    """Replace this with your kernel implementation."""
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    num_tiles = asctile.ceildiv(size, tile_size)
    for tile_id in asctile.range(
        asctile.block_idx(), num_tiles, asctile.block_num(), unroll_factor=2
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
        # TODO: Replace with your operation
        out = x + y
        asctile.copy_out(out, out_gm, [tile_offset], real_shape=[valid])


def kernel_launch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    core_num = min(CORE_NUM, num_tiles)
    kernel_func[core_num](x, y, out, size, TILE_SIZE)
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
