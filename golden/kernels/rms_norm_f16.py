#!/usr/bin/env python3.10
"""
Golden reference: rms_norm_f16 kernel (asc2 API)
Row-wise RMSNorm using the asc2.rms_norm builtin for float16 tensors.
The number of rows is runtime-dynamic; the normalization (last) dim is
compile-time `ConstExpr`. All AI cores share the work via blockwise
distribution over the row dim.
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

CORE_NUM = 16
EPS = 1e-6

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def rms_norm_kernel(x_ptr: asc.GlobalAddress, gamma_ptr: asc.GlobalAddress,
                    out_ptr: asc.GlobalAddress,
                    num_rows: int, num_cols: asc.ConstExpr[int],
                    block_size: asc.ConstExpr[int],
                    eps: asc.ConstExpr[float]):
    x_gm = asc2.tensor(x_ptr, [num_rows, num_cols])
    gamma_gm = asc2.tensor(gamma_ptr, [num_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, num_cols])
    start_row = asc2.block_idx() * block_size
    rows = asc2.load(x_gm, [block_size, num_cols], offsets=[start_row, 0])
    gamma = asc2.load(gamma_gm, [num_cols], offsets=[0])
    out = asc2.rms_norm(rows, gamma, eps)
    asc2.store(out, out_gm, offsets=[start_row, 0])


def rms_norm_launch(x: np.ndarray, gamma: np.ndarray, eps: float = EPS) -> np.ndarray:
    out = np.empty_like(x)
    num_rows, num_cols = x.shape
    block_size = asc.ceildiv(num_rows, CORE_NUM)
    rms_norm_kernel[CORE_NUM](x, gamma, out, num_rows, num_cols, block_size, eps)
    return out


def rms_norm_numpy(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    mean_sq = (x32 * x32).mean(axis=-1, keepdims=True)
    y = x32 / np.sqrt(mean_sq + eps)
    return (y * gamma.astype(np.float32)).astype(x.dtype)


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    rng = np.random.default_rng(seed=2026)
    num_rows, num_cols = 32, 4096
    x = (rng.random((num_rows, num_cols), dtype=np.float32) * 2 - 1).astype(np.float16)
    gamma = (rng.random((num_cols,), dtype=np.float32) + 0.5).astype(np.float16)
    out = rms_norm_launch(x, gamma, EPS)
    expected = rms_norm_numpy(x, gamma, EPS)
    np.testing.assert_allclose(out.astype(np.float32),
                               expected.astype(np.float32),
                               atol=5e-2, rtol=5e-2)
    logging.info(f"[PASS] rms_norm verified for shape ({num_rows}, {num_cols}).")


def test_rms_norm_f16(backend: config.Backend, platform: config.Platform):
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
    logging.info(f"[INFO] Running kernel with backend={backend}, platform={platform}")
    run_kernel(backend, platform)
    logging.info("[INFO] Kernel run complete.")
