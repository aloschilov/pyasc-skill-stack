# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

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
    core_num = 16
    tile_size = 128
    num_tiles = asc.ceildiv(size, tile_size)
    vadd_kernel[core_num](x, y, out, size, tile_size, asc.ceildiv(num_tiles, core_num))
    return out


def vadd_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    rng = np.random.default_rng(seed=2026)
    size = 8192
    x = rng.random(size, dtype=np.float32) * 10
    y = rng.random(size, dtype=np.float32) * 10
    out = vadd_launch(x, y)
    np.testing.assert_allclose(out, x + y)


def test_vadd(backend: config.Backend, platform: config.Platform):
    """pytest entry point."""
    vadd_custom(backend, platform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend to run")
    parser.add_argument("-v", type=str, default=None, help="platform to run")
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
    logging.info("[INFO] start process sample add.")
    vadd_custom(backend, platform)
    logging.info("[INFO] Sample add run success.")
