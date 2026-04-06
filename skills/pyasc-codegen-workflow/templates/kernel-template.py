#!/usr/bin/env python3.10
"""
pyasc kernel template: {kernel_name}

Operation: {describe operation}
Usage:
    python3.10 kernel.py -r Model -v Ascend910B1   # Run with simulator
    python3.10 kernel.py -r NPU                    # Run with NPU hardware
"""

import logging
import argparse
import torch
try:
    import torch_npu
except ModuleNotFoundError:
    pass

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt

USE_CORE_NUM = 8
BUFFER_NUM = 2
TILE_NUM = 8

logging.basicConfig(level=logging.INFO)


@asc.jit
def kernel_func(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, block_length: int):
    """Replace this with your kernel implementation."""
    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset, block_length)
    y_gm.set_global_buffer(y + offset, block_length)
    z_gm.set_global_buffer(z + offset, block_length)

    tile_length = block_length // TILE_NUM // BUFFER_NUM
    data_type = x.dtype
    buffer_size = tile_length * BUFFER_NUM * data_type.sizeof()

    x_local = asc.LocalTensor(data_type, asc.TPosition.VECIN, 0, tile_length * BUFFER_NUM)
    y_local = asc.LocalTensor(data_type, asc.TPosition.VECIN, buffer_size, tile_length * BUFFER_NUM)
    z_local = asc.LocalTensor(data_type, asc.TPosition.VECOUT, buffer_size + buffer_size, tile_length * BUFFER_NUM)

    for i in range(TILE_NUM * BUFFER_NUM):
        buf_id = i % BUFFER_NUM

        asc.data_copy(x_local[buf_id * tile_length:], x_gm[i * tile_length:], tile_length)
        asc.data_copy(y_local[buf_id * tile_length:], y_gm[i * tile_length:], tile_length)

        asc.set_flag(asc.HardEvent.MTE2_V, buf_id)
        asc.wait_flag(asc.HardEvent.MTE2_V, buf_id)

        # TODO: Replace with your operation
        asc.add(z_local[buf_id * tile_length:], x_local[buf_id * tile_length:], y_local[buf_id * tile_length:],
                tile_length)

        asc.set_flag(asc.HardEvent.V_MTE3, buf_id)
        asc.wait_flag(asc.HardEvent.V_MTE3, buf_id)

        asc.data_copy(z_gm[i * tile_length:], z_local[buf_id * tile_length:], tile_length)

        asc.set_flag(asc.HardEvent.MTE3_MTE2, buf_id)
        asc.wait_flag(asc.HardEvent.MTE3_MTE2, buf_id)


def kernel_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(x)
    total_length = z.numel()
    block_length = total_length // USE_CORE_NUM
    kernel_func[USE_CORE_NUM, rt.current_stream()](x, y, z, block_length)
    return z


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    size = 8 * 2048
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    z = kernel_launch(x, y)
    # TODO: Update expected result for your operation
    assert torch.allclose(z, x + y), f"Output mismatch! Max diff: {(z - (x + y)).abs().max()}"
    logging.info("[PASS] Kernel output verified.")


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
