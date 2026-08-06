#!/usr/bin/env python3.10
"""Golden kernel: layer_norm_v4/bfloat16

High-level LayerNorm (mean subtraction + beta) with full_row and split_d
kernels plus a host-side N-D flatten dispatcher. bf16 in / f32 reduce.

Cell metadata (mirrors capabilities.yaml; do not drift):
  - shape_regime: dynamic         # full_row vs split_d via dispatcher
  - reduce_axis: -1
  - output_shape: same_as_input
  - accumulator_dtype: float32
  - identity: "0"
  - tail_behavior: host_dispatcher
  - padding: null                 # split_d uses host zero-pad to tile_cols
  - partitioning: host_dispatcher
  - unsupported_regimes: []
  - dispatcher_note: layer_norm_v4_launch(x, gamma, beta, eps) picks
    full_row when num_cols % 8 == 0 and num_cols*4*6 <= 64KB, else split_d.
"""

import argparse
import logging

import asc
import asc.runtime.config as config
import asc2
import torch
import torch.nn.functional as F

CORE_NUM = 8
TILE_COLS = 64
EPS = 1e-5
UB_BUDGET_BYTES = 64 * 1024
F32_INTERMEDIATE_FACTOR = 6
ALIGN = 8

logging.basicConfig(level=logging.INFO)

BF16_SHAPES = [
    [8, 4096],
    [8, 1024],
]


@asc2.jit(always_compile=True)
def layer_norm_v4_full_row_kernel(
    x_ptr: asc.GlobalAddress,
    gamma_ptr: asc.GlobalAddress,
    beta_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    num_rows: int,
    num_cols: int,
    padded_cols: asc.ConstExpr[int],
    epsilon: asc.ConstExpr[float],
):
    x_gm = asc2.tensor(x_ptr, [num_rows, padded_cols])
    gamma_gm = asc2.tensor(gamma_ptr, [1, padded_cols])
    beta_gm = asc2.tensor(beta_ptr, [1, padded_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, padded_cols])

    for row in asc2.range(
        asc2.block_idx(), num_rows, asc2.block_num(), unroll_factor=2
    ):
        x_row = asc2.load(x_gm, [1, padded_cols], offsets=[row, 0])
        x_f32 = x_row.to(asc.float32)

        mean = asc2.reduce_sum(x_f32, 1, keep_dims=True) / num_cols
        mean_b = asc2.broadcast_to(mean, 1, padded_cols)
        sum_x2 = asc2.reduce_sum(x_f32 * x_f32, 1, keep_dims=True) / num_cols
        var = sum_x2 - mean * mean
        rstd = asc2.rsqrt(var + epsilon)
        rstd_b = asc2.broadcast_to(rstd, 1, padded_cols)
        xc = x_f32 - mean_b

        gamma_row = asc2.load(gamma_gm, [1, padded_cols], offsets=[0, 0]).to(asc.float32)
        beta_row = asc2.load(beta_gm, [1, padded_cols], offsets=[0, 0]).to(asc.float32)
        out_f32 = xc * rstd_b * gamma_row + beta_row
        asc2.store(out_f32.to(x_row.dtype), out_gm, offsets=[row, 0])


@asc2.jit(always_compile=True)
def layer_norm_v4_split_d_kernel(
    x_ptr: asc.GlobalAddress,
    gamma_ptr: asc.GlobalAddress,
    beta_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    num_rows: int,
    num_cols: int,
    padded_cols: int,
    num_tiles: int,
    tile_cols: asc.ConstExpr[int],
    epsilon: asc.ConstExpr[float],
):
    x_gm = asc2.tensor(x_ptr, [num_rows, padded_cols])
    gamma_gm = asc2.tensor(gamma_ptr, [1, padded_cols])
    beta_gm = asc2.tensor(beta_ptr, [1, padded_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, padded_cols])

    for row in asc2.range(
        asc2.block_idx(), num_rows, asc2.block_num(), unroll_factor=2
    ):
        zero_seed = asc2.full([1, tile_cols], 0.0, dtype=asc.float32)
        sum_x = asc2.reduce_sum(zero_seed)
        sum_x2 = asc2.reduce_sum(zero_seed)
        for tile_id in asc2.range(num_tiles, unroll_factor=2):
            col = tile_id * tile_cols
            x = asc2.load(x_gm, [1, tile_cols], offsets=[row, col])
            x_f32 = x.to(asc.float32)
            sum_x = sum_x + asc2.reduce_sum(x_f32)
            sum_x2 = sum_x2 + asc2.reduce_sum(x_f32 * x_f32)

        mean = sum_x / num_cols
        var = sum_x2 / num_cols - mean * mean
        rstd = 1.0 / asc2.sqrt(var + epsilon)

        for tile_id in asc2.range(num_tiles, unroll_factor=2):
            col = tile_id * tile_cols
            x = asc2.load(x_gm, [1, tile_cols], offsets=[row, col])
            gamma = asc2.load(gamma_gm, [1, tile_cols], offsets=[0, col])
            beta = asc2.load(beta_gm, [1, tile_cols], offsets=[0, col])
            x_f32 = x.to(asc.float32)
            gamma_f32 = gamma.to(asc.float32)
            beta_f32 = beta.to(asc.float32)
            mean_tile = asc2.full([1, tile_cols], mean, dtype=asc.float32)
            xc = x_f32 - mean_tile
            out_f32 = xc * rstd * gamma_f32 + beta_f32
            asc2.store(out_f32.to(x.dtype), out_gm, offsets=[row, col])


def _use_split_d(num_cols: int) -> bool:
    if num_cols % 8 != 0:
        return True
    return num_cols * 4 * F32_INTERMEDIATE_FACTOR > UB_BUDGET_BYTES


def _pad_cols_align(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_cols: int
):
    padded_cols = ((num_cols + ALIGN - 1) // ALIGN) * ALIGN
    if padded_cols == num_cols:
        return x, gamma, beta, padded_cols
    x_pad = torch.zeros((x.shape[0], padded_cols), dtype=x.dtype, device=x.device)
    x_pad[:, :num_cols] = x
    gamma_pad = torch.zeros((padded_cols,), dtype=gamma.dtype, device=gamma.device)
    gamma_pad[:num_cols] = gamma
    beta_pad = torch.zeros((padded_cols,), dtype=beta.dtype, device=beta.device)
    beta_pad[:num_cols] = beta
    return x_pad, gamma_pad, beta_pad, padded_cols


def _pad_cols_tile(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_cols: int
):
    align_pad = ((num_cols + ALIGN - 1) // ALIGN) * ALIGN
    padded_cols = ((align_pad + TILE_COLS - 1) // TILE_COLS) * TILE_COLS
    if padded_cols == num_cols:
        return x, gamma, beta, padded_cols
    x_pad = torch.zeros((x.shape[0], padded_cols), dtype=x.dtype, device=x.device)
    x_pad[:, :num_cols] = x
    gamma_pad = torch.zeros((padded_cols,), dtype=gamma.dtype, device=gamma.device)
    gamma_pad[:num_cols] = gamma
    beta_pad = torch.zeros((padded_cols,), dtype=beta.dtype, device=beta.device)
    beta_pad[:num_cols] = beta
    return x_pad, gamma_pad, beta_pad, padded_cols


def _full_row_launch(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float, core_num: int
) -> torch.Tensor:
    num_rows, num_cols = x.shape
    x_w, gamma_w, beta_w, padded_cols = _pad_cols_align(x, gamma, beta, num_cols)
    out_pad = torch.empty_like(x_w)
    layer_norm_v4_full_row_kernel[core_num](
        x_w, gamma_w, beta_w, out_pad, num_rows, num_cols, padded_cols, eps
    )
    return out_pad[:, :num_cols].clone()


def _split_d_launch(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float, core_num: int
) -> torch.Tensor:
    num_rows, num_cols = x.shape
    x_w, gamma_w, beta_w, padded_cols = _pad_cols_tile(x, gamma, beta, num_cols)
    out_pad = torch.zeros_like(x_w)
    num_tiles = padded_cols // TILE_COLS
    layer_norm_v4_split_d_kernel[core_num](
        x_w, gamma_w, beta_w, out_pad, num_rows, num_cols, padded_cols, num_tiles, TILE_COLS, eps
    )
    return out_pad[:, :num_cols].clone()


def layer_norm_v4_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = EPS,
    core_num: int = CORE_NUM,
) -> torch.Tensor:
    orig_shape = x.shape
    cols = orig_shape[-1]
    rows = x.numel() // cols
    x2d = x.reshape(rows, cols)
    gamma_v = gamma.reshape(cols)
    beta_v = beta.reshape(cols)

    if _use_split_d(cols):
        out2d = _split_d_launch(x2d, gamma_v, beta_v, eps, core_num)
    else:
        out2d = _full_row_launch(x2d, gamma_v, beta_v, eps, core_num)
    return out2d.reshape(orig_shape)


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    rng = torch.Generator().manual_seed(2026)

    for shape in BF16_SHAPES:
        cols = shape[-1]
        x = torch.randn(shape, generator=rng, dtype=torch.float32).to(torch.bfloat16)
        gamma = (torch.randn((cols,), generator=rng, dtype=torch.float32) * 0.5 + 1.0).to(
            torch.bfloat16
        )
        beta = (torch.randn((cols,), generator=rng, dtype=torch.float32) * 0.1).to(
            torch.bfloat16
        )
        out = layer_norm_v4_launch(x, gamma, beta, EPS)
        expected = F.layer_norm(x.float(), [cols], gamma.float(), beta.float(), EPS).to(
            torch.bfloat16
        )
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
        logging.info("[PASS] shape %s", shape)


def test_layer_norm_v4_bf16(backend: config.Backend, platform: config.Platform):
    run_kernel(backend, platform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend: Model or NPU")
    parser.add_argument(
        "-v",
        type=str,
        default="Ascend950PR_9599",
        help="platform/SoC version (LayerNorm requires C310)",
    )
    args = parser.parse_args()
    backend = config.Backend(args.r)
    if args.v is not None:
        platform = config.Platform(args.v)
    else:
        platform = None
    logging.info("[INFO] Running kernel with backend=%s, platform=%s", backend, platform)
    run_kernel(backend, platform)
    logging.info("[INFO] Kernel run complete.")
