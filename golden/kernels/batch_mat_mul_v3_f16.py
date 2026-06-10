#!/usr/bin/env python3.11
"""Golden kernel: batch_mat_mul_v3/float16 (CUBE-only)

Batched matrix multiplication ``C[b] = A[b] @ B[b]`` for float16 inputs with a
float32 cube accumulator, stored as float32 output. This is the initial
CUBE-only operator-generation demo: the operator runs entirely on the cube unit
(``asc2.matmul`` / ``@`` -> L0C), with the batch axis distributed across cores.

Composed from the asc2 cube patterns:
  - MN-block tiling: pyasc-v2-eval/python/test/asc2/kernels/test_matmul_mnblock.py
    (mirrored by golden/kernels/matmul_f16.py) -- per (m,n) block, load A->L0A,
    B->L0B, ``A @ B`` -> L0C.

Reference for the perf gate: the canonical ops-nn BatchMatMulV3 operator
(``aclnnBatchMatMul``), measured on the same Ascend950PR_9599 / dav_3510 camodel.
The AscendC matmul samples (samples/operator/ascendc/.../MatmulCustom*) are
single-GEMM and only ground the cube skill guidance.

Cell metadata (mirrors capabilities.yaml; do not drift):
  - shape_regime: fixed
  - reduce_axis: -1               # K is the reduction axis (per batch)
  - output_shape: [B, M, N]
  - accumulator_dtype: float32    # cube accumulator is always f32, output also f32
  - identity: "0"
  - tail_behavior: aligned_only
  - padding: null
  - partitioning: batch_per_core  # one batch matrix per cube core
  - unsupported_regimes: [non_16_multiple_shapes, broadcast_batch, k_tiled]

Non-obvious constraints:
  - Layout: the [B,M,K] / [B,K,N] / [B,M,N] tensors are flattened on the host to
    2D [B*M, K] / [B*K, N] / [B*M, N] so each batch is a contiguous row-block.
    Batch ``bi`` reads rows ``bi*M`` of A / ``bi*K`` of B and writes rows
    ``bi*M`` of C -- all 2D loads/offsets, matching the cube tile rules.
  - Tiling: M and N are tiled (``M_TILE`` x ``N_TILE``) to fit the L0A/L0B/L0C
    cube buffers; the full K dimension is loaded into one A tile (K fits at
    M_TILE=128, K=256 -> 64 KiB L0A). All of M, K, N, M_TILE, N_TILE MUST be
    multiples of 16.
  - Dtype: f16 inputs, f32 cube accumulate, f32 output (cube accumulator is
    always f32 and stored directly without casting).
  - CRITICAL platform: Ascend950PR_9599 (C310) is the only platform exposing the
    cube unit; this kernel will not run elsewhere.
  - CRITICAL host buffers: inputs MUST be torch.Tensor (numpy is silently zeroed
    on the C310 cube path).
"""

import logging
import argparse
import torch

import asc
import asc.runtime.config as config
import asc2

# Contract shape for the perf demo: B=16, M=K=N=256, float16.
BATCH = 16
M = 256
K = 256
N = 256
M_TILE = 128
# N_TILE=64 keeps the double-buffered (parallel) L0B b-tile within the 64 KiB L0B
# budget: [K=256, 64] f16 = 32 KiB per buffer -> 64 KiB for the 2-deep pipeline.
N_TILE = 64
CORE_NUM = BATCH  # one batch matrix per cube core

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def bmm_kernel(a_ptr: asc.GlobalAddress, b_ptr: asc.GlobalAddress, c_ptr: asc.GlobalAddress,
               a_shape: asc.ConstExpr, b_shape: asc.ConstExpr, c_shape: asc.ConstExpr,
               m: asc.ConstExpr[int], k: asc.ConstExpr[int], n: asc.ConstExpr[int],
               m_tile: asc.ConstExpr[int], n_tile: asc.ConstExpr[int]):
    """Batched matmul: one batch matrix per core, MN-block tiled, K loaded whole.

    a_gm/b_gm/c_gm are 2D views [B*M, K] / [B*K, N] / [B*M, N]; batch ``bi`` is
    the row-block starting at ``bi*m`` (A/C) and ``bi*k`` (B).
    """
    a_gm = asc2.tensor(a_ptr, a_shape)
    b_gm = asc2.tensor(b_ptr, b_shape)
    c_gm = asc2.tensor(c_ptr, c_shape)
    bi = asc2.block_idx()
    a_row0 = bi * m
    b_row0 = bi * k
    m_tiles = m // m_tile
    n_tiles = n // n_tile
    # Direct GM->L0A/L0B loads for each tile. The M loop loads A tiles once per
    # m-tile (A stays resident in L0A across the N sweep). The N loop loads B
    # tiles with parallel=True + unroll_factor=2 to overlap the next B load with
    # the current MMAD, mirroring test_matmul_tiled.py's double-buffering.
    for i in range(m_tiles):
        m_off = i * m_tile
        a_i = asc2.load(a_gm, [m_tile, k], offsets=[a_row0 + m_off, 0], location=asc2.TileLocation.L0A)
        for j in asc2.range(n_tiles, unroll_factor=2, parallel=True):
            n_off = j * n_tile
            b_j = asc2.load(b_gm, [k, n_tile], offsets=[b_row0, n_off], location=asc2.TileLocation.L0B)
            c_ij = a_i @ b_j
            asc2.store(c_ij, c_gm, offsets=[a_row0 + m_off, n_off])


def bmm_launch(a: torch.Tensor, b: torch.Tensor,
               m_tile: int = M_TILE, n_tile: int = N_TILE,
               core_num: int = CORE_NUM) -> torch.Tensor:
    """Host launcher for batched matmul.

    Args:
        a: [B, M, K] float16 tensor.
        b: [B, K, N] float16 tensor.
    Returns:
        [B, M, N] float32 tensor (C[b] = A[b] @ B[b]).
    """
    batch, m, k = a.shape
    _, _, n = b.shape
    a2d = a.reshape(batch * m, k).contiguous()
    b2d = b.reshape(batch * k, n).contiguous()
    c2d = torch.zeros((batch * m, n), dtype=torch.float32)
    bmm_kernel[core_num](a2d, b2d, c2d, a2d.shape, b2d.shape, c2d.shape,
                         m, k, n, m_tile, n_tile)
    return c2d.reshape(batch, m, n)


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    test_cases = [
        # (batch, m, k, n, m_tile, n_tile)
        (4, 32, 32, 32, 16, 16),
        (8, 64, 64, 64, 64, 64),
        (BATCH, M, K, N, M_TILE, N_TILE),
    ]
    dtype = torch.float16
    torch.manual_seed(2026)

    for batch, m, k, n, m_tile, n_tile in test_cases:
        a = torch.rand((batch, m, k), dtype=dtype)
        b = torch.rand((batch, k, n), dtype=dtype)
        c = bmm_launch(a, b, m_tile, n_tile, core_num=batch)
        c_ref = torch.bmm(a.to(torch.float32), b.to(torch.float32))
        torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)
        logging.info(f"[PASS] Kernel output verified for [{batch},{m},{k}]x[{batch},{k},{n}].")


def test_batch_mat_mul_v3_f16(backend: config.Backend, platform: config.Platform):
    """pytest entry point."""
    run_kernel(backend, platform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend: Model or NPU")
    parser.add_argument("-v", type=str, default="Ascend950PR_9599",
                        help="platform/SoC version (matmul requires Ascend950PR_9599)")
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
