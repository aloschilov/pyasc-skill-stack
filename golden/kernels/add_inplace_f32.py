#!/usr/bin/env python3.11
"""
pyasc kernel: add_inplace_f32

Operation: dedicated in-place element-wise add, float32:

    a <- a + b

The output aliases input ``a``: the kernel loads the ``a`` and ``b`` tiles,
computes ``a + b``, and ``asc2.store``s the result straight back into ``a``'s
GM tensor. There is NO separate output GM buffer — ``a_ptr`` is both an input
and the destination. This is the single-tensor aliasing demonstrator for the
"In-place" capability (contrast ``apply_adam``, which aliases each of several
inputs var/m/v as their own outputs).

Usage:
    python3.11 kernel.py -r Model -v Ascend950PR_9599
    pytest kernel.py --backend Model --platform Ascend950PR_9599

Alignment requirement: element count must be a multiple of
TILE_SIZE * CORE_NUM = 32768 (aligned_only, matching the apply_adam/abs/add cells).

Cell metadata (mirrors capabilities.yaml; do not drift):
  - shape_regime: fixed
  - reduce_axis: null
  - output_shape: same_as_input
  - accumulator_dtype: null
  - identity: null
  - tail_behavior: aligned_only
  - padding: null
  - partitioning: tile_per_core
  - unsupported_regimes: []
  - in_place: true

Non-obvious constraints:
  - Aliasing contract: the output buffer IS input ``a``. The kernel wraps a
    single GM tensor ``a_gm`` for ``a_ptr`` and uses it as both the first
    load source and the store destination. The host passes ONE ndarray for
    input+output (no ``np.empty_like``); ``a`` is mutated in place and returned.
  - Rank-consistent tiling: 1D tensor + 1D load shape + 1D offsets (Pattern A).
  - Alignment: ``size`` must be a multiple of TILE_SIZE * CORE_NUM = 32768.
  - UB placement: each ``a``/``b`` tile is loaded into UB, ``+`` runs on the
    AIV vector pipeline, and the result is stored straight back to GM. No
    L0 / cube involvement.
  - Tolerance: ``atol=rtol=1e-5`` — an exact f32 elementwise add.
  - Simulator/platform assumptions: ``Ascend950PR_9599`` (C310); numpy buffers
    are safe for this elementwise UB-only path.
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

TILE_SIZE = 2048
CORE_NUM = 16
ALIGNMENT = TILE_SIZE * CORE_NUM  # 32768 elements

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def add_inplace_kernel(a_ptr: asc.GlobalAddress, b_ptr: asc.GlobalAddress,
                       size: int, tile_size: asc.ConstExpr[int],
                       tile_per_block: asc.ConstExpr[int]):
    # a_gm is both an input AND the output: we store the sum back into it.
    a_gm = asc2.tensor(a_ptr, [size])
    b_gm = asc2.tensor(b_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        a = asc2.load(a_gm, [tile_size], offsets=[tile_offset])
        b = asc2.load(b_gm, [tile_size], offsets=[tile_offset])
        out = a + b
        asc2.store(out, a_gm, offsets=[tile_offset])


def add_inplace_launch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Host launcher; updates ``a`` in place (a <- a + b) and returns it.

    ``a`` is passed as BOTH the input and the output — no separate output
    ndarray is allocated. ``b`` is a read-only addend.
    """
    shape = a.shape
    # reshape(-1) returns a view for contiguous arrays, so the store lands
    # back in the caller's ``a`` buffer — this is the in-place aliasing.
    a_flat = a.reshape(-1)
    b_flat = np.ascontiguousarray(b.reshape(-1))
    size = a_flat.size
    if size % ALIGNMENT != 0:
        raise ValueError(f"add_inplace is aligned_only; size {size} not a multiple of {ALIGNMENT}")
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    add_inplace_kernel[CORE_NUM](a_flat, b_flat, size, TILE_SIZE,
                                 asc.ceildiv(num_tiles, CORE_NUM))
    return a_flat.reshape(shape)


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    test_shapes = [(16, 2048), (32, 4096)]
    rng = np.random.default_rng(seed=2026)
    for shape in test_shapes:
        a = (rng.random(shape, dtype=np.float32) * 2 - 1).astype(np.float32)
        b = (rng.random(shape, dtype=np.float32) * 2 - 1).astype(np.float32)
        # Compute the reference on copies BEFORE the in-place launch mutates a.
        expected = a.copy() + b.copy()
        out = add_inplace_launch(a, b)
        np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)
        logging.info(f"[PASS] In-place add verified for shape {shape}.")


def test_add_inplace_f32(backend: config.Backend, platform: config.Platform):
    """pytest entry point."""
    run_kernel(backend, platform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend: Model or NPU")
    parser.add_argument("-v", type=str, default="Ascend950PR_9599", help="platform/SoC version")
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
