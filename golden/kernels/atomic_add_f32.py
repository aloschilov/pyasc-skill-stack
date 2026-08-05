#!/usr/bin/env python3.11
"""
pyasc kernel: atomic_add_f32

Operation: multi-core atomic-add reduction into a SHARED global-memory
buffer, float32:

    out[j] = sum_i in[i, j]        (i over CORE_NUM cores, j over N)

Each of ``CORE_NUM`` blocks owns one row ``block_idx()`` of a ``[CORE_NUM, N]``
input (packed flat as ``[CORE_NUM * N]``), ``copy_in``s its rank-1 ``[N]``
slice at offset ``block_idx() * N``, and ``asc2.atomic_add``s it into the SAME
shared ``[N]`` output GM buffer at ``offsets=[0]``. The hardware serialises the
overlapping cross-core writes, so the adds accumulate deterministically and the
shared output ends up holding the column-wise sum across cores. There is NO
per-tile reduce — each core contributes its whole slice to the shared buffer.

This is the dedicated demonstrator for the M3 "Atomic" family and the enabling
primitive for scatter-add / histogram / segment-sum / multi-core reduce-to-GM.
It uses the **fork target-test API** (``asc2.global_tensor`` / ``asc2.copy_in``
/ ``asc2.atomic_add``, torch tensors) — the same surface as
``pyasc-fork/python/test/asc2/operations/test_atomic_ops.py`` (which passes on
C310 across atomic_add/max/min and int16/int32/float16/bfloat16/float32).

Usage:
    python3.11 kernel.py -r Model -v Ascend950PR_9599
    pytest kernel.py --backend Model --platform Ascend950PR_9599

Alignment requirement: N=4096 is a multiple of the 32-byte f32 vector lane
(aligned_only, matching the elementwise cells).

Cell metadata (mirrors capabilities.yaml; do not drift):
  - tier: advanced
  - shape_regime: fixed
  - reduce_axis: null            # cross-core GM accumulation, not a tile-axis reduce
  - output_shape: [4096]         # shared dst is smaller than the packed input
  - accumulator_dtype: null      # accumulation happens in GM at dst dtype
  - identity: "0"                # additive identity of the RMW
  - tail_behavior: aligned_only
  - padding: null
  - partitioning: tile_per_core
  - unsupported_regimes: []

Non-obvious constraints:
  - Shared-destination contract: ALL cores atomic_add into the SAME [N] output
    region at offsets=[0]; the destination is smaller than the [CORE_NUM, N]
    input. The overlap is intentional — it is what produces the column sum.
  - MANDATORY host zero-init: atomic_add ACCUMULATES into the destination, so
    the host must zero the output buffer before launch. Passing a non-zeroed
    buffer adds its prior contents to the result (out += sum_i in[i, j]).
  - Rank-consistent tiling: 1D input tensor + 1D copy_in shape + 1D offsets,
    and a 1D [N] dst with 1D offsets=[0]. Never mix ranks.
  - Supported dtypes for the op: int16 / int32 / float16 / bfloat16 / float32;
    this golden pins float32. src and dst dtypes must match.
  - Torch tensors on C310: the Ascend950PR_9599 (C310) simulator path expects
    torch tensors (numpy is silently zeroed for this path, as with matmul).
  - Tolerance: ``atol=rtol=1e-3`` — f32 accumulation on the simulator.
  - Generalizes to atomic_max/atomic_min by swapping the op (identity +inf/-inf
    and the torch reference maximum/minimum); see references/atomic-rmw.md.
"""

import logging
import argparse
import torch

import asc
import asc.runtime.config as config
import asc2

CORE_NUM = 16
N = 4096

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def atomic_add_kernel(in_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress,
                      in_length: asc2.ConstExpr, tile_length: asc2.ConstExpr):
    # in_gm packs [CORE_NUM, N] flat; out_gm is the SHARED [N] destination that
    # every core atomically accumulates into at offsets=[0].
    in_gm = asc2.global_tensor(in_ptr, [in_length])
    out_gm = asc2.global_tensor(out_ptr, [tile_length])
    offset = asc2.block_idx() * tile_length
    src = asc2.copy_in(in_gm, [offset], [tile_length])
    asc2.atomic_add(src, out_gm, offsets=[0])


def atomic_add_launch(x: torch.Tensor) -> torch.Tensor:
    """Host launcher; returns out[j] = sum_i x[i, j].

    ``x`` is a ``[CORE_NUM, N]`` tensor. The output is a fresh ZERO-INITIALISED
    ``[N]`` buffer — the atomic_add accumulates into it, so it MUST start at
    zero. Do not reuse a dirty buffer.
    """
    in_flat = x.reshape(-1).contiguous()
    # MANDATORY zero-init: atomic_add accumulates into the destination.
    out = torch.zeros([N], dtype=x.dtype)
    atomic_add_kernel[CORE_NUM](in_flat, out, CORE_NUM * N, N)
    return out


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    torch.manual_seed(2026)
    x = torch.randn([CORE_NUM, N], dtype=torch.float32)
    # Reference: column-wise sum across the CORE_NUM cores.
    expected = x.reshape(CORE_NUM, N).sum(0)
    out = atomic_add_launch(x)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
    logging.info(f"[PASS] Multi-core atomic-add reduction verified for [{CORE_NUM}, {N}].")


def test_atomic_add_f32(backend: config.Backend, platform: config.Platform):
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
