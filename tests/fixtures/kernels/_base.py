"""Shared pieces for the verification-contract fixture kernels.

These fixtures exist to prove that the harness cannot be made to report PASS by
a kernel that does not actually compute, launch, or check anything. Each one is
a *deliberately* defective kernel paired with the outcome the harness must
produce. See tests/integration/test-kernel-verification-contract.sh.

Kept dependency-light and deliberately close to the shape of a real generated
kernel, so the fixtures exercise the same code path.
"""

import torch

import asc

# The published pyasc-dev wheel exposes asctile as asc.experimental.asctile;
# older source builds expose it as a top-level module.
try:
    from asc.experimental import asctile
except ImportError:  # pragma: no cover - legacy source build
    import asctile

SIZE = 1024
TILE_SIZE = 256
CORE_NUM = 8


def make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(SIZE, dtype=torch.float32)
    y = torch.randn(SIZE, dtype=torch.float32)
    return x, y


@asctile.jit(always_compile=True)
def add_kernel(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress,
               out_ptr: asc.GlobalAddress, size: asc.ConstExpr[int],
               tile_size: asc.ConstExpr[int]):
    """Correct elementwise add: out[i] = x[i] + y[i]."""
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    num_tiles = asctile.ceildiv(size, tile_size)
    for tile_id in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num()):
        tile_offset = tile_id * tile_size
        valid = min(size - tile_offset, tile_size)
        x = asctile.copy_in(x_gm, [tile_offset], [tile_size], real_shape=[valid], pad_value=0)
        y = asctile.copy_in(y_gm, [tile_offset], [tile_size], real_shape=[valid], pad_value=0)
        out = x + y
        asctile.copy_out(out, out_gm, [tile_offset], real_shape=[valid])


@asctile.jit(always_compile=True)
def sub_kernel(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress,
               out_ptr: asc.GlobalAddress, size: asc.ConstExpr[int],
               tile_size: asc.ConstExpr[int]):
    """Deliberately wrong for an "add" kernel: out[i] = x[i] - y[i]."""
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    num_tiles = asctile.ceildiv(size, tile_size)
    for tile_id in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num()):
        tile_offset = tile_id * tile_size
        valid = min(size - tile_offset, tile_size)
        x = asctile.copy_in(x_gm, [tile_offset], [tile_size], real_shape=[valid], pad_value=0)
        y = asctile.copy_in(y_gm, [tile_offset], [tile_size], real_shape=[valid], pad_value=0)
        out = x - y
        asctile.copy_out(out, out_gm, [tile_offset], real_shape=[valid])


def launch(kernel, x, y):
    out = torch.empty_like(x)
    num_tiles = (SIZE + TILE_SIZE - 1) // TILE_SIZE
    kernel[min(CORE_NUM, num_tiles)](x, y, out, SIZE, TILE_SIZE)
    return out
