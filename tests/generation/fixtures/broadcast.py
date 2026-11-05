"""Fixed fixture: one output row per core, with a short final tile."""
import asc
from asc.experimental import asctile


@asctile.jit(always_compile=True)
def broadcast_kernel(x: asc.GlobalAddress, out: asc.GlobalAddress):
    a = asctile.global_tensor(x, [40])
    b = asctile.global_tensor(out, [4, 40])
    data = asctile.copy_in(a, [0], [64], real_shape=[40], pad_value=0)
    row = data.reshape(1, 64)
    asctile.copy_out(row, b, [asctile.block_idx(), 0], real_shape=[1, 40])


def launch(x, out):
    broadcast_kernel[4](x, out)
