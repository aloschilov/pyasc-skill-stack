"""Fixed fixture for an exact int32 sum with padding."""
import asc
from asc.experimental import asctile


@asctile.jit(always_compile=True)
def sum_kernel(x: asc.GlobalAddress, out: asc.GlobalAddress):
    a = asctile.global_tensor(x, [1, 61])
    b = asctile.global_tensor(out, [1])
    data = asctile.copy_in(a, [0, 0], [1, 64], real_shape=[1, 61], pad_value=0)
    reduced = asctile.reduce_sum(data, 1)
    asctile.copy_out(reduced, b, [0])


def launch(x, out):
    sum_kernel[1](x, out)
