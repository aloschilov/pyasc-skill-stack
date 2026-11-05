"""Fixed fixture, never evidence of model generation."""
import asc
from asc.experimental import asctile


@asctile.jit
def add_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, out: asc.GlobalAddress):
    a = asctile.global_tensor(x, [199])
    b = asctile.global_tensor(y, [199])
    c = asctile.global_tensor(out, [199])
    for i in asctile.range(2):
        valid = min(199-i*128, 128)
        ax = asctile.copy_in(a, [i*128], [128], real_shape=[valid], pad_value=0)
        bx = asctile.copy_in(b, [i*128], [128], real_shape=[valid], pad_value=0)
        asctile.copy_out(ax+bx, c, [i*128], real_shape=[valid])


def launch(x, y, out):
    add_kernel[1](x, y, out)
