"""Fixed add fixture for runtime qualification; no generation or NPU dispatch."""
from asc.experimental import asctile
import numpy as np


@asctile.jit(always_compile=True)
def add(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress,
        out_ptr: asctile.GlobalAddress):
    x = asctile.global_tensor(x_ptr, [128])
    y = asctile.global_tensor(y_ptr, [128])
    out = asctile.global_tensor(out_ptr, [128])
    asctile.copy_out(asctile.copy_in(x, [0], [128]) + asctile.copy_in(y, [0], [128]), out, [0])


def runtime_probe(x, y, out):
    for value in (x, y, out):
        if value.shape != (128,) or value.dtype != np.float32:
            raise ValueError('Expected float32[128]')
    add[1](x, y, out)
