"""Fixed handwritten Add matching the official Ascend C [8,2048] float32 sample."""
from asc.experimental import asctile


@asctile.jit
def candidate(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress,
              out_ptr: asctile.GlobalAddress):
    x = asctile.global_tensor(x_ptr, [16384])
    y = asctile.global_tensor(y_ptr, [16384])
    out = asctile.global_tensor(out_ptr, [16384])
    offset = asctile.block_idx() * 2048
    a = asctile.copy_in(x, [offset], [2048])
    b = asctile.copy_in(y, [offset], [2048])
    asctile.copy_out(a + b, out, [offset])
