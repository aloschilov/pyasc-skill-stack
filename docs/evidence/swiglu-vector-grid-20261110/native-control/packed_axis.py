"""Compare two public ways to extract halves from a packed FP32 row tile."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from fp32_stream_remainder import SOURCE_SHA


def render(source, mode):
    if sha256(source.encode()).hexdigest() != SOURCE_SHA or mode not in ("sum", "product"):
        raise ValueError("expected the qualified W59 module and a packed-axis mode")
    start = source.index("@asctile.jit", source.index("def single_wave_rows("))
    kernel = '''@asctile.jit(reuse_alloc=2, vf_fusion=True)
def packed_axis_rows(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                     R: asctile.ConstExpr[int], H: asctile.ConstExpr[int],
                     T: asctile.ConstExpr[int], W: asctile.ConstExpr[int],
                     rotate: asctile.ConstExpr[bool]):
    b = asctile.block_idx()
    n = asctile.block_num()
    if rotate:
        b = (b + n // 2) % n
    r = b * T
    valid = min(T, R - r)
    gx = asctile.global_tensor(x, [2 * R, H], l2_cache=False)
    gy = asctile.global_tensor(y, [R, H])
    packed = asctile.copy_in(gx, [2 * r, 0], [2 * T, W], real_shape=[2 * valid, H], pad_value=0.0)
    slab = asctile.reshape(packed, T, 2, W)
    yes = asctile.full([1, W], 1, asctile.float32)
    no = asctile.zeros([1, W], asctile.float32)
    selector = asctile.broadcast_to(asctile.reshape(asctile.concat(yes, no), 1, 2, W), T, 2, W)
    mask = selector > 0.0
'''
    if mode == "sum":
        kernel += '''    gate = asctile.reduce_sum(asctile.where(mask, slab, 0.0), 1)
    value = asctile.reduce_sum(asctile.where(mask, 0.0, slab), 1)
    out = (gate / (1.0 + asctile.exp(-gate))) * value
'''
    else:
        kernel += '''    gated = slab / (1.0 + asctile.exp(-slab))
    factors = asctile.where(mask, gated, slab)
    out = asctile.reduce_prod(factors, 1)
'''
    kernel += '''    asctile.copy_out(out, gy, [r, 0], real_shape=[valid, H])


'''
    launch = "single_wave_rows[blocks](input, out, rows, half, h, width, rotate)"
    if source.count(launch) != 1:
        raise ValueError("single-wave launch missing or ambiguous")
    return (source[:start] + kernel + source[start:]).replace(launch, launch.replace("single_wave_rows", "packed_axis_rows"))


if __name__ == "__main__":
    source = Path("base/swi_glu.py").read_text()
    for mode in ("sum", "product"):
        dest = Path(mode)
        s = render(source, mode)
        compile(s, str(dest), "exec")
        dest.mkdir()
        (dest / "swi_glu.py").write_text(s)
        (dest / "__init__.py").touch()
        copyfile("base/_pyasc_runtime.py", dest / "_pyasc_runtime.py")
