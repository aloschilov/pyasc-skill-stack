"""Compare two public ways to extract halves from a packed FP32 row tile."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from fp32_stream_remainder import SOURCE_SHA


def render(source, mode):
    if sha256(source.encode()).hexdigest() != SOURCE_SHA or mode not in ("sum", "product", "sum_outer", "product_outer", "sum_planar", "product_planar", "product_aligned"):
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
    if mode.startswith("sum"):
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
    if mode.endswith(("_outer", "_planar", "_aligned")):
        # The public 950PR Reduce{Sum,Prod} accepts a leading reduction.
        axis_start = kernel.index("    slab = ")
        end = kernel.index("    mask = ", axis_start)
        kernel = (kernel[:axis_start] + """    rows = asctile.reshape(packed, T, 2, W)
    slab = asctile.reshape(asctile.transpose(rows, 1, 0, 2), 2, T * W)
    yes = asctile.full([1, T * W], 1, asctile.float32)
    no = asctile.zeros([1, T * W], asctile.float32)
    selector = asctile.concat(yes, no)
""" + kernel[end:])
        kernel = kernel.replace("), 1)", "), 0)").replace("reduce_prod(factors, 1)", "reduce_prod(factors, 0)")
        kernel = kernel.replace("copy_out(out, gy", "copy_out(asctile.reshape(out, T, W), gy")
    if mode.endswith(("_planar", "_aligned")):
        # Transpose a 2D tile with aligned extents. Flattening its columns
        # puts the gate/value axis first. Restore row-major order on output.
        kernel = kernel.replace("    packed = ", "    P = (T + 7) // 8 * 8\n    packed = ")
        kernel = kernel.replace("[2 * T, W]", "[2 * P, W]")
        kernel = kernel.replace("asctile.reshape(packed, T, 2, W)", "asctile.reshape(packed, P, 2 * W)")
        kernel = kernel.replace("asctile.transpose(rows, 1, 0, 2)", "asctile.transpose(rows, 1, 0)")
        kernel = kernel.replace("T * W", "P * W")
        kernel = kernel.replace("asctile.reshape(out, T, W)", "asctile.transpose(asctile.reshape(out, W, P), 1, 0)")
    if mode == "product_aligned":
        kernel = kernel.replace("    P = (T + 7) // 8 * 8", "    W = (H + 15) // 16 * 16\n    P = (T + 15) // 16 * 16")
    launch = "single_wave_rows[blocks](input, out, rows, half, h, width, rotate)"
    if source.count(launch) != 1:
        raise ValueError("single-wave launch missing or ambiguous")
    return (source[:start] + kernel + source[start:]).replace(launch, launch.replace("single_wave_rows", "packed_axis_rows"))


if __name__ == "__main__":
    source = Path("base/swi_glu.py").read_text()
    for mode in ("sum", "product", "sum_outer", "product_outer", "sum_planar", "product_planar", "product_aligned"):
        dest = Path(mode)
        s = render(source, mode)
        compile(s, str(dest), "exec")
        dest.mkdir()
        (dest / "swi_glu.py").write_text(s)
        (dest / "__init__.py").touch()
        copyfile("base/_pyasc_runtime.py", dest / "_pyasc_runtime.py")
