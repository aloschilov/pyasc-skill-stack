"""Compare a single input slab plus an immutable FP32 gather table at a fixed grid."""
import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from row_exact_divisor import W69


KERNEL = """@asctile.jit(reuse_alloc=2)
def indexed_row_fp32(
    x: asctile.GlobalAddress,
    y: asctile.GlobalAddress,
    index: asctile.GlobalAddress,
    R: asctile.ConstExpr[int],
    H: asctile.ConstExpr[int],
    T: asctile.ConstExpr[int],
    W: asctile.ConstExpr[int],
):
    halves = asctile.global_tensor(x, [2 * R, H], l2_cache=False)
    output = asctile.global_tensor(y, [R, H])
    table = asctile.global_tensor(index, [T * W])
    b = asctile.block_idx()
    n = asctile.block_num()
    tile_index = (b + n // 2) % n if W == H else b
    first = tile_index * T
    valid = min(T, R - first)
    slab = asctile.copy_in(halves, [2 * first, 0], [2 * T, W], real_shape=[2 * valid, H], pad_value=0.0)
    flat = asctile.reshape(slab, 2 * T * W)
    gate_index = asctile.copy_in(table, [0], [T * W])
    gate = asctile.reindex(flat, gate_index)
    value = asctile.reindex(flat, gate_index + W)
    result = (gate / (1.0 + asctile.exp(-gate))) * value
    asctile.copy_out(asctile.reshape(result, T, W), output, [first, 0], real_shape=[valid, H])


def fp32_half_indices(width, height, device):
    stream = torch.npu.current_stream(device).npu_stream if device.type == "npu" else None
    key = ("fp32_half_indices", width, height, device, stream)
    if key not in _INDEX_CACHE:
        q = torch.arange(height * width, dtype=torch.int32, device=device)
        _INDEX_CACHE[key] = q + (q // width) * width
    return _INDEX_CACHE[key]


"""


def render(source):
    if sha256(source.encode()).hexdigest() != W69:
        raise ValueError("expected the independently generated W69 package")
    anchor = "@asctile.jit(reuse_alloc=2)\ndef row_single_wave_fp32("
    call = "            row_single_wave_fp32[grid](input, out, R, H, height, W)"
    if source.count(anchor) != 1 or source.count(call) != 1:
        raise ValueError("unexpected single-wave entry point")
    return source.replace(anchor, KERNEL + anchor).replace(call, """            if 40 * height * W <= usable_ub:
                index = fp32_half_indices(W, height, input.device)
                indexed_row_fp32[grid](input, out, index, R, H, height, W)
            else:
                row_single_wave_fp32[grid](input, out, R, H, height, W)""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    code = render((args.package / "swi_glu.py").read_text())
    compile(code, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(code)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
