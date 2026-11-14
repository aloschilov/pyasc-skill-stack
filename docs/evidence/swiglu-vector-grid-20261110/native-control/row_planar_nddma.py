"""Compile the [row, half, column] load directly into two adjacent UB planes."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from flat_half_vf import vector_code
from row_register_chains import W65


def render(source, height, pitch):
    if sha256(source.encode()).hexdigest() != W65:
        raise ValueError("expected the qualified W65 source")
    extent = height * pitch
    # The two local planes occupy consecutive extents in one input tile.
    code = vector_code(extent, True, mask_overlap=True)
    ptr = "reinterpret_cast<__ubuf__ float*>($2.GetPhyAddr())"
    if code.count(ptr) != 1:
        raise ValueError("expected a separate value pointer in the register control")
    code = code.replace(ptr, f"g + {extent}")
    fn, = (f for f in ast.parse(source).body if getattr(f, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    start = fn.decorator_list[0].lineno - 1
    kernel = "".join(lines[start:fn.end_lineno])
    kernel = kernel.replace("def row_single_wave_fp32(", "def planar_nddma(")
    kernel = kernel.replace("[R, 2 * H]", "[R, 2, H]")
    i = kernel.index("    if W > H:")
    j = kernel.index("    asctile.copy_out", i)
    kernel = kernel[:i] + f"""    slab = asctile.copy_in(gx, [r, 0, 0], [T, 2, W], real_shape=[valid, 2, H], pad_value=0.0)
    planes = asctile.transpose(slab, 1, 0, 2)
    out = asctile.inline_vf({code!r}, [T, W], asctile.float32, [planes])
""" + kernel[j:].replace("out.to(gate.dtype)", "out")
    lines.insert(start, kernel + "\n\n")
    marker = "            row_single_wave_fp32[launch_grid](input, out, R, H, height, width)"
    result = "".join(lines)
    if result.count(marker) != 1:
        raise ValueError("missing the single-wave dispatch")
    return result.replace(marker, f"""            if height == {height} and width == {pitch}:
                planar_nddma[launch_grid](input, out, R, H, height, width)
            else:
    {marker}""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("height", type=int)
    parser.add_argument("pitch", type=int)
    args = parser.parse_args()
    if args.height <= 0 or args.pitch <= 0 or args.pitch % 8:
        parser.error("expected positive dimensions and a 32-byte-aligned FP32 row")
    source = render((args.package / "swi_glu.py").read_text(), args.height, args.pitch)
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(source)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
