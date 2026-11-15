"""Compare one bounded FP32 GM input burst with W79's two planar reads."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from lower_planar_nddma import lower_fp32
from output_l2_priority import W79
from row_store_stream import vector_body


GEOMETRIES = ((143, 536, 4, 536), (273, 509, 6, 512))


def render(source):
    if sha256(source.encode()).hexdigest() != W79:
        raise ValueError("expected the qualified W79 source")
    tree = ast.parse(source)
    kernel, = (f for f in tree.body if getattr(f, "name", "") == "row_single_wave_fp32")
    aligned, = (s for s in kernel.body if isinstance(s, ast.If) and ast.unparse(s.test) == "W == H")
    lines = source.splitlines(keepends=True)
    # Keep the generated dispatcher, owner selection and all other specializations.
    fallback = "".join("    " + line for line in lines[aligned.lineno-1:kernel.end_lineno])
    branches = []
    for _, half, height, pitch in GEOMETRIES:
        # The last full-vector load (including unaligned-load lookahead) stays in UB.
        capacity = ((2 * height * half + 63) // 64 + 1) * 64
        body = vector_body(height, half, pitch, "padded", dense_input=half % 8 != 0)
        keyword = "elif" if branches else "if"
        branches.append(f"""    {keyword} (H == {half} and T == {height}) and W == {pitch}:
        gx_flat = asctile.global_tensor(x, [2 * R * H], l2_cache=False)
        packed = asctile.copy_in(gx_flat, [2 * r * H], [{capacity}],
                                 real_shape=[2 * valid * H], pad_value=0.0)
        result = asctile.inline_vf({body!r}, [T, W], asctile.float32, [packed])
        gy_flat = asctile.global_tensor(y, [R, H])
        asctile.copy_out(result, gy_flat, [r, 0], real_shape=[valid, H])
""")
    lines[aligned.lineno-1:kernel.end_lineno] = ["".join(branches) + "    else:\n" + fallback]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lower", type=Path, metavar="WORKERS")
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text())
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(source)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
    if args.lower:
        for geometry in GEOMETRIES:
            lower_fp32(args.output.resolve(), args.lower.resolve(), "row_single_wave_fp32",
                       geometry, args.output / f"h{geometry[1]}.cpp")
