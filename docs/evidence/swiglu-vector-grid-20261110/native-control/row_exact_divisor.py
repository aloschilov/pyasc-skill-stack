"""Compare a neighboring exact height and fixed GM windows on padded FP32 rows."""
import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from lower_planar_nddma import lower_fp32

W69 = "65c1b287a575586d211fa708052d469d33a4489b708516b7a3cfe1fb70506fe4"


def render(source, fixed):
    if sha256(source.encode()).hexdigest() != W69:
        raise ValueError("expected the unedited W69 package")
    old = "            height = padded_single_wave_height(R, H, W, height, aivs)\n"
    if source.count(old) != 1:
        raise ValueError("expected one single-wave height selector")
    source = source.replace(old, """            ordinary_height = height
""" + old + """            if W > H and height < ordinary_height:
                # Increasing T within the fitted height also preserves one-wave coverage.
                for trial in range(height + 1, ordinary_height + 1):
                    if R % trial == 0:
                        height = trial
                        break
""")
    if fixed:
        start = source.index("    if W > H:\n", source.index("def row_single_wave_fp32("))
        # Real widths remain H even on the rebased window. W includes UB padding.
        fixed_reads = """    if W > H and R % T == 0:
        window = asctile.global_tensor(x + r * 2 * H, [T, 2 * H], l2_cache=False)
        value = asctile.copy_in(window, [0, H], [T, W], real_shape=[T, H], pad_value=0.0)
        gate = asctile.copy_in(window, [0, 0], [T, W], real_shape=[T, H], pad_value=0.0)
"""
        source = source[:start] + fixed_reads + source[start:].replace("    if W > H:", "    elif W > H:", 1)
        old_store = "    asctile.copy_out(out.to(gate.dtype), gy, [r, 0], real_shape=[valid, H])\n"
        pos = source.index(old_store, start)
        source = source[:pos] + """    if W > H and R % T == 0:
        target = asctile.global_tensor(y + r * H, [T, H])
        asctile.copy_out(out.to(gate.dtype), target, [0, 0], real_shape=[T, H])
    else:
""" + "    " + source[pos:]
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--lower", type=Path, metavar="WORKERS")
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text(), args.fixed)
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(source)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
    if args.lower:
        lower_fp32(args.output.resolve(), args.lower.resolve(), "row_single_wave_fp32",
                   (273, 509, 7, 512), args.output / "ascendc.cpp")
