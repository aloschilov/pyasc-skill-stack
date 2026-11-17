"""Round an aligned FP32 one-wave grid and test exact row extents per active owner."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from textwrap import indent

from row_register_chains import W65


def render(source, grid, exact):
    if sha256(source.encode()).hexdigest() != W65 or grid not in ("down", "natural", "up"):
        raise ValueError("expected W65 and a supported grid rounding direction")
    fn, = (n for n in ast.parse(source).body if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    start = fn.decorator_list[0].lineno - 1
    kernel = "".join(lines[start:fn.end_lineno])
    sig = kernel.split("    gx = ", 1)[0]
    sig = sig.replace("row_single_wave_fp32(", "row_balanced(").replace(
        "    W: asctile.ConstExpr[int],", "    N: asctile.ConstExpr[int],")
    sig = sig.replace("    T: asctile.ConstExpr[int],\n", "")
    setup = '''    owner = (asctile.block_idx() + N // 2) % N
    short_rows = R // N
    extra = R % N
    r = owner * short_rows + min(owner, extra)
'''
    # The input and output descriptors start at the owner's interval, so
    # fixed extents are valid on both arms, including the final owner.
    def body(height, dynamic=False):
        logical = "valid" if dynamic else height
        bound = ", real_shape=[valid, H], pad_value=0.0" if dynamic else ""
        output_bound = ", real_shape=[valid, H]" if dynamic else ""
        return f'''gx = asctile.global_tensor(x + 2 * r * H, [{logical}, 2 * H], l2_cache=False)
gy = asctile.global_tensor(y + r * H, [{logical}, H])
gate = asctile.copy_in(gx, [0, 0], [{height}, H]{bound})
value = asctile.copy_in(gx, [0, H], [{height}, H]{bound})
gate_f = gate.to(asctile.float32)
out = (gate_f / (1.0 + asctile.exp(-gate_f))) * value.to(asctile.float32)
asctile.copy_out(out.to(gate.dtype), gy, [0, 0]{output_bound})
'''
    if exact:
        schedule = ("    if owner < extra:\n" + indent(body("short_rows + 1"), "        ")
                    + "    else:\n" + indent(body("short_rows"), "        "))
    else:
        schedule = ("    valid = short_rows + min(1, max(0, extra - owner))\n"
                    + indent(body("(R + N - 1) // N", dynamic=True), "    "))
    lines.insert(start, sig + setup + schedule + "\n\n")
    source = "".join(lines)
    mark = "            launch_grid = blocks\n"
    call = "            row_single_wave_fp32[launch_grid](input, out, R, H, height, width)"
    if source.count(mark) != 1 or source.count(call) != 1:
        raise ValueError("expected one W65 single-wave launch")
    trial = {"down": "blocks // 8 * 8", "natural": "blocks", "up": "(blocks + 7) // 8 * 8"}[grid]
    source = source.replace(mark, f"""            trial = {trial}
            eligible = width == H and 0 < trial <= min(aivs, R) and ((R + trial - 1) // trial) * H * bytes_per_lane_one <= usable_ub
            if eligible:
                blocks = trial
{mark}""")
    return source.replace(call, """            if eligible:
                row_balanced[launch_grid](input, out, R, H, blocks)
            else:
""" + indent(call, "    "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("grid", choices=("down", "natural", "up"))
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--lower", type=Path, metavar="WORKERS",
                        help="compile the generated case-13 specialization without a device launch")
    args = parser.parse_args()
    result = render((args.package / "swi_glu.py").read_text(), args.grid, args.exact)
    compile(result, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(result)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
    if args.lower:
        from lower_planar_nddma import lower_fp32
        blocks = {"down": 32, "natural": 36, "up": 40}[args.grid]
        lower_fp32(args.output.resolve(), args.lower.resolve(), "row_balanced",
                   (143, 536, blocks), args.output / "ascendc.cpp")
