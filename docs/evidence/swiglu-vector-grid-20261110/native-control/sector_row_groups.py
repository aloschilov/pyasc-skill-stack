"""Compare equal-work four-owner groups at different padded output sector boundaries."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from textwrap import dedent, indent

from packed_math_cost import W81


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W81:
        raise ValueError("expected the independent W81 operator")
    kernel, = [fn for fn in ast.parse(source).body if getattr(fn, "name", "") == "row_single_wave_fp32"]
    lines = source.splitlines(keepends=True)
    padded = kernel.body[-1].orelse
    start, end = padded[0].lineno - 1, padded[-1].end_lineno
    original = dedent("".join(lines[start:end]))
    if original.count("[T, W]") != 2 or original.count("real_shape=[valid, H]") != 3:
        raise ValueError("unexpected register-fused padded row body")

    def full(height):
        return original.replace("[T, W]", f"[{height}, W]").replace(
            "real_shape=[valid, H]", f"real_shape=[{height}, H]"
        )

    if mode == "equal":
        group = full("T")
    else:
        # Partition four consecutive T-row intervals. Both permutations
        # have two short and two long owners and end at exactly 4*T.
        edge, inner = {"aligned": ("T + 2", "T - 2"),
                       "away": ("T - 2", "T + 2")}[mode]
        group = (
            "slot = tile_index % 4\n"
            "if slot == 0 or slot == 3:\n"
            f"    r = group_row + min(slot, 1) * (4 * T - ({edge}))\n"
            + indent(full(edge), "    ")
            + "else:\n"
            f"    r = group_row + ({edge}) + (slot - 1) * ({inner})\n"
            + indent(full(inner), "    ")
        )
    # Odd FP32 half-widths repeat their output sector phase every eight rows.
    # W81 produces the matching tile by contracting its UB-fitted height
    # of eight to six. Limit the experiment to full register pitches.
    branch = ("group_row = (tile_index // 4) * (4 * T)\n"
              "eligible = (T == 6 and W % 64 == 0) and H % 2 == 1\n"
              "if eligible and group_row + 4 * T <= R:\n"
              + indent(group, "    ") + "else:\n" + indent(original, "    "))
    lines[start:end] = [indent(branch, "        ")]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("equal", "aligned", "away"))
    args = parser.parse_args()
    result = render((args.package / "swi_glu.py").read_text(), args.mode)
    compile(result, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(result)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
