"""Test a partial first output tile with an in-bounds, fixed-size input read."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from packed_math_cost import W81


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W81:
        raise ValueError("expected the independently generated W81 module")
    fn, = (f for f in ast.parse(source).body if getattr(f, "name", "") == "packed_prefix_rows")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[fn.lineno - 1:fn.end_lineno])
    start = body.index("    r = tile * T\n")
    end = body.index("    result = asctile.inline_vf", start)
    orig = body[start:end]
    size = {
        "bounded": ", real_shape=[2 * valid * H], pad_value=0.0",
        "fixed": "",
    }[mode]
    new = f"""    if R >= T:
        first = (R - 1) % T + 1
        r = max(0, tile * T - (T - first))
        valid = T
        if tile == 0:
            valid = first
        gx = asctile.global_tensor(x + 2 * r * H, [2 * T * H], l2_cache=False)
        slab = asctile.copy_in(gx, [0], [2 * T * H]{size})
    else:
"""
    # For less than one full tile, keep the generated bounds verbatim.
    new += "".join("    " + line for line in orig.splitlines(keepends=True))
    lines[fn.lineno - 1:fn.end_lineno] = [body[:start] + new + body[end:]]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("bounded", "fixed"))
    args = parser.parse_args()
    text = render((args.package / "swi_glu.py").read_text(), args.mode)
    compile(text, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(text)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
