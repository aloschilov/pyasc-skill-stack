"""Compare two scalar implementations of the padded FP32 single-wave tile exchange."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from packed_tile_rotation import W80


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W80:
        raise ValueError("expected the unchanged register-enabled OpenCode module")
    fn, = (x for x in ast.parse(source).body if getattr(x, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    original = "".join(lines[fn.lineno - 1:fn.end_lineno])
    old = """            slot = b % half_grid
            if (half_grid - 1 - slot) % 2 == 1:
                tile_index = (b + half_grid) % count
"""
    # On even grids the partner differs by exactly half_grid. Parity of
    # the within-half slot selects the same pairs as the generated kernel.
    slot = """            slot = b
            if b >= half_grid:
                slot = b - half_grid
"""
    if mode == "slot":
        new = slot + """            if (half_grid - 1 - slot) % 2 == 1:
                tile_index = (b + half_grid) % count
"""
    elif mode == "partner":
        new = """            partner = b + half_grid
            slot = b
            if b >= half_grid:
                slot = b - half_grid
                partner = b - half_grid
            if slot % 2 == half_grid % 2:
                tile_index = partner
"""
    else:
        raise ValueError(mode)
    if original.count(old) != 1:
        raise ValueError("unexpected padded single-wave scheduling code")
    lines[fn.lineno - 1:fn.end_lineno] = [original.replace(old, new)]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("slot", "partner"))
    args = parser.parse_args()
    code = render((args.package / "swi_glu.py").read_text(), args.mode)
    compile(code, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "__init__.py").touch()
    (args.output / "swi_glu.py").write_text(code)
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
