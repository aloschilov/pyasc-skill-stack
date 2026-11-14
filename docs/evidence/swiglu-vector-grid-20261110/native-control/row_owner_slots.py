"""Measure the placement of a fixed set of active FP32 row owners in a larger launch."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from textwrap import indent

from row_register_chains import W65

# Each layout uses the same number of active row owners. In doubled grids
# all other logical block IDs skip both reads, the math and the store.
LAYOUTS = {
    "compact": ("b", "b < owners", 1),
    "lower": ("b", "b < owners", 2),
    "upper": ("b - owners", "b >= owners", 2),
    "even": ("b // 2", "b % 2 == 0", 2),
    "odd": ("b // 2", "b % 2 == 1", 2),
}


def render(source, layout):
    if sha256(source.encode()).hexdigest() != W65:
        raise ValueError("expected the qualified W65 generation")
    owner, active, multiplier = LAYOUTS[layout]
    fn, = (f for f in ast.parse(source).body if getattr(f, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[fn.lineno - 1:fn.end_lineno])
    head, remainder = body.split("    n = asctile.block_num()\n")
    rotation, transfers = remainder.split("    r = tile_index * T\n")
    expected = "    if W == H:\n        tile_index = (b + n // 2) % n\n    else:\n        tile_index = b\n"
    if rotation != expected:
        raise ValueError("unexpected row assignment")
    # Preserve the existing rotation and the entire memory/math body.
    assigned = rotation.replace("(b + n // 2) % n", "(owner + owners // 2) % owners").replace("tile_index = b\n", "tile_index = owner\n")
    lines[fn.lineno - 1:fn.end_lineno] = [head + "    owners = (R + T - 1) // T\n"
        + f"    if {active}:\n        owner = {owner}\n"
        + indent(assigned + "    r = tile_index * T\n" + transfers, "    ")]
    marker = "        if blocks >= tiles and dtype == torch.float32:\n"
    result = "".join(lines)
    if result.count(marker) != 1:
        raise ValueError("missing W65 one-wave branch")
    # This diagnostic is only valid when the enlarged grid fits one wave.
    return result.replace(marker, marker + f"            if {multiplier} * blocks > aivs:\n"
        + "                raise ValueError('owner placement exceeds one wave')\n"
        + f"            blocks = {multiplier} * blocks\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("layout", choices=LAYOUTS)
    args = parser.parse_args()
    result = render((args.package / "swi_glu.py").read_text(), args.layout)
    compile(result, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "__init__.py").touch()
    (args.output / "swi_glu.py").write_text(result)
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
