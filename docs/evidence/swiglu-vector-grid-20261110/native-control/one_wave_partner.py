"""Keep W79 one-wave tile ownership and vary the scalar wrap arithmetic."""

import argparse
import hashlib
from pathlib import Path


W79 = "338ee63229612fc35cc287eff4a2d10bb1e9a1c47844b0d33bb0baa49da14202"


def render(source):
    if hashlib.sha256(source.encode()).hexdigest() != W79:
        raise ValueError("expected the independent W79 generation")
    # The host launches exactly count owners. With even count the corresponding
    # partner across the halves is b +/- count//2. The lower slot selects the
    # pairs whose parity equals that of half_grid, leaving the last pair fixed.
    old = """            slot = b % half_grid
            if (half_grid - 1 - slot) % 2 == 1:
                tile_index = (b + half_grid) % count
    else:
        tile_index = (b + n // 2) % n
"""
    new = """            slot = b if b < half_grid else b - half_grid
            partner = b + half_grid if b < half_grid else b - half_grid
            if (slot & 1) == (half_grid & 1):
                tile_index = partner
    else:
        shifted = b + count // 2
        tile_index = shifted - count if shifted >= count else shifted
"""
    if source.count(old) != 1:
        raise ValueError("expected both qualified owner calculations")
    return source.replace(old, new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    module = render(args.source.read_text())
    compile(module, str(args.output), "exec")
    with args.output.open("x") as stream:
        stream.write(module)
