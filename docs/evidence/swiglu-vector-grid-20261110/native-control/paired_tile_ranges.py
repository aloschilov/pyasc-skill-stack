"""Distribute two contiguous tile ranges across the halves of an even one-wave grid."""
import argparse
import hashlib
from pathlib import Path

W77 = "dfc9cd891023b2aa6600018dee91984abbc8d07c8985e1c0bbf13c45cd2659ab"


def render(source):
    if hashlib.sha256(source.encode()).hexdigest() != W77:
        raise ValueError("expected the qualified W77 source")
    start = source.index("def row_single_wave_fp32(")
    point = source.index("    r = tile_index * T\n", start)
    # Pair tiles at the same offset in the lower and upper ranges. Exchange
    # alternate pairs. The last pair stays put, preserving the partial tile.
    body = """    count = (R + T - 1) // T
    if count >= 4 and count % 2 == 0:
        half_grid = count // 2
        slot = tile_index % half_grid
        if (half_grid - 1 - slot) % 2 == 1:
            tile_index = (tile_index + half_grid) % count
"""
    return source[:point] + body + source[point:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generated = render(args.base.read_text())
    compile(generated, str(args.output), "exec")
    with args.output.open("x") as stream:
        stream.write(generated)
