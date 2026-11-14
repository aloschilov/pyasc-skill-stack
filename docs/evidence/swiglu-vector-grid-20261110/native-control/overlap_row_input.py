"""Test fixed-height reads on a partial first output tile with disjoint writes."""
import argparse
import hashlib
from pathlib import Path

from row_math_cost import W62
from wave_rotation import prepare, run


def render(source, mode):
    if hashlib.sha256(source.encode()).hexdigest() != W62:
        raise ValueError("expected the unedited W62 module")
    if mode == "base":
        return source
    start = source.index("def row_single_wave_fp32(")
    stop = source.index("\n\n@asctile.jit", start)
    body = source[start:stop]
    geometry = """    r = tile_index * T
    valid = min(T, R - r)"""
    # Keep the same physical AIV for the short tile. Only its logical output
    # moves to the front. Read at most T rows from the new tile base.
    new_geometry = """    tail = R - ((R - 1) // T) * T
    last = tile_index == (R - 1) // T
    r = 0 if last else tail + tile_index * T
    valid = tail if last else T
    tile_in = asctile.global_tensor(x + r * (2 * H), [T, 2 * H], l2_cache=False)"""
    if geometry not in body or body.count("gx, [r,") != 4:
        raise ValueError("unexpected W62 row body")
    if mode not in ("bounded", "overlap"):
        raise ValueError(mode)
    body = body.replace(geometry, new_geometry).replace("gx, [r,", "tile_in, [0,")
    input_shape = "[T, H]" if mode == "overlap" else "[valid, H]"
    body = body.replace("real_shape=[valid, H], pad_value", f"real_shape={input_shape}, pad_value")
    return source[:start] + body + source[stop:]


def audit(rows, height, aligned):
    blocks = (rows + height - 1) // height
    if rows < height:
        raise ValueError("fixed-height reads require at least one full input tile")
    owners = []
    excess = 0
    for b in range(blocks):
        tile = (b + blocks // 2) % blocks if aligned else b
        tail = rows - (blocks - 1) * height
        first, valid = (0, tail) if tile == blocks-1 else (tail + tile*height, height)
        assert 0 <= first and first + height <= rows
        excess += height - valid
        owners.extend(range(first, first + valid))
    assert sorted(owners) == list(range(rows)) and excess == blocks*height-rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "task", "worker"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--case", type=int, choices=(11, 13), default=13)
    args = parser.parse_args()
    modes = ("base", "bounded", "overlap")
    # Keep row count and half-width fixed while changing values and rank factorization.
    neighbor = [273, 1, 1, 1018] if args.case == 11 else [143, 1, 16, 67]
    shapes = {"public": None, "finite": neighbor}
    # Reuse the fixture staging and seeded, fresh-process experiment runner.
    for rows in (139, 140, 143, 144):
        audit(rows, 4, True)
    for rows in (272, 273, 274, 279):
        audit(rows, 8, False)
    prepare(args.base.resolve(), args.task.resolve(), shapes, modes, renderer=render, case=args.case)
    raise SystemExit(run(args.worker.resolve(), shapes, modes, case=args.case))
