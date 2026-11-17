"""Run the same short FP32 row tiles on one or two waves of productive AIVs."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from textwrap import indent

from lower_planar_nddma import lower_fp32
from row_exact_divisor import W69


def render(source, waves):
    if sha256(source.encode()).hexdigest() != W69 or waves not in (1, 2):
        raise ValueError("expected W69 and one or two waves")
    fn, = (n for n in ast.parse(source).body if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    i = next(n.lineno - 1 for n in fn.body if isinstance(n, ast.Assign) and n.targets[0].id == "b")
    j = next(n.lineno - 1 for n in fn.body if isinstance(n, ast.Assign) and n.targets[0].id == "r")
    # Preserve the original extent and aligned rotation in logical tile space.
    lines[i:fn.end_lineno] = ["""    tiles = (R + T - 1) // T
    for ordinal in asctile.range(asctile.block_idx(), tiles, asctile.block_num()):
        tile_index = (ordinal + tiles // 2) % tiles if W == H else ordinal
""", indent("".join(lines[j:fn.end_lineno]), "    ")]
    s = "".join(lines)
    mark = "            blocks = (R + height - 1) // height\n            if input.device.type"
    if s.count(mark) != 1:
        raise ValueError("unexpected host dispatch")
    return s.replace(mark, f"            tiles = (R + height - 1) // height\n"
                           f"            blocks = (tiles + {waves - 1}) // {waves}\n            if input.device.type")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("waves", type=int, choices=(1, 2))
    p.add_argument("--lower", type=Path, metavar="WORKERS")
    args = p.parse_args()
    code = render((args.package / "swi_glu.py").read_text(), args.waves)
    compile(code, "swi_glu.py", "exec")
    args.destination.mkdir()
    (args.destination / "swi_glu.py").write_text(code)
    (args.destination / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.destination / "_pyasc_runtime.py")
    if args.lower:
        for r,h,t,w in ((143,536,4,536), (273,509,6,512)):
            lower_fp32(args.destination.resolve(), args.lower.resolve(), "row_single_wave_fp32",
                       (r,h,t,w), args.destination / f"h{h}.cpp")
