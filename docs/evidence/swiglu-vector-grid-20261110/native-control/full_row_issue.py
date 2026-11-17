"""Compare serial and read-ahead issue for three cyclic, disjoint FP32 tiles."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from lower_planar_nddma import lower_fp32
from row_exact_divisor import W69


def render(source, ahead):
    if sha256(source.encode()).hexdigest() != W69:
        raise ValueError("expected the qualified W69 package")
    fn, = (node for node in ast.parse(source).body if getattr(node, "name", "") == "row_full_fp32")
    loop, = (node for node in fn.body if isinstance(node, ast.For))
    reads, stores = [], []
    for stage in range(3):
        reads.append(f"""        r{stage} = (first + {stage} * blocks) * T
        gate{stage} = asctile.copy_in(gx, [r{stage}, 0], [T, H])
        value{stage} = asctile.copy_in(gx, [r{stage}, H], [T, H])
""")
        stores.append(f"""        out{stage} = (gate{stage} / (1.0 + asctile.exp(-gate{stage}))) * value{stage}
        asctile.copy_out(out{stage}, gy, [r{stage}, 0])
""")
    operations = reads + stores if ahead else [part for pair in zip(reads, stores) for part in pair]
    header = """    block = asctile.block_idx()
    blocks = asctile.block_num()
    owned = (count + blocks - 1 - block) // blocks
    stop = block + (owned // 3) * (3 * blocks)
    for first in asctile.range(block, stop, 3 * blocks):
"""
    lines = source.splitlines(keepends=True)
    lines[loop.lineno-1:loop.body[0].lineno-1] = [
        header, *operations, "    for tile_index in asctile.range(stop, count, blocks):\n"
    ]
    return "".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("package", type=Path)
    ap.add_argument("destination", type=Path)
    ap.add_argument("mode", choices=("serial", "ahead"))
    ap.add_argument("--lower", type=Path, metavar="WORKERS")
    args = ap.parse_args()
    src = render((args.package / "swi_glu.py").read_text(), args.mode == "ahead")
    compile(src, "swi_glu.py", "exec")
    args.destination.mkdir()
    (args.destination / "swi_glu.py").write_text(src)
    (args.destination / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.destination / "_pyasc_runtime.py")
    if args.lower:
        for rows in (2046, 2048):
            lower_fp32(args.destination.resolve(), args.lower.resolve(), "row_full_fp32",
                       (rows, 2048, 2, 2048, 3), args.destination / f"rows{rows}.cpp")
