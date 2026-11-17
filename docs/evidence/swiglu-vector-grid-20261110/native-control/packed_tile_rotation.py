"""Isolate the cost of half-grid rotation in a one-wave packed FP32 kernel."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile


W80 = "92e96cf5daaebb51ce9e95ecf01f4ad1e4a7bf0eaee6b04398a6da8aa11af704"


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W80:
        raise ValueError("expected the unchanged register-enabled OpenCode module")
    fn, = (f for f in ast.parse(source).body if getattr(f, "name", "") == "packed_prefix_rows")
    lines = source.splitlines(keepends=True)
    begin = fn.body[1].lineno - 1
    end = fn.body[2].end_lineno
    old = "".join(lines[begin:end])
    if old != ("    count = asctile.block_num()\n"
               "    tile = (block + count // 2) % count\n"):
        raise ValueError("unexpected packed tile ownership")
    count = {"runtime": "asctile.block_num()", "specialized": "(R + T - 1) // T"}[mode]
    # The host launches exactly ceil(R/T) blocks. Adding half a grid crosses
    # the upper bound at most once, including when the grid size is odd.
    lines[begin:end] = [
        f"    count = {count}\n",
        "    half_grid = count // 2\n",
        "    tile = block + half_grid\n",
        "    if block >= count - half_grid:\n",
        "        tile = block - (count - half_grid)\n",
    ]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("runtime", "specialized"))
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text(), args.mode)
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "__init__.py").touch()
    (args.output / "swi_glu.py").write_text(source)
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
