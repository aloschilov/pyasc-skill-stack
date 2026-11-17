"""Isolate tile order on a multi-wave, aligned, full-row FP32 kernel."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from row_exact_divisor import W69


def render(source):
    if sha256(source.encode()).hexdigest() != W69:
        raise ValueError("expected the qualified W69 package")
    fn, = (n for n in ast.parse(source).body if getattr(n, "name", "") == "row_full_fp32")
    loop, = (n for n in fn.body if isinstance(n, ast.For))
    lines = source.splitlines(keepends=True)
    # Assign the same number of full tiles to each block as the cyclic loop.
    # Consecutive addresses inside each block are the independent variable.
    lines[loop.lineno - 1:loop.body[0].lineno - 1] = ["""    b = asctile.block_idx()
    n = asctile.block_num()
    short = count // n
    extra = count % n
    begin = b * short + min(b, extra)
    length = short + min(1, max(0, extra - b))
    for tile_index in asctile.range(begin, begin + length, unroll_factor=U):
"""]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text())
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(source)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
