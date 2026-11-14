"""Check whether the installed AscTile backend supports two UB submatrix copies."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from lower_planar_nddma import lower_fp32
from row_exact_divisor import W69


def render(source):
    if sha256(source.encode()).hexdigest() != W69:
        raise ValueError("expected the unchanged W69 package")
    fn, = (n for n in ast.parse(source).body if getattr(n, "name", "") == "row_single_wave_fp32")
    choice, = (n for n in fn.body if isinstance(n, ast.If))
    lines = source.splitlines(keepends=True)
    lines[choice.lineno-1:choice.end_lineno] = ["""    halves = asctile.global_tensor(x, [2 * R, H], l2_cache=False)
    slab = asctile.copy_in(halves, [2 * r, 0], [2 * T, W], real_shape=[2 * valid, H], pad_value=0.0)
    packed = asctile.reshape(slab, T, 2 * W)
    gate = asctile.copy(packed, [0, 0], [T, W])
    value = asctile.copy(packed, [0, W], [T, W])
"""]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "destination", "workers"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    pkg = args.destination
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    source = render((args.package / "swi_glu.py").read_text())
    compile(source, "swi_glu.py", "exec")
    (pkg / "swi_glu.py").write_text(source)
    copyfile(args.package / "_pyasc_runtime.py", pkg / "_pyasc_runtime.py")
    lower_fp32(pkg.resolve(), args.workers.resolve(), "row_single_wave_fp32",
               (143, 536, 4, 536), pkg / "aligned.cpp")
    lower_fp32(pkg.resolve(), args.workers.resolve(), "row_single_wave_fp32",
               (273, 509, 6, 512), pkg / "padded.cpp")
