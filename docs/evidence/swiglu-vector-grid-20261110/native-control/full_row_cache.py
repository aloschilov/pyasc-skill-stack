"""Cross the input/output L2 hints on the unchanged multi-wave FP32 body."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from lower_planar_nddma import lower_fp32
from row_exact_divisor import W69


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W69:
        raise ValueError("expected the qualified W69 package")
    if mode not in ("input", "output", "both"):
        raise ValueError("unknown descriptor mode")
    kernel, = (node for node in ast.parse(source).body
               if getattr(node, "name", "") == "row_full_fp32")
    lines = source.splitlines(keepends=True)
    for variable, name in (("gx", "input"), ("gy", "output")):
        if mode in (name, "both"):
            definition, = (node for node in kernel.body if isinstance(node, ast.Assign)
                           and getattr(node.targets[0], "id", "") == variable)
            if ast.unparse(definition.value.func) != "asctile.global_tensor":
                raise ValueError("unexpected GM descriptor")
            line = definition.lineno - 1
            lines[line] = lines[line].removesuffix(")\n") + ", l2_cache=False)\n"
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("mode", choices=("input", "output", "both"))
    parser.add_argument("--lower", type=Path, metavar="WORKERS")
    args = parser.parse_args()
    code = render((args.package / "swi_glu.py").read_text(), args.mode)
    compile(code, "swi_glu.py", "exec")
    args.destination.mkdir()
    (args.destination / "swi_glu.py").write_text(code)
    (args.destination / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.destination / "_pyasc_runtime.py")
    if args.lower:
        for rows in (2046, 2048):
            lower_fp32(args.destination.resolve(), args.lower.resolve(), "row_full_fp32",
                       (rows, 2048, 2, 2048, 3), args.destination / f"rows{rows}.cpp")
