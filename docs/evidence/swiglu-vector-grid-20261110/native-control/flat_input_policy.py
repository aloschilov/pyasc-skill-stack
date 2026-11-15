"""Replay the template's shared input-cache launch on the independent W76 module."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

EXPECTED = "ed354ee75ee14a842d691c463f935012a81af57730f8560f5e2566690ff34e71"


def render(source, template):
    if sha256(source.encode()).hexdigest() != EXPECTED:
        raise ValueError("expected the independently generated W76 source")
    helper, = (n for n in ast.parse(template).body
               if getattr(n, "name", "") == "launch_single_pass_flat")
    parts = [
        ("grid", "8192", "4"),
        ("blocks", "width", "3"),
    ]
    for grid, tile, unroll in parts:
        original = (
            "            bypass = (2 * R * H * itemsize) > 64 * 1024**2\n"
            f"            swiglu_flat[{grid}](input, out, R, H, {tile}, {unroll}, bypass)\n"
            "            return out"
        )
        if source.count(original) != 1:
            raise ValueError("expected one matching selected flat launch")
        source = source.replace(original,
            f"            return launch_single_pass_flat(input, out, R, H, {tile}, {unroll}, {grid})")
    start = source.index("def swi_glu(")
    result = source[:start] + ast.get_source_segment(template, helper) + "\n\n\n" + source[start:]
    compile(result, "swi_glu.py", "exec")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("package", "template", "destination"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    result = render((args.package / "swi_glu.py").read_text(), args.template.read_text())
    args.destination.mkdir()
    (args.destination / "__init__.py").touch()
    (args.destination / "swi_glu.py").write_text(result)
    copyfile(args.package / "_pyasc_runtime.py", args.destination / "_pyasc_runtime.py")
