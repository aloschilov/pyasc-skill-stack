"""Bind the generated FP32 edge and complete-row launches to the input stream."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil

SOURCE_SHA = "198239e7fd7fa11edea12f0917f0eb95e4e2e43277ee7d646b54c8fef7b1dd5e"


def render(source):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError("unexpected generated SwiGLU source")
    lines = source.splitlines(keepends=True)
    for name, grid, tensor in (("single_row_edge", "aivs", "x"),
                               ("row_full_tiles", "blocks", "input")):
        key = f"{name}[{grid}]("
        index, = (i for i, line in enumerate(lines) if line.lstrip().startswith(key))
        line = lines[index]
        indent = line[:len(line) - len(line.lstrip())]
        lines[index] = (indent + f"stream = _current_stream({tensor})\n"
                        + indent + f"grid = ({grid}, stream) if stream is not None else {grid}\n"
                        + line.replace(key, f"{name}[grid](", 1))
    result = "".join(lines)
    ast.parse(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = (args.package / "swi_glu.py").read_text()
    result = render(source)
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(result)
    shutil.copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
