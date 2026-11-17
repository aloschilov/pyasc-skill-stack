"""Measure the gate arithmetic on the two physical layouts emitted by W81."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from row_math_cost import CONTROL_ORACLES, DIRECT, EXPRESSIONS
from row_order_neighbors import prepare

W81 = "b842b4c5bc708b1a5ea2b73877fc15816b609283318701fc07360051d6c53e2a"


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W81:
        raise ValueError("expected the independent W81 SwiGLU source")
    lines = source.splitlines(keepends=True)
    for fn in reversed(ast.parse(source).body):
        name = getattr(fn, "name", "")
        if name not in ("row_single_wave_fp32", "packed_prefix_rows"):
            continue
        part = "".join(lines[fn.lineno - 1:fn.end_lineno])
        if name == "row_single_wave_fp32":
            if mode == "product":
                assert part.count(DIRECT) == 2
                part = part.replace(DIRECT, EXPRESSIONS["multiply"])
        else:
            start = "                Muls(acc, gate, -1.0f, mask);\n"
            end = "                Mul(acc, acc, value, mask);\n"
            division = "                Div(acc, gate, acc, mask);\n"
            assert part.count(start) == part.count(end) == part.count(division) == 2
            if mode == "product":
                original = start + ("                Exp(acc, acc, mask);\n"
                                    "                Adds(acc, acc, 1.0f, mask);\n") + division + end
                assert part.count(original) == 2
                part = part.replace(original, "                Mul(acc, gate, value, mask);\n")
            elif mode == "late":
                # Preserve the two distinct affine addresses of the peeled loops.
                for off in ("", " + offset"):
                    read = "                LoadAlign(value, src + (2 * row + 1) * {H}" + off + ");\n"
                    assert part.count(read) == 1
                    a = part.index(read)
                    b = part.index(division, a) + len(division)
                    part = part[:a] + part[a + len(read):b] + read + part[b:]
            else:
                raise ValueError(mode)
        lines[fn.lineno - 1:fn.end_lineno] = [part]
    return "".join(lines)


def stage(package, task, output):
    source = (package / "swi_glu.py").read_text()
    for mode in ("product", "late"):
        candidate = render(source, mode)
        compile(candidate, "swi_glu.py", "exec")
        root = output / mode
        root.mkdir()
        (root / "swi_glu.py").write_text(candidate)
        (root / "__init__.py").touch()
        copyfile(package / "_pyasc_runtime.py", root / "_pyasc_runtime.py")
    prepare(task, output / "finite", "aligned-width")
    # Give the reduced operation its own oracle and private task tree. Leave
    # the public task and the full SwiGLU golden untouched.
    for data, src in [("public", task), ("finite", output / "finite/h536/tasks/level1/swi_glu")]:
        dst = output / "product-input" / data / "tasks/level1/swi_glu"
        dst.mkdir(parents=True)
        for item in ("cases.csv", "cases.yaml", "proto.yaml"):
            copyfile(src / item, dst / item)
        (dst / "golden.py").write_text(
            "def swi_glu(input, dim=-1):\n"
            "    gate, value = input.chunk(2, dim=dim)\n"
            f"    return {CONTROL_ORACLES['multiply']}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("task", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.package, args.task, args.output)
