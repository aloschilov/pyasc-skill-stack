"""Separate near-capacity grid choice from fixed 2D extents on aligned FP16 rows."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from lower_planar_nddma import lower_typed

SOURCE = "be93251bf40bcba2e359e73164ebd2dfcaa78710e7af6e48fdcd498e2de0a735"


def render(source, exact_grid, fixed):
    if sha256(source.encode()).hexdigest() != SOURCE:
        raise ValueError("expected the unedited qualified generation")
    fn, = (n for n in ast.parse(source).body if getattr(n, "name", "") == "bounded_rows")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[fn.decorator_list[0].lineno - 1:fn.end_lineno])
    fixed_body = body.replace("def bounded_rows(", "def exact_fp16_rows(")
    fixed_body = fixed_body.replace("        valid = min(T, R - r)\n", "")
    fixed_body = fixed_body.replace(", real_shape=[valid, H], pad_value=0.0", "")
    fixed_body = fixed_body.replace(", real_shape=[valid, H]", "")
    if "valid" in fixed_body or fixed_body.count("asctile.copy_in(") != 2:
        raise ValueError("unexpected row body")
    entry = "        bounded_rows[blocks](input, out, R, H, height, width, unroll)"
    if source.count(entry) != 1:
        raise ValueError("unexpected row dispatcher")
    kernel = "exact_fp16_rows" if fixed else "bounded_rows"
    policy = """        if dtype == torch.float16 and width == H and R % height == 0:
            tiles = R // height
            if unroll == 2 and tiles >= 3 * blocks:
"""
    if exact_grid:
        policy += """                low = (7 * blocks + 7) // 8
                for n in range(blocks, low - 1, -1):
                    if tiles % (unroll * n) == 0:
                        blocks = n
                        break
"""
    policy += f"                {kernel}[blocks](input, out, R, H, height, width, unroll)\n                return out\n"
    if fixed:
        lines[fn.end_lineno:fn.end_lineno] = ["\n\n", fixed_body]
    return "".join(lines).replace(entry, policy + entry)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("mode", choices=("grid", "fixed", "both"))
    p.add_argument("--lower", type=Path)
    a = p.parse_args()
    fixed = a.mode in ("fixed", "both")
    result = render((a.package / "swi_glu.py").read_text(), a.mode in ("grid", "both"), fixed)
    compile(result, "swi_glu.py", "exec")
    a.output.mkdir()
    (a.output / "swi_glu.py").write_text(result)
    (a.output / "__init__.py").touch()
    copyfile(a.package / "_pyasc_runtime.py", a.output / "_pyasc_runtime.py")
    if a.lower:
        lower_typed(a.output.resolve(), a.lower.resolve(),
                    "exact_fp16_rows" if fixed else "bounded_rows", ("float16", "float16"),
                    (1024, 1024, 4, 1024, 2), a.output / "fp16.cpp")
