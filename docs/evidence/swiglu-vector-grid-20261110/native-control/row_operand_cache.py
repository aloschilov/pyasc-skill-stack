"""Vary gate and value read hints independently in the generated FP32 row kernel."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil


def render(source, cached_operand):
    if hashlib.sha256(source.encode()).hexdigest() != "1f0e5562b4a41228792ea34b61e8feaa7e295508d78986b670810d19958a0ff6":
        raise ValueError("expected the qualified W65 generation")
    func, = (n for n in ast.parse(source).body if getattr(n, "name", None) == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    start, end = func.lineno - 1, func.end_lineno
    body = "".join(lines[start:end])
    descriptor = "    gx = asctile.global_tensor(x, [R, 2 * H], l2_cache=False)\n"
    operand = f"{cached_operand} = asctile.copy_in(gx,"
    if body.count(descriptor) != 1 or body.count(operand) != 2:
        raise ValueError("unexpected row DMA definitions")
    lines[start:end] = [body.replace(descriptor, descriptor + "    cached = asctile.global_tensor(x, [R, 2 * H])\n")
                        .replace(operand, f"{cached_operand} = asctile.copy_in(cached,")]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("operand", choices=["gate", "value"])
    args = parser.parse_args()
    result = render((args.package / "swi_glu.py").read_text(), args.operand)
    compile(result, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "__init__.py").touch()
    (args.output / "swi_glu.py").write_text(result)
    shutil.copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
