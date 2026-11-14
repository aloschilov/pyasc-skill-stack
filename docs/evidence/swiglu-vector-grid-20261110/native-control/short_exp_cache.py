"""Vary input L2 caching on the two generated short contiguous Exp kernels."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil


def render(source):
    if hashlib.sha256(source.encode()).hexdigest() != "112da3565aed706120834d408b6f0c9f3f74a57320ac6273d60531a53bc5df08":
        raise ValueError("expected the qualified OpenCode Exp source")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    for name in ("_exp_contiguous_kernel", "_exp_contiguous_opt_kernel"):
        fn, = (n for n in tree.body if getattr(n, "name", None) == name)
        descriptor, = (n for n in fn.body if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "x_gm")
        line = descriptor.lineno - 1
        if lines[line].strip() != "x_gm = asctile.global_tensor(x_ptr, [size])":
            raise ValueError("unexpected input descriptor")
        lines[line] = lines[line].replace("[size])", "[size], l2_cache=False)")
    result = "".join(lines)
    compile(result, "exp.py", "exec")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = render((args.package / "exp.py").read_text())
    args.output.mkdir()
    (args.output / "exp.py").write_text(source)
    shutil.copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
