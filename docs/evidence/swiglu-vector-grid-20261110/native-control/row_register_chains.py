"""Compare FP32 register scheduling and overlapping edge masks on fixed 2D DMA."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from flat_half_vf import vector_code

W65 = "1f0e5562b4a41228792ea34b61e8feaa7e295508d78986b670810d19958a0ff6"


def render(source, chains, mask_overlap):
    if sha256(source.encode()).hexdigest() != W65:
        raise ValueError("expected the qualified W65 source")
    fn, = (f for f in ast.parse(source).body if getattr(f, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    start = next(n.lineno - 1 for n in fn.body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", "") == "gate_f")
    end = fn.end_lineno - 1
    body = []
    for i, count in enumerate((2144, 4096)):
        code = vector_code(count, False, chains=chains, mask_overlap=mask_overlap)
        branch = "if" if i == 0 else "elif"
        body += [f"    {branch} T * W == {count}:\n",
                 f"        out = asctile.inline_vf({code!r}, [T, W], asctile.float32, [gate, value])\n"]
    lines[start:end] = body + ["    else:\n"] + ["    " + line for line in lines[start:end]]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("chains", type=int, choices=(1, 2, 4))
    parser.add_argument("--mask-overlap", action="store_true")
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text(), args.chains, args.mask_overlap)
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(source)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
