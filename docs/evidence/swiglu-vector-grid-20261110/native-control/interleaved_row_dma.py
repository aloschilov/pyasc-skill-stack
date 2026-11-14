"""Compare adjacent and interleaved aligned-row ownership at fixed tile height and grid."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from textwrap import indent

W65 = "1f0e5562b4a41228792ea34b61e8feaa7e295508d78986b670810d19958a0ff6"


def render(source, owners):
    if sha256(source.encode()).hexdigest() != W65 or owners not in (1, 2, 4):
        raise ValueError("expected the W65 source and one, two or four owners per group")
    func, = (n for n in ast.parse(source).body if getattr(n, "name", None) == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    start = func.lineno - 1
    original = "".join(lines[start:func.end_lineno])
    before, body = original.split("    r = tile_index * T\n", 1)
    setup = f"""    group = tile_index // {owners}
    owner = tile_index % {owners}
    first = group * {owners} * T
    if W == H and first + {owners} * T <= R:
        src = asctile.global_tensor(x + first * 2 * H, [T, {2*owners} * H], l2_cache=False)
        dst = asctile.global_tensor(y + first * H, [T, {owners} * H])
        gate = asctile.copy_in(src, [0, 2 * owner * H], [T, W])
        value = asctile.copy_in(src, [0, (2 * owner + 1) * H], [T, W])
        out = (gate / (1.0 + asctile.exp(-gate))) * value
        asctile.copy_out(out, dst, [0, owner * H])
    else:
"""
    # Full groups cover a real contiguous span, including both halves of
    # the last row. The residual group uses the original bounded descriptors.
    lines[start:func.end_lineno] = [before + setup + indent("    r = tile_index * T\n" + body, "    ")]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("owners", type=int, choices=(1, 2, 4))
    args = parser.parse_args()
    result = render((args.package / "swi_glu.py").read_text(), args.owners)
    compile(result, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(result)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
