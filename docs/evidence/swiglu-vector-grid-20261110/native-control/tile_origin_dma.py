"""Factor GM descriptor origin independently on the two reads and one store."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from bf16_slab_fusion import W72
from lower_planar_nddma import lower_fp32


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W72:
        raise ValueError("expected the unchanged W72 module")
    node, = (f for f in ast.parse(source).body if getattr(f, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1:node.end_lineno])
    anchor = "    valid = min(T, R - r)\n"
    descriptors = []
    if mode in ("input", "both"):
        body = body.replace("    gx = asctile.global_tensor(x, [R, 2 * H], l2_cache=False)\n", "")
        descriptors.append("    gx = asctile.global_tensor(x + r * (2 * H), [valid, 2 * H], l2_cache=False)\n")
        body = body.replace("asctile.copy_in(gx, [r,", "asctile.copy_in(gx, [0,")
    if mode in ("output", "both"):
        body = body.replace("    gy = asctile.global_tensor(y, [R, H])\n", "")
        descriptors.append("    gy = asctile.global_tensor(y + r * H, [valid, H])\n")
        body = body.replace("gy, [r, 0]", "gy, [0, 0]")
    lines[node.lineno - 1:node.end_lineno] = [body.replace(anchor, anchor + "".join(descriptors))]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("input", "output", "both"))
    parser.add_argument("--lower", metavar="WORKERS", type=Path)
    args = parser.parse_args()
    src = render((args.package / "swi_glu.py").read_text(), args.mode)
    compile(src, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(src)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
    if args.lower:
        for geometry in ((273, 509, 6, 512), (143, 536, 4, 536)):
            lower_fp32(args.output.resolve(), args.lower.resolve(), "row_single_wave_fp32",
                       geometry, args.output / f"h{geometry[1]}.cpp")
