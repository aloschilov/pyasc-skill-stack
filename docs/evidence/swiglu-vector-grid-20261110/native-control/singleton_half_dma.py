"""Compare two strided 2D reads with two N-D reads of the same half-row rectangles."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from bf16_slab_fusion import W72
from lower_planar_nddma import lower_fp32


def render(source, explicit):
    if sha256(source.encode()).hexdigest() != W72:
        raise ValueError("expected the unchanged W72 module")
    node, = (f for f in ast.parse(source).body if getattr(f, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    kernel = "".join(lines[node.lineno - 1:node.end_lineno])
    kernel = kernel.replace("global_tensor(x, [R, 2 * H]", "global_tensor(x, [R, 2, H]")
    start = kernel.index("    if W > H:")
    end = kernel.index("    gate_f =", start)
    bounds = ", real_shape=[valid, 1, H], pad_value=0.0" if explicit else ""

    def read(operand, plane):
        return (f"        {operand}_raw = asctile.copy_in(gx, [r, {plane}, 0], [T, 1, W]{bounds})\n"
                f"        {operand} = asctile.reshape(asctile.transpose({operand}_raw, 1, 0, 2), T, W)\n")

    reads = "    if W > H:\n" + read("value", 1) + read("gate", 0)
    reads += "    else:\n" + read("gate", 0) + read("value", 1)
    lines[node.lineno - 1:node.end_lineno] = [kernel[:start] + reads + kernel[end:]]
    return "".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("bounds", choices=("explicit", "descriptor"))
    p.add_argument("--lower", type=Path, metavar="WORKERS")
    args = p.parse_args()
    src = render((args.package / "swi_glu.py").read_text(), args.bounds == "explicit")
    compile(src, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(src)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
    if args.lower:
        for geometry in ((273, 509, 6, 512), (143, 536, 4, 536)):
            lower_fp32(args.output.resolve(), args.lower.resolve(), "row_single_wave_fp32",
                       geometry, args.output / f"h{geometry[1]}.cpp")
