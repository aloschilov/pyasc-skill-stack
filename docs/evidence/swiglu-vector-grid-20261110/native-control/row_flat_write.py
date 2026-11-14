"""Collapse adjacent output rows into one 1D DMA on an aligned FP32 tile."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from lower_planar_nddma import lower_fp32
from row_register_chains import W65


def render(source):
    if sha256(source.encode()).hexdigest() != W65:
        raise ValueError("expected the qualified W65 package")
    fn, = (node for node in ast.parse(source).body if getattr(node, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    index = fn.end_lineno - 1
    store = "    asctile.copy_out(out.to(gate.dtype), gy, [r, 0], real_shape=[valid, H])\n"
    if lines[index] != store:
        raise ValueError("expected the original bounded output store")
    # Consecutive physical UB rows form a dense output only when W equals H.
    lines[index] = """    if W == H:
        flat_output = asctile.global_tensor(y, [R * H])
        flat_tile = asctile.reshape(out.to(gate.dtype), T * W)
        asctile.copy_out(flat_tile, flat_output, [r * H], real_shape=[valid * H])
    else:
""" + "    " + store
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lower", type=Path, metavar="WORKERS")
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text())
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(source)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
    if args.lower:
        for tag, geometry in (("aligned", (143, 536, 4, 536)),
                              ("padded", (273, 509, 8, 512))):
            lower_fp32(args.output.resolve(), args.lower.resolve(), "row_single_wave_fp32",
                       geometry, args.output / f"{tag}.cpp")
