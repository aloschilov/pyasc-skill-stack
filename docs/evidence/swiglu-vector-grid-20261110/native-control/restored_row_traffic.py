"""Build private diagnostics for the qualified FP32 row kernel on restored inputs."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil

from row_math_cost import CONTROL_ORACLES, DIRECT, EXPRESSIONS


def build(package, task, target, mode):
    source = (package / "swi_glu.py").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != "1f0e5562b4a41228792ea34b61e8feaa7e295508d78986b670810d19958a0ff6":
        raise ValueError("expected the qualified W65 generation")
    func, = (n for n in ast.parse(source).body if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[func.lineno - 1:func.end_lineno])
    if mode == "sector":
        # Retain all rows, the same tile assignment and bounded final block.
        body = body[:body.index("    if W > H:")] + (
            "    gate = asctile.copy_in(gx, [r, 0], [T, 8], real_shape=[valid, 8])\n"
            "    asctile.copy_out(gate, gy, [r, 0], real_shape=[valid, 8])\n"
        )
        body = body.replace("global_tensor(y, [R, H])", "global_tensor(y, [R, 8])")
        expr = "gate.reshape(rows, -1)[:, :8].contiguous()"
    else:
        body = body.replace(DIRECT, EXPRESSIONS[mode])
        if mode == "gate_copy":
            # Remove the unused read explicitly, independent of compiler DCE.
            begin, end = body.index("    if W > H:"), body.index("    gate_f =")
            gate = next(line for line in body[begin:end].splitlines() if "gate = asctile.copy_in" in line)
            body = body[:begin] + "    " + gate.strip() + "\n" + body[end:]
        expr = CONTROL_ORACLES[mode]
    lines[func.lineno - 1:func.end_lineno] = [body]
    source = "".join(lines)
    allocation = "    out = input.new_empty(out_shape)"
    if source.count(allocation) != 1:
        raise ValueError("expected one output allocation")
    source = source.replace(allocation, (
        "    if input.dtype != torch.float32 or (tuple(shape), d) not in "
        "(((11, 13, 16, 67), 2), ((3, 7, 13, 1018), 3)):\n"
        '        raise ValueError("private row traffic probe supports only the two audited shapes")\n'
    ) + ("    out = input.new_empty((*shape[:d], 8))" if mode == "sector" else allocation))
    compile(source, "swi_glu.py", "exec")
    private = target / "tasks/level1/swi_glu"
    private.mkdir(parents=True)
    for filename in ("cases.csv", "cases.yaml", "proto.yaml"):
        shutil.copyfile(task / filename, private / filename)
    (private / "golden.py").write_text(
        "from math import prod\n\n"
        "def swi_glu(input, dim=-1):\n"
        "    dim %= input.ndim\n"
        "    gate, value = input.chunk(2, dim=dim)\n"
        "    rows = prod(input.shape[:dim])\n"
        f"    return {expr}" + (".reshape(*input.shape[:dim], 8)" if mode == "sector" else "") + "\n"
    )
    probe = target / "probe"
    probe.mkdir()
    (probe / "__init__.py").touch()
    (probe / "swi_glu.py").write_text(source)
    shutil.copyfile(package / "_pyasc_runtime.py", probe / "_pyasc_runtime.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("package", "task", "target"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("mode", choices=["sector", "gate_copy", "multiply"])
    args = parser.parse_args()
    build(args.package, args.task, args.target, args.mode)
