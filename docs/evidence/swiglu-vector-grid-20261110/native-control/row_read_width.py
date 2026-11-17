"""Compare logical and sector-wide batched reads on padded FP32 half-rows."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from textwrap import indent

W68 = "c39508e6deeca69283a644b139ca4ae443e65902fd318709f3dd235fc389aff7"


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W68:
        raise ValueError("expected the independently generated W68 module")
    if mode not in ("exact", "gate", "value", "both"):
        raise ValueError("unknown read mode")
    fn, = (n for n in ast.parse(source).body if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[fn.lineno - 1:fn.end_lineno])
    start = body.index("    if W > H:")
    original = body[start:]
    calc = original[original.index("    gate_f = "):]
    gate = "W" if mode in ("gate", "both") else "H"
    value = "W" if mode in ("value", "both") else "H"
    # A gate may extend into its own value half. A value may extend into
    # the next gate only on interior tiles. The final tile uses exact values.
    math = indent(calc, "        ")
    common_gate = f"            gate = asctile.copy_in(gx, [r, 0], [T, W], real_shape=[valid, {gate}], pad_value=0.0)\n"
    interior = f"""    if H < W and W <= 2 * H:
        if r + T < R:
            gv = asctile.global_tensor(x + H, [R - 1, 2 * H], l2_cache=False)
            value = asctile.copy_in(gv, [r, 0], [T, W], real_shape=[valid, {value}], pad_value=0.0)
"""
    terminal = """        else:
            value = asctile.copy_in(gx, [r, H], [T, W], real_shape=[valid, H], pad_value=0.0)
"""
    lines[fn.lineno - 1:fn.end_lineno] = [body[:start] + interior + common_gate + math
        + terminal + common_gate + math + "    else:\n" + indent(original, "    ")]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("exact", "gate", "value", "both"))
    args = parser.parse_args()
    result = render((args.package / "swi_glu.py").read_text(), args.mode)
    compile(result, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(result)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
