"""Vary the two input read hints and their issue order in the generated FP32 row kernel."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil
from textwrap import indent


def render(source, cached_operand, reverse=False, value_first_parity=None):
    if hashlib.sha256(source.encode()).hexdigest() != "1f0e5562b4a41228792ea34b61e8feaa7e295508d78986b670810d19958a0ff6":
        raise ValueError("expected the qualified W65 generation")
    func, = (n for n in ast.parse(source).body if getattr(n, "name", None) == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    start, end = func.lineno - 1, func.end_lineno
    body = "".join(lines[start:end])
    if value_first_parity is not None:
        if value_first_parity not in (0, 1) or reverse:
            raise ValueError("select either a uniform order or a physical block parity")
        marker = "    if W > H:\n"
        if body.count(marker) != 1:
            raise ValueError("unexpected row read order")
        # In each branch the same two named operands feed the original math.
        # The choice is based on the physical AIV, before tile rotation.
        prefix, ordered = body.split(marker)
        value_first, gate_first = ordered.split("    else:\n")
        gate_first, math = gate_first.split("    gate_f = ", 1)
        # A tensor constructed in a dynamic branch is scoped to that branch
        # on the public wheel. Keep its downstream math and store there.
        finish = indent("    gate_f = " + math, "    ")
        body = (prefix + f"    if b % 2 == {value_first_parity}:\n"
                + value_first + finish + "    else:\n" + gate_first + finish)
    if reverse:
        order = "    if W > H:\n"
        if body.count(order) != 1:
            raise ValueError("unexpected row read order")
        body = body.replace(order, "    if W <= H:\n")
    if cached_operand == "bypass":
        lines[start:end] = [body]
        return "".join(lines)
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
    parser.add_argument("operand", choices=["bypass", "gate", "value"])
    parser.add_argument("--reverse", action="store_true", help="swap the issue order for both row layouts")
    parser.add_argument("--value-first-parity", type=int, choices=(0, 1))
    args = parser.parse_args()
    result = render((args.package / "swi_glu.py").read_text(), args.operand, args.reverse, args.value_first_parity)
    compile(result, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "__init__.py").touch()
    (args.output / "swi_glu.py").write_text(result)
    shutil.copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
