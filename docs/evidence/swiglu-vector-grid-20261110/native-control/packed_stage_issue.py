"""Stage public AscTile two-part schedules from the qualified packed-prefix kernel."""
import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from packed_math_cost import W81


def render(source, early):
    if sha256(source.encode()).hexdigest() != W81:
        raise ValueError("expected the qualified W81 source")
    start = source.index("def packed_prefix_rows(")
    end = source.index("\n\n", source.index("    asctile.copy_out(result", start))
    original = source[start:end]
    before, rest = original.split("    slab = ", 1)
    _, rest = rest.split("    result = ", 1)
    vf, _ = rest.split("    gy = ", 1)
    # Every permitted tile includes a full first half and at least one row in
    # its second half. Reuse the original register computation on each slab.
    before = before.replace("def packed_prefix_rows(", "def packed_prefix_stages(")
    before += "    gy = asctile.global_tensor(y, [R, H])\n    S = T // 2\n"
    before += "    second = valid - S\n"
    read_a = "    first_input = asctile.copy_in(gx, [2 * r * H], [2 * S * H])\n"
    read_b = (
        "    second_input = asctile.copy_in(gx, [2 * (r + S) * H], [2 * S * H],\n"
        "                                   real_shape=[2 * second * H], pad_value=0.0)\n"
    )
    def compute(label, row, valid_rows):
        body = vf.replace("{T}", "{S}").replace("[T, H]", "[S, H]").replace("[slab]", f"[{label}_input]")
        return (f"    {label}_result = {body}" +
                f"    asctile.copy_out({label}_result, gy, [{row}, 0], real_shape=[{valid_rows}, H])\n")

    first = compute("first", "r", "S")
    last = compute("second", "r + S", "second")
    stages = before + read_a + (read_b + first if early else first + read_b) + last
    dispatch = "                packed_prefix_rows[grid](x, out, rows, half, height)"
    assert source.count(dispatch) == 1
    select = (
        "                tail = rows % height\n"
        "                if height % 2 == 0 and (tail == 0 or tail > height // 2):\n"
        "                    packed_prefix_stages[grid](x, out, rows, half, height)\n"
        "                    return out\n"
    )
    return (source[:end] + "\n\n\n@asctile.jit(reuse_alloc=2, vf_fusion=True)\n" +
            stages + source[end:]).replace(dispatch, select + dispatch)


def merge_outputs(source, single_vf):
    """Use the early-read schedule and one bounded whole-tile output."""
    start = source.index("    first_result = asctile.inline_vf")
    end = source.index("\n", source.index("    asctile.copy_out(second_result", start))
    first, second = source[start:end].split("    second_result = ")
    first, first_store = first.split("    asctile.copy_out(first_result")
    second, second_store = second.split("    asctile.copy_out(second_result")
    assert "real_shape=[S, H]" in first_store and "real_shape=[second, H]" in second_store
    if single_vf:
        # Write the second half directly into the same output UB tile. Distinct
        # scopes let each half retain its original constant vector loop bounds.
        def scoped(call, operand, destination):
            begin = call.index("        auto* src =")
            body, _ = call[begin:].split('    """', 1)
            body = body.replace("$1.GetPhyAddr()", f"${operand}.GetPhyAddr()")
            body = body.replace("$0.GetPhyAddr());", "$0.GetPhyAddr())" + destination + ";")
            return "        {{\n" + body + "        }}\n"

        merged = ('    result = asctile.inline_vf(f"""\n        using namespace AscendC::Reg;\n'
                  + scoped(first, 1, "") + scoped(second, 2, " + {S * H}")
                  + '    """, [T, H], asctile.float32, [first_input, second_input])\n')
    else:
        merged = first + "    second_result = " + second
        merged += "    result = asctile.concat(first_result, second_result)\n"
    merged += "    asctile.copy_out(result, gy, [r, 0], real_shape=[valid, H])"
    return source[:start] + merged + source[end:]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--merge", action="store_true", help="compare two stores, concatenation and one register result")
    args = p.parse_args()
    original = (args.package / "swi_glu.py").read_text()
    modes = ("early", "concat", "joined") if args.merge else ("serial", "early")
    for mode in modes:
        dest = args.output / mode
        dest.mkdir(parents=True)
        source = render(original, early=mode != "serial")
        if mode in ("concat", "joined"):
            source = merge_outputs(source, single_vf=mode == "joined")
        compile(source, str(dest / "swi_glu.py"), "exec")
        (dest / "swi_glu.py").write_text(source)
        (dest / "__init__.py").touch()
        copyfile(args.package / "_pyasc_runtime.py", dest / "_pyasc_runtime.py")
