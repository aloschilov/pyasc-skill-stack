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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    original = (args.package / "swi_glu.py").read_text()
    for mode in ("serial", "early"):
        dest = args.output / mode
        dest.mkdir(parents=True)
        source = render(original, early=mode == "early")
        compile(source, str(dest / "swi_glu.py"), "exec")
        (dest / "swi_glu.py").write_text(source)
        (dest / "__init__.py").touch()
        copyfile(args.package / "_pyasc_runtime.py", dest / "_pyasc_runtime.py")
