"""Control the two tail fills independently for a tile-local aligned FP32 descriptor."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from exact_fp16_rows import SOURCE
from lower_planar_nddma import lower_fp32


def render(source, omitted):
    if sha256(source.encode()).hexdigest() != SOURCE or omitted not in ("gate", "value", "both"):
        raise ValueError("expected the independently generated FP32 row kernel and an operand selection")
    tree = ast.parse(source)
    kernel, = (n for n in tree.body if getattr(n, "name", "") == "row_single_wave_fp32")
    branch, = (n for n in kernel.body if isinstance(n, ast.If) and
               ast.unparse(n.test) == "W == H" and any(isinstance(a, ast.Assign) and
               ast.unparse(a.targets[0]) == "gx" for a in n.body))
    descriptor, gate, value = branch.body
    if ast.unparse(descriptor.value) != "asctile.global_tensor(x + r * (2 * H), [valid, 2 * H], l2_cache=False)":
        raise ValueError("input GM descriptor must bound the last tile")
    lines = source.splitlines(keepends=True)
    for node, name, column in [(gate, "gate", "0"), (value, "value", "H")]:
        call = node.value
        if (ast.unparse(node.targets[0]) != name or ast.unparse(call.func) != "asctile.copy_in"
                or [ast.unparse(a) for a in call.args] != ["gx", f"[0, {column}]", "[T, W]"]
                or {k.arg: ast.unparse(k.value) for k in call.keywords}
                != {"real_shape": "[valid, H]", "pad_value": "0.0"}):
            raise ValueError("unexpected aligned half-row read")
        if omitted in (name, "both"):
            i = node.lineno - 1
            old = "".join(lines[i:node.end_lineno])
            lines[i:node.end_lineno] = old.replace(", real_shape=[valid, H], pad_value=0.0", "").splitlines(keepends=True)
    result = "".join(lines)
    if result.count("pad_value=0.0") != source.count("pad_value=0.0") - (2 if omitted == "both" else 1):
        raise ValueError("expected only the selected input fills to change")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("gate", "value", "both"))
    parser.add_argument("--lower", type=Path)
    a = parser.parse_args()
    src = render((a.package / "swi_glu.py").read_text(), a.mode)
    compile(src, "swi_glu.py", "exec")
    a.output.mkdir()
    (a.output / "swi_glu.py").write_text(src)
    (a.output / "__init__.py").touch()
    copyfile(a.package / "_pyasc_runtime.py", a.output / "_pyasc_runtime.py")
    if a.lower:
        for rows, half, tile, width in ((143, 536, 4, 536), (273, 509, 6, 512)):
            lower_fp32(a.output.resolve(), a.lower.resolve(), "row_single_wave_fp32",
                       (rows, half, tile, width), a.output / f"h{half}.cpp")
