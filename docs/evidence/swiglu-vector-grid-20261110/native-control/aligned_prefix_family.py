"""Qualify a shape-derived packed prefix across aligned one-wave FP32 rows."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from textwrap import indent

from output_l2_priority import W79


def vector_source():
    # Each iteration type has a fixed mask and an affine byte address. Only the
    # prefix uses a mask shorter than a register. The whole half-row fits in UB.
    math = """RegTensor<float> gate, value, acc;
LoadAlign(gate, src + (2 * r) * {H} + offset);
LoadAlign(value, src + (2 * r + 1) * {H} + offset);
Muls(acc, gate, -1.0f, mask);
Exp(acc, acc, mask);
Adds(acc, acc, 1.0f, mask);
Div(acc, gate, acc, mask);
Mul(acc, acc, value, mask);
StoreAlign(dst + r * {H} + offset, acc, mask);"""
    return (
        "using namespace AscendC::Reg;\n"
        "auto* src = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());\n"
        "auto* dst = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());\n"
        "for (uint16_t r = 0; r < {T}; ++r) {{\n"
        "    {{\n"
        "        uint16_t offset = 0;\n"
        "        uint32_t active = {H % 64};\n"
        "        auto mask = UpdateMask<float>(active);\n"
        + indent(math, "        ") + "\n    }}\n"
        "    for (uint16_t c = 0; c < {H // 64}; ++c) {{\n"
        "        uint32_t offset = {H % 64} + c * 64;\n"
        "        auto mask = CreateMask<uint8_t>();\n"
        + indent(math, "        ") + "\n    }}\n}}"
    )


def render(source):
    if sha256(source.encode()).hexdigest() != W79:
        raise ValueError("expected the independent W79 package")
    f, = (f for f in ast.parse(source).body if getattr(f, "name", "") == "row_single_wave_fp32")
    start, = (s.lineno - 1 for s in f.body if isinstance(s, ast.If) and ast.unparse(s.test) == "W == H")
    lines = source.splitlines(keepends=True)
    default = "".join(lines[start:f.end_lineno])
    packed = f"""    if (W == H and H >= 64) and (H % 8 == 0 and H % 64 != 0):
        gx = asctile.global_tensor(x, [2 * R * H], l2_cache=False)
        data = asctile.copy_in(gx, [2 * r * H], [2 * T * H],
                               real_shape=[2 * valid * H], pad_value=0.0)
        result = asctile.inline_vf(f{vector_source()!r}, [T, W], asctile.float32, [data])
        gy = asctile.global_tensor(y, [R, H])
        asctile.copy_out(result, gy, [r, 0], real_shape=[valid, H])
    else:
"""
    lines[start:f.end_lineno] = [packed, indent(default, "    ")]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    updated = render((args.package / "swi_glu.py").read_text())
    compile(updated, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(updated)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
