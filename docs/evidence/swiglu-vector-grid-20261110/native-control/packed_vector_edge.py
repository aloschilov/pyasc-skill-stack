"""Move the partial FP32 vector to the end of a compact, DMA-aligned row."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from packed_tile_rotation import W80


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W80:
        raise ValueError("expected the unchanged register-enabled OpenCode module")
    fn, = (x for x in ast.parse(source).body if getattr(x, "name", "") == "packed_prefix_rows")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[fn.lineno - 1:fn.end_lineno])
    # Keep the actual full-width register arithmetic of the generated kernel.
    start = body.index("            {{\n")
    bulk = body.index("            for (uint16_t c", start)
    end = body.index("            }}\n", bulk) + len("            }}\n")
    vector_loop = body[bulk:end].replace("{H % 64} + c * 64", "c * 64")
    math_start = vector_loop.index("                RegTensor<float>")
    edge = vector_loop[math_start:]
    predicate = {
        "overlap": "                auto mask = CreateMask<uint8_t>();\n",
        "suffix": (
            "                uint32_t inactive = {64 - H % 64};\n"
            "                auto prefix = UpdateMask<float>(inactive);\n"
            "                auto all = CreateMask<uint8_t>();\n"
            "                MaskReg mask;\n"
            "                Not(mask, prefix, all);\n"
        ),
    }[mode]
    # All loads fit in the half-row. With the full mask, overlapped lanes
    # are recomputed from the same inputs and stored by the same AIV.
    block = "            {{\n                uint32_t offset = {H - 64};\n"
    lines[fn.lineno - 1:fn.end_lineno] = [body[:start] + vector_loop + block + predicate + edge + body[end:]]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("suffix", "overlap"))
    args = parser.parse_args()
    code = render((args.package / "swi_glu.py").read_text(), args.mode)
    compile(code, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "__init__.py").touch()
    (args.output / "swi_glu.py").write_text(code)
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
