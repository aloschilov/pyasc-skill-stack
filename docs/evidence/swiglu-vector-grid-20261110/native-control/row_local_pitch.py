"""Change only the physical UB pitch of two batched FP32 operand reads."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

W65 = "1f0e5562b4a41228792ea34b61e8feaa7e295508d78986b670810d19958a0ff6"


def render(source, extra_sectors):
    if sha256(source.encode()).hexdigest() != W65 or extra_sectors not in (1, 2):
        raise ValueError("expected W65 and one or two extra 32-byte FP32 sectors")
    fn, = (n for n in ast.parse(source).body if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    start = fn.decorator_list[0].lineno - 1
    kernel = "".join(lines[start:fn.end_lineno])
    if kernel.count("[T, W]") != 4:
        raise ValueError("expected the qualified bounded 2D reads")
    kernel = kernel.replace("def row_single_wave_fp32(", "def row_extra_pitch(")
    kernel = kernel.replace("[T, W]", f"[T, W + {8 * extra_sectors}]")
    lines.insert(start, kernel + "\n\n")
    source = "".join(lines)
    # W is still the original DMA-aligned logical width for ownership and read order.
    # All four live FP32 arrays fit in this conservative one-stage UB bound.
    marker = "            row_single_wave_fp32[launch_grid](input, out, R, H, height, width)"
    if source.count(marker) != 1:
        raise ValueError("unexpected host dispatch")
    return source.replace(marker, f"""            if 16 * height * (width + {8 * extra_sectors}) <= usable_ub:
                row_extra_pitch[launch_grid](input, out, R, H, height, width)
            else:
    {marker}""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("extra_sectors", type=int, choices=(1, 2))
    args = parser.parse_args()
    result = render((args.package / "swi_glu.py").read_text(), args.extra_sectors)
    compile(result, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(result)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
