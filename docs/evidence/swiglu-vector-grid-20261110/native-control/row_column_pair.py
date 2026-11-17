"""Use idle AIVs to split aligned FP32 half-rows into two balanced column bands."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil


def render(source, mapping):
    if hashlib.sha256(source.encode()).hexdigest() != "1f0e5562b4a41228792ea34b61e8feaa7e295508d78986b670810d19958a0ff6":
        raise ValueError("expected the qualified W65 generation")
    fn, = (n for n in ast.parse(source).body if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    original = "".join(lines[fn.decorator_list[0].lineno - 1:fn.end_lineno])
    head = original[:original.index("    n = asctile.block_num()")].replace("row_single_wave_fp32", "row_column_pair")
    head = head.replace("    W: asctile.ConstExpr[int],", "    S: asctile.ConstExpr[int],")
    if mapping == "adjacent":
        owners = "    row_tile = b // 2\n    band = b % 2\n"
    elif mapping == "halves":
        owners = "    count = (R + T - 1) // T\n    band = b // count\n    row_tile = b % count\n"
    else:
        raise ValueError(mapping)
    kernel = head + owners + """    r = row_tile * T
    valid = min(T, R - r)
    col = band * S
    cols = min(S, H - col)
    gate = asctile.copy_in(gx, [r, col], [T, S], real_shape=[valid, cols], pad_value=0.0)
    value = asctile.copy_in(gx, [r, H + col], [T, S], real_shape=[valid, cols], pad_value=0.0)
    out = (gate / (1.0 + asctile.exp(-gate))) * value
    asctile.copy_out(out, gy, [r, col], real_shape=[valid, cols])


"""
    lines.insert(fn.decorator_list[0].lineno - 1, kernel)
    result = "".join(lines)
    original_call = "            launch_grid = blocks\n"
    if result.count(original_call) != 1:
        raise ValueError("unexpected single-wave host launch")
    return result.replace(original_call, """            if H % 8 == 0 and 2 * tiles <= aivs and H >= 16:
                width_pair = ((H + 15) // 16) * 8
                pair_grid = 2 * tiles
                if input.device.type == "npu":
                    stream = ctypes.c_void_p(torch.npu.current_stream(input.device).npu_stream)
                    pair_grid = (pair_grid, stream)
                row_column_pair[pair_grid](input, out, R, H, height, width_pair)
                return out
""" + original_call)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mapping", choices=["adjacent", "halves"])
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text(), args.mapping)
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "__init__.py").touch()
    (args.output / "swi_glu.py").write_text(source)
    shutil.copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
