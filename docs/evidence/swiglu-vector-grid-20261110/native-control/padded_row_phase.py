"""Hold the GM row-address phase constant inside a batched FP32 DMA."""
import argparse
import ast
from hashlib import sha256
from math import gcd
from pathlib import Path
from shutil import copyfile

from lower_planar_nddma import lower_fp32
from row_exact_divisor import W69


def periods(half):
    # Input row and output row strides are in bytes. FP32 DMA sectors are 32 B.
    return 32 // gcd(32, 8 * half), 32 // gcd(32, 4 * half)


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W69:
        raise ValueError("expected the qualified W69 package")
    divisor = {"adjacent": "1", "input": "read_period", "joint": "period"}[mode]
    tree = ast.parse(source)
    fn, = (n for n in tree.body if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    signature = "".join(lines[fn.decorator_list[0].lineno - 1:fn.body[0].lineno - 1])
    signature = signature.replace("row_single_wave_fp32(", "padded_row_phase(")
    signature = signature.replace("    W: asctile.ConstExpr[int],", """    W: asctile.ConstExpr[int],
    P: asctile.ConstExpr[int],
    S: asctile.ConstExpr[int],""")
    body = """    b = asctile.block_idx()
    group_row = (b // P) * P * T
    if group_row + P * T <= R:
        slot = b % P
        first = group_row + (slot // S) * S * T
        lane = slot % S
        src = asctile.global_tensor(x + first * 2 * H, [T, 2 * S * H], l2_cache=False)
        dst = asctile.global_tensor(y + first * H, [T, S * H])
        value = asctile.copy_in(src, [0, (2 * lane + 1) * H], [T, W], real_shape=[T, H], pad_value=0.0)
        gate = asctile.copy_in(src, [0, 2 * lane * H], [T, W], real_shape=[T, H], pad_value=0.0)
        result = (gate / (1.0 + asctile.exp(-gate))) * value
        asctile.copy_out(result, dst, [0, lane * H], real_shape=[T, H])
    else:
        r = b * T
        valid = min(T, R - r)
        gx = asctile.global_tensor(x, [R, 2 * H], l2_cache=False)
        gy = asctile.global_tensor(y, [R, H])
        value_tail = asctile.copy_in(gx, [r, H], [T, W], real_shape=[valid, H], pad_value=0.0)
        gate_tail = asctile.copy_in(gx, [r, 0], [T, W], real_shape=[valid, H], pad_value=0.0)
        tail = (gate_tail / (1.0 + asctile.exp(-gate_tail))) * value_tail
        asctile.copy_out(tail, gy, [r, 0], real_shape=[valid, H])


"""
    lines.insert(fn.decorator_list[0].lineno - 1, signature + body)
    source = "".join(lines)
    launch = "            row_single_wave_fp32[grid](input, out, R, H, height, W)\n"
    if source.count(launch) != 1:
        raise ValueError("expected a single W69 one-wave FP32 launch")
    source = source.replace("import ctypes\n", "import ctypes\nfrom math import gcd\n", 1)
    return source.replace(launch, f"""            read_period = 32 // gcd(32, 8 * H)
            period = 32 // gcd(32, 4 * H)
            if W > H and period * height <= R:
                padded_row_phase[grid](input, out, R, H, height, W, period, {divisor})
            else:
    """ + launch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("adjacent", "input", "joint"))
    parser.add_argument("--lower", type=Path, metavar="WORKERS")
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text(), args.mode)
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(source)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
    if args.lower:
        ip, op = periods(509)
        stride = {"adjacent": 1, "input": ip, "joint": op}[args.mode]
        lower_fp32(args.output.resolve(), args.lower.resolve(), "padded_row_phase",
                   (273, 509, 6, 512, op, stride), args.output / "ascendc.cpp")
