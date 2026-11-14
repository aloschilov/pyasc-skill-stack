"""Cross caller-stream selection with exact two-wave FP16 identity geometry."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil

SOURCE_SHA = "005bf140b28a4fd7104a71a78d99e09e9eed0ad6e9298553dd06537351d0cb63"


def render(source, two_wave, caller_stream):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError("unexpected generated Exp source")
    start = "                grid = int(blocks)\n                _exp_stride[grid](\n"
    args = "                    x, out, n, tile, 3,\n"
    anchor = start + args
    if source.count(anchor) != 1:
        raise ValueError("ambiguous short identity dispatch")
    guard = """                unroll = 3
                aligned = 1 << (aivs.bit_length() - 1)
                width, tail = divmod(n, 2 * aligned)
                if (x.dtype == torch.float16 and 8 * aligned >= 7 * aivs
                        and tail == 0 and width > 0 and width & (width - 1) == 0
                        and width * itemsize % 32 == 0
                        and 2 * width * (8 + itemsize) <= ub):
                    tile, blocks, unroll = width, aligned, 2
"""
    launch = start.replace("int(blocks)", "_current_stream_grid(blocks, x)") if caller_stream else start
    body = guard + launch + args.replace("tile, 3,", "tile, unroll,") if two_wave else launch + args
    result = source.replace(anchor, body)
    ast.parse(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--two-wave", action="store_true")
    parser.add_argument("--caller-stream", action="store_true")
    options = parser.parse_args()
    source = (options.package / "exp.py").read_text()
    result = render(source, options.two_wave, options.caller_stream)
    options.output.mkdir()
    (options.output / "exp.py").write_text(result)
    shutil.copyfile(options.package / "_pyasc_runtime.py", options.output / "_pyasc_runtime.py")
