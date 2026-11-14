"""Change only the short identity launch geometry in the generated Exp module."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil


def render(source):
    expected = "112da3565aed706120834d408b6f0c9f3f74a57320ac6273d60531a53bc5df08"
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise ValueError("unexpected generated Exp source")
    anchor = '            # FP16 identity: grid-stride, unroll 3.\n'
    start = source.index(anchor) + len(anchor)
    end = source.index('            return out', start)
    old = source[start:end]
    assert old.count('_exp_kernel[blocks]') == old.count('n, tile, 3,') == 1
    launch = """            unroll = 3
            aligned_blocks = 1 << (aivs.bit_length() - 1)
            width, tail = divmod(n, 4 * aligned_blocks)
            if (8 * aligned_blocks >= 7 * aivs and tail == 0 and width >= 1
                    and (width & (width - 1)) == 0 and width * itemsize % 32 == 0
                    and 4 * width * (8 + itemsize) <= ub):
                tile, blocks, unroll = width, aligned_blocks, 4
"""
    result = source[:start] + launch + old.replace('n, tile, 3,', 'n, tile, unroll,') + source[end:]
    ast.parse(result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = render((args.package / "exp.py").read_text())
    args.output.mkdir()
    (args.output / "exp.py").write_text(result)
    shutil.copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
