"""Compare one-wave tile heights using the byte size of the single packed read."""

import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from packed_math_cost import W81


def render(source, target_bytes):
    if sha256(source.encode()).hexdigest() != W81:
        raise ValueError("expected the independent W81 operator")
    if target_bytes not in (8192, 12288):
        raise ValueError("expected an 8 or 12 KiB packed-input target")
    anchor = "            if use_packed_prefix(rows, half, width, height, aivs, usable_ub):\n"
    if source.count(anchor) != 1:
        raise ValueError("expected a single packed register dispatch")
    return source.replace(anchor, anchor + f"""                packed_row_bytes = 2 * half * itemsize
                dma_height = ({target_bytes} + packed_row_bytes - 1) // packed_row_bytes
                single_wave_height = (rows + aivs - 1) // aivs
                height = min(height, max(dma_height, single_wave_height))
                blocks = (rows + height - 1) // height
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("target_bytes", type=int, choices=(8192, 12288))
    args = parser.parse_args()
    text = render((args.package / "swi_glu.py").read_text(), args.target_bytes)
    compile(text, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(text)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
