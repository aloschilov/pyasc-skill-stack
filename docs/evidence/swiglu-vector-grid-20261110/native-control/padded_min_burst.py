"""Compare the minimum height for an 8-KiB logical half read to the W69 height."""
import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from row_exact_divisor import W69


def render(source):
    if sha256(source.encode()).hexdigest() != W69:
        raise ValueError("expected the qualified W69 package")
    marker = "            blocks = (R + height - 1) // height\n            if input.device.type"
    if source.count(marker) != 1:
        raise ValueError("unexpected single-wave geometry")
    return source.replace(marker, """            minimum = (8192 + 4 * H - 1) // (4 * H)
            if (W > H and minimum < height and R >= minimum
                    and minimum * W % 64 == 0 and (R + minimum - 1) // minimum <= aivs):
                height = minimum
""" + marker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text())
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(source)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
