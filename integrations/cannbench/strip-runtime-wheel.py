#!/usr/bin/env python3
"""Strip the native extension inside a pyasc runtime wheel."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def exactly_one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"expected one {description}, got {paths}")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--strip", required=True, type=Path)
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    with tempfile.TemporaryDirectory(prefix="pyasc-strip-wheel-") as temporary:
        root = Path(temporary) / "root"
        output = Path(temporary) / "output"
        root.mkdir()
        output.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)

        extension = exactly_one(
            list(root.glob("asc/_C/libpyasc*.so")), "libpyasc extension"
        )
        subprocess.run(
            [str(args.strip), "--strip-unneeded", str(extension)], check=True
        )
        dist_info = exactly_one(list(root.glob("pyasc-*.dist-info")), "dist-info")
        (dist_info / "RECORD").unlink(missing_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                "--dest-dir",
                str(output),
                str(root),
            ],
            check=True,
        )
        packed = exactly_one(list(output.glob("pyasc-*.whl")), "packed wheel")
        shutil.copy2(packed, wheel)


if __name__ == "__main__":
    main()
