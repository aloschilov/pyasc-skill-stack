#!/usr/bin/env python3
"""Merge the exact current-v2 runtime into a CANNBench wheel."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


PYASC_COMMIT = "0a631f70968c3cb7c33ce45330a85768dd5a6f06"


def one(path: Path, pattern: str) -> Path:
    matches = list(path.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} under {path}, got {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-wheel", required=True, type=Path)
    parser.add_argument("--runtime-wheel", required=True, type=Path)
    parser.add_argument("--pybind11-wheel", required=True, type=Path)
    parser.add_argument("--license", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cannbench-gelu-v2-") as temp:
        root = Path(temp)
        base, runtime, pybind = root / "base", root / "runtime", root / "pybind"
        for directory, wheel in (
            (base, args.base_wheel),
            (runtime, args.runtime_wheel),
            (pybind, args.pybind11_wheel),
        ):
            directory.mkdir()
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(directory)
        base_info = one(base, "*.dist-info")
        runtime_info = one(runtime, "*.dist-info")
        pybind_info = one(pybind, "*.dist-info")
        for package in ("asc", "asctile"):
            shutil.copytree(runtime / package, base / package)
        shutil.copytree(pybind / "pybind11", base / "pybind11")
        licenses = base_info / "licenses"
        licenses.mkdir(exist_ok=True)
        shutil.copy2(args.license, licenses / "PYASC-LICENSE")
        shutil.copy2(pybind_info / "LICENSE", licenses / "PYBIND11-LICENSE")
        tags = [
            line for line in (runtime_info / "WHEEL").read_text().splitlines()
            if line.startswith("Tag: ")
        ]
        wheel_lines = [
            line for line in (base_info / "WHEEL").read_text().splitlines()
            if line.strip()
            and not line.startswith("Tag: ")
            and not line.startswith("Root-Is-Purelib: ")
        ]
        (base_info / "WHEEL").write_text(
            "\n".join(wheel_lines + ["Root-Is-Purelib: false", *tags]) + "\n"
        )
        metadata = (base_info / "METADATA").read_text()
        head, separator, body = metadata.partition("\n\n")
        provenance = (
            "\nProject-URL: Vendored pyasc v2 source, "
            f"https://gitcode.com/compiler-team/pyasc/tree/{PYASC_COMMIT}"
            f"\nX-PyAsc-Source-Commit: {PYASC_COMMIT}"
        )
        (base_info / "METADATA").write_text(
            head + provenance + separator + body
        )
        (base_info / "RECORD").unlink(missing_ok=True)
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", "--dest-dir",
             str(args.output_dir), str(base)],
            check=True,
        )


if __name__ == "__main__":
    main()
