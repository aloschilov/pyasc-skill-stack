#!/usr/bin/env python3
"""Static integrity checks for the self-contained CANN Bench wheel."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def main() -> None:
    wheel = Path(sys.argv[1])
    commit = sys.argv[2]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        required = {
            "asc/__init__.py",
            "asc2/__init__.py",
            "cann_bench/masked_scale.py",
            "pybind11/__init__.py",
        }
        missing = required - names
        if missing:
            raise RuntimeError(f"wheel is missing required files: {sorted(missing)}")
        if not any(name.startswith("asc/_C/libpyasc") and name.endswith(".so") for name in names):
            raise RuntimeError("wheel is missing asc._C.libpyasc native extension")
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
        if f"X-PyAsc-Source-Commit: {commit}" not in metadata:
            raise RuntimeError("wheel metadata is missing pyasc source provenance")
        if not any(name.endswith(".dist-info/licenses/PYASC-LICENSE") for name in names):
            raise RuntimeError("wheel is missing the pyasc license")
        if not any(name.endswith(".dist-info/licenses/PYBIND11-LICENSE") for name in names):
            raise RuntimeError("wheel is missing the pybind11 license")


if __name__ == "__main__":
    main()
