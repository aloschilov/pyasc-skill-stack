#!/usr/bin/env python3
"""Merge the vendored pyasc runtime into the CANN Bench submission wheel."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


PYASC_COMMIT = "ac1222a48c8914d3f81297c7570d1a84f0f26778"
PYASC_SOURCE_URL = f"https://gitcode.com/compiler-team/pyasc/tree/{PYASC_COMMIT}"


def one(path: Path, pattern: str) -> Path:
    matches = list(path.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} under {path}, got {matches}")
    return matches[0]


def replace_wheel_tags(base_wheel_file: Path, runtime_wheel_file: Path) -> None:
    runtime_lines = runtime_wheel_file.read_text(encoding="utf-8").splitlines()
    runtime_tags = [line for line in runtime_lines if line.startswith("Tag: ")]
    if not runtime_tags:
        raise RuntimeError("pyasc runtime wheel has no compatibility tag")

    base_lines = base_wheel_file.read_text(encoding="utf-8").splitlines()
    kept = [
        line
        for line in base_lines
        if line.strip()
        and not line.startswith("Root-Is-Purelib: ")
        and not line.startswith("Tag: ")
    ]
    kept.extend(("Root-Is-Purelib: false", *runtime_tags))
    base_wheel_file.write_text("\n".join(kept) + "\n\n", encoding="utf-8")


def add_source_metadata(metadata_file: Path) -> None:
    metadata = metadata_file.read_text(encoding="utf-8")
    provenance = (
        f"Project-URL: Vendored pyasc v2 source, {PYASC_SOURCE_URL}\n"
        f"X-PyAsc-Source-Commit: {PYASC_COMMIT}\n"
    )
    head, separator, body = metadata.partition("\n\n")
    metadata_file.write_text(head + "\n" + provenance + separator + body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-wheel", required=True, type=Path)
    parser.add_argument("--runtime-wheel", required=True, type=Path)
    parser.add_argument("--pybind11-wheel", required=True, type=Path)
    parser.add_argument("--license", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cann-bench-wheel-") as temp:
        temp_path = Path(temp)
        base_root = temp_path / "base"
        runtime_root = temp_path / "runtime"
        pybind11_root = temp_path / "pybind11"
        base_root.mkdir()
        runtime_root.mkdir()
        pybind11_root.mkdir()
        with zipfile.ZipFile(args.base_wheel) as archive:
            archive.extractall(base_root)
        with zipfile.ZipFile(args.runtime_wheel) as archive:
            archive.extractall(runtime_root)
        with zipfile.ZipFile(args.pybind11_wheel) as archive:
            archive.extractall(pybind11_root)

        base_dist_info = one(base_root, "*.dist-info")
        runtime_dist_info = one(runtime_root, "*.dist-info")
        pybind11_dist_info = one(pybind11_root, "*.dist-info")
        for package in ("asc", "asc2"):
            source = runtime_root / package
            if not source.is_dir():
                raise RuntimeError(f"pyasc runtime wheel is missing {package}/")
            shutil.copytree(source, base_root / package)
        pybind11_package = pybind11_root / "pybind11"
        if not pybind11_package.is_dir():
            raise RuntimeError("pybind11 wheel is missing pybind11/")
        shutil.copytree(pybind11_package, base_root / "pybind11")

        license_dir = base_dist_info / "licenses"
        license_dir.mkdir(exist_ok=True)
        shutil.copy2(args.license, license_dir / "PYASC-LICENSE")
        shutil.copy2(pybind11_dist_info / "LICENSE", license_dir / "PYBIND11-LICENSE")
        replace_wheel_tags(base_dist_info / "WHEEL", runtime_dist_info / "WHEEL")
        add_source_metadata(base_dist_info / "METADATA")
        (base_dist_info / "RECORD").unlink(missing_ok=True)

        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", "--dest-dir", str(args.output_dir), str(base_root)],
            check=True,
        )


if __name__ == "__main__":
    main()
