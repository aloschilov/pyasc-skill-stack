#!/usr/bin/env python3
"""Materialise a missing upstream pyasc v2 CMake code-generation dependency."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pybind11


def required_tool(name: str) -> str:
    result = shutil.which(name)
    if result is None:
        raise RuntimeError(f"required build tool is unavailable: {name}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    args = parser.parse_args()

    suffix = f"{sysconfig.get_platform()}-{sys.implementation.name}-{sysconfig.get_python_version()}"
    cmake_dir = args.build_dir / f"cmake.{suffix}"
    ext_dir = args.build_dir / f"lib.{suffix}" / "asc" / "_C"
    cmake_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)

    configure = [
        required_tool("cmake"),
        "-S",
        str(args.source_dir),
        "-B",
        str(cmake_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={required_tool('ninja')}",
        f"-DCMAKE_BUILD_TYPE={os.environ.get('PYASC_SETUP_CONFIG', 'Release')}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
        f"-DPython3_EXECUTABLE:FILEPATH={sys.executable}",
        f"-DPython3_INCLUDE_DIR={sysconfig.get_path('platinclude')}",
        f"-Dpybind11_INCLUDE_DIR={pybind11.get_include()}",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        f"-DLLVM_PREFIX_PATH={os.environ['LLVM_INSTALL_PREFIX']}",
    ]
    compiler = os.environ.get("PYASC_SETUP_COMPILER")
    linker = os.environ.get("PYASC_SETUP_LINKER")
    if os.environ.get("PYASC_SETUP_CLANG_LLD") in {"1", "true", "ON"}:
        compiler = compiler or "clang++"
        linker = linker or "lld"
    if compiler:
        configure.append(f"-DCMAKE_CXX_COMPILER={compiler}")
    if linker:
        configure.extend(
            (
                f"-DCMAKE_LINKER={linker}",
                f"-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld={linker}",
                f"-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld={linker}",
                f"-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld={linker}",
            )
        )
    configure.extend(shlex.split(os.environ.get("PYASC_SETUP_CMAKE_APPEND", "")))
    subprocess.run(configure, check=True)
    subprocess.run(
        [required_tool("cmake"), "--build", str(cmake_dir), "--target", "MLIRAscIncGen", "--parallel"],
        check=True,
    )


if __name__ == "__main__":
    main()
