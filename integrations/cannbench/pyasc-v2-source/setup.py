# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
from distutils.command.clean import clean
import functools
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
from typing import List, Optional, Tuple

import pybind11
import setuptools
import setuptools_scm
from setuptools.command import bdist_wheel, build_ext, build_py, egg_info, install

DEFAULT_VERSION = "1.1.1"


@dataclass
class Package:
    url: str
    short_name: str
    full_name: str
    sym_name: Optional[str] = None


def check_env_bool(env: str) -> bool:
    return os.environ.get(env) in ["1", "true", "ON"]


def get_cache_dir() -> Path:
    user_home = os.getenv("PYASC_HOME")
    if not user_home:
        user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return Path(user_home).resolve() / ".pyasc"


@functools.lru_cache(maxsize=1)
def get_base_dir() -> Path:
    return Path(os.path.dirname(__file__)).resolve()


@functools.lru_cache(maxsize=1)
def get_build_dir() -> Path:
    build_dir = os.environ.get("PYASC_SETUP_BUILD_DIR", "").strip()
    if build_dir:
        return Path(build_dir).resolve()
    return get_base_dir() / "build"


@functools.lru_cache(maxsize=1)
def get_platform_suffix() -> str:
    platform_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    return f"{platform_name}-{sys.implementation.name}-{python_version}"


@functools.lru_cache(maxsize=1)
def get_cmake_dir() -> Path:
    cmake_dir = get_build_dir() / f"cmake.{get_platform_suffix()}"
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


def get_llvm_install_prefix():
    env = "LLVM_INSTALL_PREFIX"
    llvm_path = os.environ.get(env)
    return llvm_path if (llvm_path and llvm_path.strip()) else None


def require_tool(name: str) -> str:
    tool = shutil.which(name)
    if tool is None:
        raise RuntimeError(f"{name} is required")
    return tool


def require_tools(*names) -> Tuple[str, ...]:
    return tuple(require_tool(name) for name in names)


@functools.lru_cache(maxsize=1)
def get_requested_devtools() -> Tuple[str, ...]:
    if check_env_bool("PYASC_SETUP_DEVTOOLS"):
        return "ascir-lsp", "ascir-opt", "ascir-translate"
    return tuple()


class LocalExtension(setuptools.Extension):

    def __init__(self, modulename):
        super().__init__(modulename, sources=[])


class LocalBuildPy(build_py.build_py):

    def initialize_options(self):
        super().initialize_options()
        self.build_lib = str(get_build_dir() / f"lib.{get_platform_suffix()}")

    def run(self) -> None:
        self.run_command("build_ext")
        return super().run()


class LocalBuildExt(build_ext.build_ext):

    @staticmethod
    def populate_toolchain_args(args: List[str]) -> None:
        compiler = os.environ.get("PYASC_SETUP_COMPILER")
        linker = os.environ.get("PYASC_SETUP_LINKER")
        if check_env_bool("PYASC_SETUP_CLANG_LLD"):
            compiler = compiler or "clang++"
            linker = linker or "lld"
        if compiler:
            args.append(f"-DCMAKE_CXX_COMPILER={compiler}")
        if linker:
            args.extend((
                f"-DCMAKE_LINKER={linker}",
                f"-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld={linker}",
                f"-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld={linker}",
                f"-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld={linker}",
            ))

    @classmethod
    def get_configure_args(cls, cmake: str, ninja: str, cmake_dir: str, extdir: str) -> List[str]:
        args = [
            cmake,
            "-S",
            str(get_base_dir()),
            "-B",
            cmake_dir,
            "-G",
            "Ninja",
            "-DCMAKE_MAKE_PROGRAM=" + ninja,
            "-DCMAKE_BUILD_TYPE=" + os.environ.get("PYASC_SETUP_CONFIG", "Release"),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DPython3_INCLUDE_DIR=" + sysconfig.get_path("platinclude"),
            "-Dpybind11_INCLUDE_DIR=" + pybind11.get_include(),
            "-Dpybind11_DIR=" + pybind11.get_cmake_dir(),
        ]
        if check_env_bool("PYASC_SETUP_CCACHE"):
            args.append("-DASCIR_CCACHE=ON")
        cls.populate_toolchain_args(args)
        if check_env_bool("PYASC_SETUP_COVERAGE"):
            args.append("-DASCIR_COVERAGE=ON")
        if check_env_bool("PYASC_SETUP_ASAN"):
            args.append("-DASCIR_ASAN=ON")
        llvm_dir = get_llvm_install_prefix()
        if llvm_dir:
            args.append("-DLLVM_PREFIX_PATH=" + str(llvm_dir))
        cmake_append = os.environ.get("PYASC_SETUP_CMAKE_APPEND", "").strip()
        if cmake_append:
            args += shlex.split(cmake_append)
        return args

    def build_extension(self, ext: LocalExtension) -> None:
        cmake, ninja = require_tools("cmake", "ninja")
        os.makedirs(self.build_temp, exist_ok=True)
        cmake_dir = str(get_cmake_dir())
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        configure_args = self.get_configure_args(cmake, ninja, cmake_dir, extdir)
        subprocess.check_call(configure_args)
        targets = ["libpyasc", *get_requested_devtools()]
        if check_env_bool("PYASC_SETUP_DOCS"):
            targets.append("mlir-doc")
        build_args = [cmake, "--build", cmake_dir, "--target", *targets, "--parallel"]
        build_jobs = os.environ.get("PYASC_SETUP_JOBS")
        if build_jobs:
            build_args.append(build_jobs)
        subprocess.check_call(build_args)

    def initialize_options(self):
        super().initialize_options()
        build_dir = get_build_dir()
        self.build_lib = str(build_dir / f"lib.{get_platform_suffix()}")
        self.build_temp = str(build_dir / f"temp.{get_platform_suffix()}")

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)


class LocalBdistWheel(bdist_wheel.bdist_wheel):

    def initialize_options(self):
        super().initialize_options()
        self.bdist_dir = str(get_build_dir() / f"bdist.{get_platform_suffix()}")


class LocalClean(clean):

    def initialize_options(self):
        super().initialize_options()
        self.build_temp = get_cmake_dir()


class LocalEggInfo(egg_info.egg_info):

    def initialize_options(self):
        super().initialize_options()
        try:
            rel_dir = get_build_dir().relative_to(get_base_dir())
            if rel_dir.is_dir():
                self.egg_base = str(rel_dir)
        except ValueError:
            self.egg_base = os.curdir


class LocalInstall(install.install):

    def initialize_options(self):
        super().initialize_options()
        self.build_lib = str(get_build_dir() / f"lib.{get_platform_suffix()}")


def wheel_base_version(version: setuptools_scm.ScmVersion) -> str:
    return os.environ.get("PYASC_SETUP_VERSION", "1.1.1")


def wheel_local_version(version: setuptools_scm.ScmVersion) -> str:
    local = ""
    commit_date = version.node_date
    if commit_date:
        date_num = commit_date.strftime("%Y%m%d")
        local += f".dev{date_num}"
    local += f"+{version.node}"
    if version.dirty:
        today = version.time.strftime("%Y%m%d")
        local += f".d{today}"
    suffix = os.environ.get("PYASC_SETUP_VERSION_SUFFIX")
    if suffix:
        local += suffix
    return local


def get_project_version():
    try:
        return setuptools_scm.get_version(
            version_scheme=wheel_base_version,
            root=".",
            relative_to=__file__,
        )
    except Exception:
        return DEFAULT_VERSION


packages = [
    "asc",
    "asc/_C",
    "asc/codegen",
    "asc/common",
    "asc/language",
    "asc/language/adv",
    "asc/language/basic",
    "asc/language/core",
    "asc/language/fwk",
    "asc/lib",
    "asc/lib/host",
    "asc/lib/profiling",
    "asc/lib/runtime",
    "asc/runtime",
    "asc2",
    "asc2/language",
    "asc2/runtime",
]

extras_require = {
    "coverage": [
        "coverage[toml]",
        "gcovr",
        "pytest-cov",
    ],
    "dev": [
        "isort>=6.0.1",
        "mypy>=1.15.0",
        "ruff>=0.11.5",
        "yapf>=0.43.0",
    ],
    "docs": [
        "myst-parser==4.0.0",
        "Sphinx",
        "sphinx-gallery==0.21.0",
        "sphinx-rtd-theme==3.0.2",
        "sphinx-markdown-builder==0.6.8",
    ],
    "test": [
        "lit",
        "pytest",
        "pytest-xdist",
    ],
}


def setup() -> None:
    if sys.version_info < (3, 9):
        raise RuntimeError("Python 3.9+ is required")
    data_files = None
    devtools = get_requested_devtools()
    if devtools:
        print("packaging development tools:", *devtools)
        data_files = [("bin", [str(get_cmake_dir() / "bin" / tool) for tool in devtools])]
    cur_dir = Path(__file__).parent
    setuptools.setup(
        name=os.environ.get("PYASC_SETUP_NAME", "pyasc"),
        description="Programming language for writing efficient custom operators " \
            "with native support for Python standard specifications",
        long_description=(cur_dir / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        version=get_project_version(),
        license="CANN Open Software License Agreement Version 2.0",
        url="https://gitcode.com/cann/pyasc",
        packages=packages,
        package_dir={"": "python"},
        data_files=data_files,
        ext_modules=[
            LocalExtension("asc._C.libpyasc"),
        ],
        package_data={
            "asc/lib/runtime": ["rt_wrapper.cpp", "npu_utils.cpp", "print_utils.cpp"],
            "asc/lib/host": ["bindings/*.cpp"],
        },
        install_requires=[
            "pybind11==2.10.3",
            "numpy<2",
            "typing_extensions",
        ],
        extras_require=extras_require,
        cmdclass={
            "bdist_wheel": LocalBdistWheel,
            "build_ext": LocalBuildExt,
            "build_py": LocalBuildPy,
            "clean": LocalClean,
            "egg_info": LocalEggInfo,
            "install": LocalInstall,
        },
        zip_safe=False,
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
    )


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    setup()
