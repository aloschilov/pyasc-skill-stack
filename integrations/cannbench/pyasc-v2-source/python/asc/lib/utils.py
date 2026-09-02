# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import functools
import os
from pathlib import Path
import shutil
import sysconfig


@functools.lru_cache()
def get_ascend_path() -> str:
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise EnvironmentError("ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first")
    return Path(path)


@functools.lru_cache()
def get_cxx_compiler():
    cxx = os.environ.get("CXX")
    if cxx is None:
        cxx = os.environ.get("CC")
    if cxx is None:
        clangxx = shutil.which("clang++")
        gxx = shutil.which("g++")
        cxx = gxx if gxx is not None else clangxx
        if cxx is None:
            raise RuntimeError("Failed to find C++ compiler")
    return cxx


@functools.lru_cache()
def get_py_include_dir():
    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use a-scpy with system-wide python.
    if scheme == "posix_local":
        scheme = "posix_prefix"
    return sysconfig.get_paths(scheme=scheme)["include"]
