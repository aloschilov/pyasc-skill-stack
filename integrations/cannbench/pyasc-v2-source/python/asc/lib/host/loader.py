# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import hashlib
import os
import platform
import subprocess
import sysconfig
from pathlib import Path
import pybind11

from ...runtime.cache import get_cache_manager
from ..utils import get_ascend_path, get_cxx_compiler, get_py_include_dir


class Loader:
    module = None

    @staticmethod
    def build(so_name: str):
        so_path = os.path.join(os.path.dirname(__file__), so_name)
        asc_path = get_ascend_path()
        cc_cmd = [
            get_cxx_compiler(),
            os.path.join(os.path.dirname(__file__), "bindings/Platform.cpp"),
            os.path.join(os.path.dirname(__file__), "bindings/Enums.cpp"),
            os.path.join(os.path.dirname(__file__), "bindings/MatmulApiTiling.cpp"),
            os.path.join(os.path.dirname(__file__), "bindings/Module.cpp"),
            f"-I{get_py_include_dir()}",
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{os.path.join(asc_path, 'include')}",
            f"-I{pybind11.get_include()}",
            f"-L{os.path.join(asc_path, f'{platform.machine()}-linux/lib64')}",
            f"-L{os.path.join(asc_path, 'runtime/lib64')}",
            "-ltiling_api",
            "-lplatform",
            "-lregister",
            "-O2",
            "-o",
            so_path,
        ]
        subprocess.check_call(cc_cmd)
        return so_path

    @classmethod
    def load_library(cls):
        if cls.module is not None:
            return cls.module
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        suffix_key = ""
        version_cfg = get_ascend_path() / "version.cfg"
        if version_cfg.exists():
            suffix_key += version_cfg.read_text()
        so_name = f"libhost{suffix}"
        key = hashlib.sha256((so_name + suffix_key).encode("utf-8")).hexdigest()
        cache_manager = get_cache_manager(key)
        lib_host = cache_manager.get_file(so_name)
        if lib_host is None:
            so = Loader.build(so_name)
            with open(so, "rb") as f:
                lib_host = cache_manager.put(f.read(), so_name, binary=True)
        if not Path(lib_host).exists():
            raise FileNotFoundError(f"Library was not written to {lib_host}")
        spec = importlib.util.spec_from_file_location("libhost", lib_host)
        if spec is None:
            raise ImportError(f"Not create spec from {lib_host}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.module = mod
        return mod

    @classmethod
    def get_attr(cls, class_name: str):
        module = cls.module
        if module is None:
            module = Loader.load_library()
        return getattr(module, class_name)
