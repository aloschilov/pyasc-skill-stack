# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from asc.runtime.cache import get_cache_manager
from ..utils import get_ascend_path

PRINT_INTERFACE_NAME = "print_interface"


def build_print_utils(obj_name: str, src_file: str, src_dir: str) -> str:
    so_path = os.path.join(src_dir, f"{obj_name}.so")

    if not os.path.exists(src_dir):
        os.makedirs(src_dir, 0o750, exist_ok=True)
    cxx = os.getenv("CC")
    if cxx is None:
        cpp = shutil.which("c++")
        gxx = shutil.which("g++")
        cxx = cpp if cpp is not None else gxx
        if cxx is None:
            raise RuntimeError("Failed to find C++ compiler")
    cc_cmd = [cxx, src_file]
    # disable all warnings
    cc_cmd += ["-w"]
    # find the ascend library
    cc_cmd += [f"-L{os.path.join(get_ascend_path(), 'lib64')}", "-lascend_dump", "-lc_sec"]

    cc_cmd += ["-shared", "-fPIC", "-o", so_path]

    ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so_path
    else:
        raise RuntimeError("Failed to compile " + src_file)


class PrintInterface(object):

    def __init__(self):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        src_path = os.path.join(dir_name, "print_utils.cpp")
        src = Path(src_path).read_text()
        version_cfg_info = ""
        version_cfg = get_ascend_path() / "version.cfg"
        if version_cfg.exists():
            version_cfg_info += version_cfg.read_text()
        key = hashlib.sha256((src + version_cfg_info).encode("utf-8")).hexdigest()
        cache_manager = get_cache_manager(key)
        rt_lib = cache_manager.get_file(f"{PRINT_INTERFACE_NAME}.so")
        if rt_lib is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                so_file = build_print_utils(PRINT_INTERFACE_NAME, src_path, tmpdir)
                with open(so_file, "rb") as f:
                    rt_lib = cache_manager.put(f.read(), f"{PRINT_INTERFACE_NAME}.so", binary=True)
        self.lib: ctypes.CDLL = ctypes.CDLL(rt_lib, ctypes.RTLD_LOCAL)

    def call(self, *args):
        fn_name = "PrintWorkSpace"
        fn = getattr(self.lib, fn_name)
        fn(*args)


print_interface: Optional[PrintInterface] = None


def load_print_interface():
    global print_interface
    print_interface = PrintInterface()


def call_print_interface(*args):
    if print_interface is None:
        load_print_interface()
    print_interface.call(*args)
