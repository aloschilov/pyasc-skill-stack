# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import uuid
from abc import ABC, abstractmethod
from typing import Callable, Optional, Protocol
import base64
import hashlib
import functools
import sysconfig
from dataclasses import dataclass
import pkgutil


class HashDriver(Protocol):

    def hexdigest(self) -> str:
        ...

    def update(self, data: bytes) -> None:
        ...


@dataclass(frozen=True)
class CacheOptions:
    home_dir: str = os.getenv("PYASC_HOME", os.path.expanduser("~/"))
    dir: str = os.getenv("PYASC_CACHE_DIR", os.path.join(home_dir, ".pyasc", "cache"))
    hash_driver: Callable[[], HashDriver] = lambda: hashlib.blake2b(digest_size=16)


cache_options = CacheOptions()


def hexdigest(data: bytes) -> str:
    driver = cache_options.hash_driver()
    driver.update(data)
    return driver.hexdigest()


class CacheManager(ABC):

    @abstractmethod
    def get_file(self, filename: str) -> Optional[str]:
        pass

    @abstractmethod
    def put(self, data: bytes, filename: str, binary=True) -> str:
        pass


class FileCacheManager(CacheManager):

    def __init__(self, key: str):
        self.key = key
        self.lock_path = None

        # create cache directory if it doesn't exist
        self.cache_dir = cache_options.dir
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            raise RuntimeError("Could not create or locate cache dir")

    def has_file(self, filename: str) -> bool:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        return os.path.exists(self._make_path(filename))

    def get_file(self, filename: str) -> Optional[str]:
        if self.has_file(filename):
            return self._make_path(filename)
        else:
            return None

    def put(self, data: bytes, filename: str, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)

        if self.lock_path is None:
            raise ValueError("lock_path is None")
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        rnd_id = str(uuid.uuid4())
        # we use the PID in case a bunch of these around so we can see what PID made it
        pid = os.getpid()
        # use temp dir to be robust against program interruptions
        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        mode = "wb" if binary else "w"
        with open(temp_path, mode) as f:
            f.write(data)
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        os.replace(temp_path, filepath)
        os.removedirs(temp_dir)
        return filepath

    def _make_path(self, filename: str) -> str:
        return os.path.join(self.cache_dir, filename)


def _base32(key):
    # Assume key is a hex string.
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")


def get_cache_manager(key) -> CacheManager:
    cls = FileCacheManager
    return cls(_base32(key))


@functools.lru_cache()
def pyasc_key():
    pyasc_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hexdigest(f.read())]
    # codegen and language
    path_prefixes = [
        (os.path.join(pyasc_path, "codegen"), "asc.codegen."),
        (os.path.join(pyasc_path, "language"), "asc.language."),
    ]
    for path, prefix in path_prefixes:
        for lib in pkgutil.walk_packages([path], prefix=prefix):
            with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
                contents += [hexdigest(f.read())]

    # backend
    libpyasc_hash = cache_options.hash_driver()
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    with open(os.path.join(pyasc_path, "_C", f"libpyasc{ext}"), "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            libpyasc_hash.update(chunk)
    contents.append(libpyasc_hash.hexdigest())
    return '0.0.0_' + '_'.join(contents)


@functools.lru_cache()
def get_file_cache_key(fn_cache_key: str, cache_factors: str) -> str:
    key_str = f"{pyasc_key()}__{fn_cache_key}__{cache_factors}"
    return hexdigest(key_str.encode("utf-8"))


@functools.lru_cache()
def get_mem_cache_key(cache_factors: str) -> str:
    return hexdigest(cache_factors.encode("utf-8"))
