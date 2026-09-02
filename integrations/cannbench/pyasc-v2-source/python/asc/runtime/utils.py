# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
from pathlib import Path
from typing import AnyStr, Callable, Union

ONE_CORE_DUMP_SIZE = 1048576
TOTAL_DUMP_SIZE = ONE_CORE_DUMP_SIZE * 75


class FileUtils:
    FILE_FLAG = os.O_WRONLY | os.O_CREAT
    FILE_MODE_640 = 0o640
    FILE_MODE_750 = 0o750

    @staticmethod
    def dump_file(dump_dir: Path, name: str, data: Union[AnyStr, Callable[[], AnyStr]]) -> None:
        if dump_dir is None:
            return
        path = dump_dir / name
        FileUtils.create_file(str(path))
        if callable(data):
            data = data()
        if isinstance(data, str):
            path.write_text(data)
        elif isinstance(data, bytes):
            path.write_bytes(data)
        else:
            raise ValueError(f"Unable to dump file: {path}!")

    @staticmethod
    def create_dir(dump_dir: Path, mode: int = FILE_MODE_750) -> None:
        if dump_dir is None or os.path.exists(dump_dir):
            return
        each_level_path = str(dump_dir).split(os.sep)
        base_path = os.sep
        for item in each_level_path:
            base_path = os.path.join(base_path, item)
            if not item or os.path.exists(base_path):
                continue
            try:
                os.makedirs(base_path, mode, exist_ok=True)
            except OSError as e:
                raise RuntimeError("Create dir {} failed!".format(dump_dir)) from e
            finally:
                pass

    @staticmethod
    def create_file(file_path: str, flag: int = FILE_FLAG, mode: int = FILE_MODE_640) -> None:
        if os.path.exists(file_path):
            return
        try:
            with os.fdopen(os.open(file_path, flag, mode), "w"):
                pass
        except OSError as e:
            raise RuntimeError("Create file {} failed!".format(file_path)) from e
        finally:
            pass
