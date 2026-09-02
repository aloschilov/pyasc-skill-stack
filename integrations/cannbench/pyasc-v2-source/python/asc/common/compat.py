# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import builtins
import inspect
import sys
import typing
from typing import Any, Dict, Tuple, Type, Union

if sys.version_info >= (3, 10):

    def get_annotations(obj: Any) -> Dict[str, Any]:
        return inspect.get_annotations(obj)
else:

    def get_annotations(obj: Any) -> Dict[str, Any]:
        try:
            return typing.get_type_hints(obj)
        except Exception:
            return getattr(obj, "__annotations__", {})


if sys.version_info >= (3, 10):

    def isinstance(obj, constraint: Union[Type, Tuple[Type]]) -> bool:
        return builtins.isinstance(obj, constraint)
else:

    def isinstance(obj, constraint: Union[Type, Tuple[Type]]) -> bool:
        if builtins.isinstance(constraint, tuple):
            return any(builtins.isinstance(obj, t) for t in constraint)
        origin = typing.get_origin(constraint)
        if origin is not None:
            if origin is Union:
                return any(builtins.isinstance(obj, t) for t in typing.get_args(constraint))
            else:
                return builtins.isinstance(obj, origin)
        return builtins.isinstance(obj, constraint)


if sys.version_info >= (3, 9):

    def merge_dict(dict1: dict, dict2: dict) -> dict:
        return dict1 | dict2
else:

    def merge_dict(dict1: dict, dict2: dict) -> dict:
        return {**dict1, **dict2}