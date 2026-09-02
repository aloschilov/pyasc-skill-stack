# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Optional, Tuple

from .constexpr import Numeric
from .dtype import DataType
from .ir_value import PlainValue, materialize_ir_value as _mat
from .utils import require_jit, global_builder


@require_jit
def inline(code: str, args: Optional[Tuple[Any]] = None, before_function: bool = False) -> None:
    args = None if args is None else [_mat(arg).to_ir() for arg in args]
    insert_point = None
    builder = global_builder.get_ir_builder()
    if before_function:
        current_function = builder.get_current_function()
        if current_function is not None:
            insert_point = builder.save_insertion_point()
            builder.set_insertion_point(current_function)
    builder.create_emitasc_VerbatimOp(code, args)
    if insert_point is not None:
        builder.restore_insertion_point(insert_point)


@require_jit
def number(value: Numeric, dtype: DataType) -> PlainValue:
    return _mat(value, dtype)
