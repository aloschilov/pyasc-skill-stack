# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload

from ..core.dtype import DataType
from ..core.utils import require_jit, global_builder
from .utils import set_common_docstring


@overload
def set_atomic_add() -> None:
    ...


@require_jit
@set_common_docstring(api_name="set_atomic_add")
def set_atomic_add(dtype: DataType) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_SetAtomicAddOp(dtype.to_ir())


@overload
def set_atomic_max() -> None:
    ...


@require_jit
@set_common_docstring(api_name="set_atomic_max")
def set_atomic_max(dtype: DataType) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_SetAtomicMaxOp(dtype.to_ir())


@overload
def set_atomic_min() -> None:
    ...


@require_jit
@set_common_docstring(api_name="set_atomic_min")
def set_atomic_min(dtype: DataType) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_SetAtomicMinOp(dtype.to_ir())


@overload
def set_atomic_none() -> None:
    ...


@require_jit
@set_common_docstring(api_name="set_atomic_none")
def set_atomic_none() -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_SetAtomicNoneOp()


@overload
def set_atomic_type() -> None:
    ...


@require_jit
@set_common_docstring(api_name="set_atomic_type")
def set_atomic_type(dtype: DataType) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_SetAtomicTypeOp(dtype.to_ir())