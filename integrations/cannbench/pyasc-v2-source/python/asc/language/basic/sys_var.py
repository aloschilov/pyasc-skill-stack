# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload

from ..core.dtype import KnownTypes
from ..core.ir_value import PlainValue, RuntimeInt, materialize_ir_value as _mat
from ..core.utils import require_jit, global_builder
from .utils import set_common_docstring


@overload
def get_arch_version(core_version: int) -> None:
    ...


@require_jit
def get_arch_version(core_version: RuntimeInt) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_GetArchVersionOp(_mat(core_version, KnownTypes.uint32).to_ir())


@overload
def get_block_idx() -> int:
    ...


@require_jit
@set_common_docstring(api_name="get_block_idx")
def get_block_idx() -> RuntimeInt:
    return PlainValue(global_builder.get_ir_builder().create_asc_GetBlockIdxOp(KnownTypes.int_.to_ir()))


@overload
def get_block_num() -> int:
    ...


@require_jit
@set_common_docstring(api_name="get_block_num")
def get_block_num() -> RuntimeInt:
    return PlainValue(global_builder.get_ir_builder().create_asc_GetBlockNumOp(KnownTypes.int_.to_ir()))


@overload
def get_data_block_size_in_bytes() -> int:
    ...


@require_jit
@set_common_docstring(api_name="get_data_block_size_in_bytes")
def get_data_block_size_in_bytes() -> RuntimeInt:
    return PlainValue(global_builder.get_ir_builder().create_asc_GetDataBlockSizeInBytesOp(KnownTypes.int_.to_ir()))


@overload
def get_program_counter() -> int:
    ...


@require_jit
@set_common_docstring(api_name="get_program_counter")
def get_program_counter() -> RuntimeInt:
    return PlainValue(global_builder.get_ir_builder().create_asc_GetProgramCounterOp(KnownTypes.int64.to_ir()))


@overload
def get_sub_block_idx() -> int:
    ...


@require_jit
@set_common_docstring(api_name="get_sub_block_idx")
def get_sub_block_idx() -> RuntimeInt:
    return PlainValue(global_builder.get_ir_builder().create_asc_GetSubBlockIdxOp(KnownTypes.int64.to_ir()))


@overload
def get_system_cycle() -> int:
    ...


@require_jit
@set_common_docstring(api_name="get_system_cycle")
def get_system_cycle() -> RuntimeInt:
    return PlainValue(global_builder.get_ir_builder().create_asc_GetSystemCycleOp(KnownTypes.int64.to_ir()))


@overload
def get_task_ratio() -> int:
    ...


@require_jit
@set_common_docstring(api_name="get_task_ratio")
def get_task_ratio() -> RuntimeInt:
    return PlainValue(global_builder.get_ir_builder().create_asc_GetTaskRatioOp(KnownTypes.int64.to_ir()))


@require_jit
@set_common_docstring(api_name="trap")
def trap() -> None:
    global_builder.get_ir_builder().create_asc_TrapOp()


@require_jit
def get_sub_block_num() -> RuntimeInt:
    return PlainValue(global_builder.get_ir_builder().create_asc_GetSubBlockNumOp(KnownTypes.uint64.to_ir()))
