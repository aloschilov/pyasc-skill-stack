# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, overload

from ..core.dtype import KnownTypes
from ..core.ir_value import IRValue, RuntimeInt, materialize_ir_value as _mat
from ..core.tensor import BaseTensor, GlobalTensor, LocalTensor
from ..core.types import ShapeInfo
from ..core.utils import require_jit, global_builder, check_type
from .utils import set_common_docstring


@overload
def dump_tensor(tensor: GlobalTensor, desc: int, dump_size: int, shape_info: Optional[ShapeInfo] = None) -> None:
    ...


@overload
def dump_tensor(tensor: LocalTensor, desc: int, dump_size: int, shape_info: Optional[ShapeInfo] = None) -> None:
    ...


@require_jit
@set_common_docstring(api_name="dump_tensor")
def dump_tensor(tensor: BaseTensor, desc: RuntimeInt, dump_size: RuntimeInt,
                shape_info: Optional[ShapeInfo] = None) -> None:
    check_type("desc", desc, RuntimeInt)
    check_type("dump_size", dump_size, RuntimeInt)
    if shape_info is not None:
        check_type("shape_info", shape_info, ShapeInfo)
        shape_info = shape_info.to_ir()
    global_builder.get_ir_builder().create_asc_DumpTensorOp(tensor.to_ir(),
                                                            _mat(desc, KnownTypes.uint32).to_ir(),
                                                            _mat(dump_size, KnownTypes.uint32).to_ir(),
                                                            shapeInfo=shape_info)


@require_jit
@set_common_docstring(api_name="printf")
def printf(desc: str, *params) -> None:
    var_ir_values = []
    desc_str_list = desc.split("%s")
    new_desc = ""
    str_index = 0
    for var in params:
        if isinstance(var, str):
            new_desc += desc_str_list[str_index] + var
            str_index += 1
        elif isinstance(var, IRValue):
            var_ir_values.append(var.to_ir())
        else:
            var_ir_values.append(_mat(var).to_ir())
    if new_desc != "":
        new_desc += desc_str_list[str_index]
        global_builder.get_ir_builder().create_asc_PrintfOp(new_desc, var_ir_values)
    else:
        global_builder.get_ir_builder().create_asc_PrintfOp(desc, var_ir_values)


@overload
def print_time_stamp(desc_id: int) -> None:
    ...


@require_jit
@set_common_docstring(api_name="print_time_stamp")
def print_time_stamp(desc_id: RuntimeInt) -> None:
    global_builder.get_ir_builder().create_asc_PrintTimeStampOp(_mat(desc_id, KnownTypes.int_).to_ir())


@overload
def dump_acc_chk_point(tensor: LocalTensor, index: int, count_off: int, dump_size: int) -> None:
    ...


@overload
def dump_acc_chk_point(tensor: GlobalTensor, index: int, count_off: int, dump_size: int) -> None:
    ...


@require_jit
@set_common_docstring(api_name="dump_acc_chk_point")
def dump_acc_chk_point(tensor: BaseTensor, index: RuntimeInt, count_off: RuntimeInt, dump_size: RuntimeInt) -> None:
    check_type("index", index, RuntimeInt)
    check_type("count_off", count_off, RuntimeInt)
    check_type("dump_size", dump_size, RuntimeInt)

    builder = global_builder.get_ir_builder()
    builder.create_asc_DumpAccChkPointOp(
        tensor.to_ir(),
        _mat(index, KnownTypes.uint32).to_ir(),
        _mat(count_off, KnownTypes.uint32).to_ir(),
        _mat(dump_size, KnownTypes.uint32).to_ir(),
    )


@overload
def metrics_prof_start() -> None:
    ...


@require_jit
@set_common_docstring(api_name="metrics_prof_start")
def metrics_prof_start() -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_MetricsProfStartOp()


@overload
def metrics_prof_stop() -> None:
    ...


@require_jit
@set_common_docstring(api_name="metrics_prof_stop")
def metrics_prof_stop() -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_MetricsProfStopOp()