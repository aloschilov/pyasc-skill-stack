# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from typing import Optional, Sequence, Union, overload

from ..._C import ir
from .dtype import DataType
from .ir_value import IRHandle, IRValue, PlainValue, RuntimeInt, \
                    RuntimeNumeric, cast_to_index, materialize_ir_value as _mat
from .utils import check_type, require_jit, global_builder


class Array(IRValue):

    def __init__(self, handle: IRHandle, dtype: DataType, length: int):
        """This contructor should not be called by user"""
        self.handle = handle
        self.dtype = dtype
        self.length = length

    def __len__(self) -> int:
        return self.length

    @require_jit
    def __getitem__(self, index: RuntimeInt) -> RuntimeNumeric:
        handle = global_builder.get_ir_builder().create_memref_LoadOp(self.to_ir(), [cast_to_index(index)])
        return PlainValue(handle)

    @require_jit
    def __setitem__(self, index: RuntimeInt, value: RuntimeNumeric) -> None:
        value = _mat(value, self.dtype)
        global_builder.get_ir_builder().create_memref_StoreOp(value.to_ir(), self.to_ir(), [cast_to_index(index)])

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Array:
        memref_type = handle.get_type()
        dtype = DataType.from_ir(ir.get_element_type(memref_type))
        length = ir.get_shape(memref_type)[0]
        return cls(handle, dtype, length)

    def to_ir(self) -> IRHandle:
        return self.handle


@overload
def array(dtype: DataType, length: int, /, fill_value: Optional[Union[int, float]] = None) -> Array:
    ...


@overload
def array(dtype: DataType, values: Sequence[Union[int, float]], /) -> Array:
    ...


@require_jit
def array(dtype: DataType, length_or_values: Union[int, Sequence[RuntimeNumeric]],
          fill_value: Optional[RuntimeNumeric] = None) -> Array:
    if not dtype.is_numeric():
        raise RuntimeError("Array dtype must be integer or float")
    length = None
    values = None
    if isinstance(length_or_values, int):
        length = length_or_values
        if length <= 0:
            raise RuntimeError("Array length must be a positive integer")
        if fill_value is not None:
            check_type("fill_value", fill_value, RuntimeNumeric)
            values = (fill_value for _ in range(length))
    else:
        if fill_value is not None:
            raise RuntimeError("fill_value cannot be provided together with initial values")
        values = length_or_values
        length = len(values)
    builder = global_builder.get_ir_builder()
    handle = builder.create_memref_AllocaOp(ir.get_memref_type(dtype.to_ir(), length))
    arr = Array(handle, dtype, length)
    if values:
        for index, value in enumerate(values):
            arr.__setitem__(index, value)
    return arr
