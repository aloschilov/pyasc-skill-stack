# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations
from typing import overload, Union

from ..._C import ir
from ..core.constexpr import ConstExpr
from ..core.dtype import KnownTypes
from ..core.enums import Hardware, TPosition
from ..core.ir_value import IRHandle, materialize_ir_value
from ..core.tensor import LocalTensor
from ..core.types import DataType
from ..core.utils import DefaultValued, OverloadDispatcher, global_builder, require_jit, set_class_docstring
from .ir_value import RuntimeInt


class LocalMemAllocator:
    """
    LocalMemAllocator是在使用Ascend C静态Tensor编程方式时用于内存管理的类，用户无需构建TPipe/TQue，
    而是直接创建LocalTensor对象并开发算子，从而减少运行时的开销，实现更优的性能。

    LocalMemAllocator仅支持在Ascend C静态Tensor编程方式中使用，不可以与TPipe等接口混用。
    """

    @overload
    def __init__(self, hard: Hardware = Hardware.UB) -> None:
        ...

    @require_jit
    def __init__(self, *args, **kwargs) -> None:
        self.handle: IRHandle = None
        dispatcher = OverloadDispatcher(__name__)

        @dispatcher.register(hard=DefaultValued(Hardware, Hardware.UB))
        def _(hard: Hardware):
            builder = global_builder.get_ir_builder()
            allocator_type = builder.get_local_mem_allocator_type(hard.value)
            hardware_ir = materialize_ir_value(hard.value, KnownTypes.int64).to_ir()
            self.handle = builder.create_asc_LocalMemAllocatorOp(allocator_type, hardware_ir)

        dispatcher(*args, **kwargs)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> LocalMemAllocator:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    @overload
    def get_cur_addr(self) -> int:
        ...

    @require_jit
    @set_class_docstring("LocalMemAllocator", "get_cur_addr")
    def get_cur_addr(self) -> RuntimeInt:
        builder = global_builder.get_ir_builder()
        result_type = builder.get_ui32_type()
        return builder.create_asc_LocalMemAllocatorGetCurAddrOp(result_type, self.handle)

    @overload
    def alloc(self, pos: TPosition, data_type: DataType, tile_size: ConstExpr[int]) -> LocalTensor:
        ...

    @overload
    def alloc(self, pos: TPosition, data_type: DataType, tile_size: int) -> LocalTensor:
        ...

    @require_jit
    @set_class_docstring("LocalMemAllocator", "alloc")
    def alloc(self, pos: TPosition, data_type: DataType, tile_size: Union[ConstExpr[int], int]) -> LocalTensor:
        builder = global_builder.get_ir_builder()
        result_type = ir.get_local_tensor_type(data_type.to_ir())
        pos_attr = ir.TPosition.symbolize(pos)

        if isinstance(tile_size, ConstExpr):
            tile_size_value = ConstExpr.unwrap(tile_size)
            tile_size_ir = builder.create_i32(tile_size_value)
            handle = builder.create_asc_LocalMemAllocatorAllocOp(result_type, self.handle, pos_attr, data_type.to_ir(),
                                                                 tile_size_ir)
        else:
            tile_size_ir = materialize_ir_value(tile_size, KnownTypes.int32).to_ir()
            handle = builder.create_asc_LocalMemAllocatorAllocDynamicOp(result_type, self.handle, pos_attr,
                                                                        data_type.to_ir(), tile_size_ir)

        return LocalTensor.from_ir(handle)
