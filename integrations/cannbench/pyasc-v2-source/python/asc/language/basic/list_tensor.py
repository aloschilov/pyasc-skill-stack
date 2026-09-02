# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations
from typing import overload

from ..._C import ir
from ..core.dtype import DataType, KnownTypes as KT
from ..core.ir_value import GlobalAddress, IRValue, IRHandle, PlainValue, RuntimeInt, materialize_ir_value as _mat
from ..core.utils import require_jit, global_builder
from ..core.tensor import GlobalTensor
from .utils import set_tensor_docstring


class TensorDesc(IRValue):

    @overload
    def __init__(self, dtype: DataType = KT.float32) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle, dtype: DataType = KT.float32) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        dtype = kwargs.pop("dtype", KT.float32)
        if 'handle' in kwargs:
            self.handle = kwargs['handle']
            self.dtype = dtype
            return
        if global_builder.get_ir_builder() is not None:
            self.__initjit__(dtype=dtype)
            return
        builder = global_builder.get_ir_builder()
        self.dtype = dtype
        self.handle = builder.create_asc_TensorDescOp(builder.get_asc_TensorDescType(), dtype.to_ir())

    @require_jit
    def __initjit__(self, dtype: DataType = KT.float32) -> None:
        builder = global_builder.get_ir_builder()
        self.dtype = dtype
        self.handle = builder.create_asc_TensorDescOp(builder.get_asc_TensorDescType(), dtype.to_ir())

    @classmethod
    def from_ir(cls, handle: IRHandle, dtype: DataType = KT.float32) -> TensorDesc:
        return cls(handle=handle, dtype=dtype)

    def to_ir(self) -> IRHandle:
        return self.handle

    @overload
    def set_shape_addr(self, shape_ptr: int) -> None:
        ...

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="set_shape_addr")
    def set_shape_addr(self, shape_ptr: RuntimeInt) -> None:
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_TensorDescSetShapeAddrOp(self.to_ir(), _mat(shape_ptr, KT.uint64).to_ir())

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="get_dim")
    def get_dim(self) -> RuntimeInt:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TensorDescGetDimOp(builder.get_ui64_type(), self.to_ir())
        return PlainValue(handle)

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="get_index")
    def get_index(self) -> RuntimeInt:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TensorDescGetIndexOp(builder.get_ui64_type(), self.to_ir())
        return PlainValue(handle)

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="get_shape")
    def get_shape(self, offset: RuntimeInt) -> RuntimeInt:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TensorDescGetShapeOp(builder.get_i64_type(), self.to_ir(),
                                                         _mat(offset, KT.int32).to_ir())
        return PlainValue(handle)

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="get_data_ptr")
    def get_data_ptr(self) -> GlobalAddress:
        builder = global_builder.get_ir_builder()
        element_type = self.dtype.to_ir()
        handle = builder.create_asc_TensorDescGetDataPtrOp(
            ir.get_unranked_memref_type(element_type, ir.AddressSpace.gm), self.to_ir())
        return GlobalAddress(handle, self.dtype)

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="get_data_obj")
    def get_data_obj(self) -> GlobalTensor:
        builder = global_builder.get_ir_builder()
        element_type = self.dtype.to_ir()
        handle = builder.create_asc_TensorDescGetDataObjOp(ir.get_global_tensor_type(element_type), self.to_ir())
        return GlobalTensor(handle)


class ListTensorDesc(IRValue):

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, data: GlobalAddress, length: int = 0xffffffff, shape_size: int = 0xffffffff) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(self, *args, **kwargs) -> None:
        if 'handle' in kwargs:
            self.handle = kwargs['handle']
            return
        builder = global_builder.get_ir_builder()
        if 'data' in kwargs:
            data = kwargs['data']
            length = kwargs['length']
            shape_size = kwargs['shape_size']
            self.handle = builder.create_asc_ListTensorDescV2Op(builder.get_asc_ListTensorDescType(), data.to_ir(),
                                                                _mat(length, KT.uint32).to_ir(),
                                                                _mat(shape_size, KT.uint32).to_ir())
            return
        self.handle = builder.create_asc_ListTensorDescOp(builder.get_asc_ListTensorDescType())

    @classmethod
    def from_ir(cls, handle: IRHandle) -> ListTensorDesc:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    @overload
    def init(self, data: GlobalAddress, length: int = 0xffffffff, shape_size: int = 0xffffffff) -> None:
        ...

    @require_jit
    @set_tensor_docstring(tensor_name="ListTensorDesc", api_name="init")
    def init(self, data: GlobalAddress, length: RuntimeInt = 0xffffffff, shape_size: RuntimeInt = 0xffffffff) -> None:
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ListTensorDescInitOp(self.to_ir(), data.to_ir(),
                                                              _mat(length, KT.uint32).to_ir(),
                                                              _mat(shape_size, KT.uint32).to_ir())

    @overload
    def get_desc(self, desc: TensorDesc, index: int) -> None:
        ...

    @require_jit
    @set_tensor_docstring(tensor_name="ListTensorDesc", api_name="get_desc")
    def get_desc(self, desc: TensorDesc, index: RuntimeInt) -> None:
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ListTensorDescGetDescOp(self.to_ir(), desc.to_ir(),
                                                                 _mat(index, KT.uint32).to_ir())

    @overload
    def get_data_ptr(self, index: int, dtype: DataType) -> GlobalAddress:
        ...

    @require_jit
    @set_tensor_docstring(tensor_name="ListTensorDesc", api_name="get_data_ptr")
    def get_data_ptr(self, index: RuntimeInt, dtype: DataType) -> GlobalAddress:
        builder = global_builder.get_ir_builder()
        ga_type = ir.get_unranked_memref_type(dtype.to_ir(), ir.AddressSpace.gm)
        handle = builder.create_asc_ListTensorDescGetDataPtrOp(ga_type, self.to_ir(),
                                                               _mat(index, KT.uint32).to_ir(), dtype.to_ir())
        return GlobalAddress(handle)

    @overload
    def get_size(self) -> int:
        ...

    @require_jit
    @set_tensor_docstring(tensor_name="ListTensorDesc", api_name="get_size")
    def get_size(self) -> RuntimeInt:
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ListTensorDescGetSizeOp(builder.get_ui32_type(), self.to_ir())
