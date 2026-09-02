# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from typing import Generator, Iterable
from typing_extensions import Self

from asc._C import ir
from asc.language.core.dtype import DataType, int32
from asc.language.core.tensor import GlobalAddress
from asc.language.core.ir_value import IRHandle, IRValue, PlainValue, RuntimeInt, materialize_ir_value as mat
from asc.language.core.utils import global_builder, require_jit

from .validation import check_type, verify_runtime_ints


class RuntimeShape:
    """
    A tuple-like object representing a shape of a tensor.

    It has length, item getter, and can be iterated as an ordinary tuple.
    """

    def __init__(self, ir_tensor: IRHandle, shape: Iterable[int]) -> None:
        """This constructor is not called by user. Use ``shape`` attribute of :py:class:`GlobalTensor` object."""
        self.ir_tensor = ir_tensor
        self.shape = tuple(shape)

    def normalize_index(self, index: int) -> int:
        check_type("index", index, int)
        rank = len(self.shape)
        if index < 0:
            index = rank + index
        if index < 0 or index >= rank:
            raise IndexError(f"shape index {index} out of range")
        return index

    def __getitem__(self, index: int) -> RuntimeInt:
        index = self.normalize_index(index)
        dim = self.shape[index]
        if dim != ir.dynshape:
            return dim
        return PlainValue(global_builder.get_ir_builder().create_asctile_DimOp(int32.to_ir(), self.ir_tensor, index))

    def __len__(self) -> int:
        return len(self.shape)

    def __iter__(self) -> Generator[RuntimeInt, None, None]:
        for i in range(len(self)):
            yield self[i]

    def is_static(self) -> bool:
        return all(dim != ir.dynshape for dim in self.shape)

    def is_dynamic_dim(self, index: int) -> bool:
        index = self.normalize_index(index)
        return self.shape[index] == ir.dynshape


class GlobalTensor(IRValue):
    """
    A global tensor is a contiguous ND-array of values in Global Memory.

    Each element is of :py:attr:`dtype` type and number of elements is defined by :py:attr:`shape` values.
    """

    dtype: DataType
    """Tensor element type"""

    shape: RuntimeShape
    """Tensor shape"""

    rank: int
    """Number of dimensions"""

    def __init__(self, *, handle: IRHandle) -> None:
        """This constructor is not called by user. Use :py:func:`global_tensor` function to define a global tensor."""
        super().__init__()
        check_type("handle", handle, IRHandle)
        self.handle = handle
        ir_type = self.handle.get_type()
        self.dtype = DataType.from_ir(ir.get_element_type(ir_type))
        self.shape = RuntimeShape(self.handle, ir.get_shape(ir_type))
        self.rank = len(self.shape)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


@require_jit
def global_tensor(base: GlobalAddress, shape: Iterable[RuntimeInt]) -> GlobalTensor:
    """
    Define a new tensor descriptor for accessing data in global memory.

    Tensors represent contiguous ND-arrays in global memory and are used to transfer data between global and local
    memory via :py:func:`copy_in` and :py:func:`copy_out` operations.

    Args:
        base: The base address of an array in global memory representing the tensor
        shape: An iterable of integer-like values representing the number of elements for each dimension

    Returns:
        GlobalTensor: A new tensor descriptor

    Raises:
        TypeError: If base is not a GlobalAddress or shape contains non-integer values
        RuntimeError: If shape is empty

    Examples:
        Create a 1D tensor with static shape: ::

            x_gm = asc2.global_tensor(x_ptr, [1024])

        Create a 2D tensor with static shape: ::

            x_gm = asc2.global_tensor(x_ptr, [64, 128])

        Create a tensor with dynamic shape (using runtime values): ::

            x_gm = asc2.global_tensor(x_ptr, [num_rows, num_cols])
    """
    check_type("base", base, GlobalAddress)
    shape = verify_runtime_ints(shape, "shape")
    static_sizes = []
    dynamic_sizes = []
    for dim in shape:
        if isinstance(dim, int):
            static_sizes.append(dim)
        else:
            static_sizes.append(ir.dynshape)
            dynamic_sizes.append(mat(dim, int32).to_ir())
    ir_type = ir.get_asctile_GlobalTensorType(static_sizes, base.dtype.to_ir())
    handle = global_builder.get_ir_builder().create_asctile_TensorOp(ir_type, base.to_ir(), dynamic_sizes)
    return GlobalTensor.from_ir(handle)
