# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List, Tuple, Union, overload

from asc._C import ir
from asc.language.core.dtype import DataType, KnownTypes as KT
from asc.language.core.ir_value import PlainValue
from asc.language.core.utils import global_builder, require_jit

from .local_tensor import LocalTensor, bind_tensor_method
from .tensor_location import TensorLocation
from .utils import cast_tensor_location as cast_loc
from .validation import check_dtype, check_type


def get_reduction_shape(tensor_shape: Tuple[int, ...], keep_dims: bool, dims: Tuple[int, ...]) -> List[int]:
    reduce_dims = [False] * len(tensor_shape)
    for dim in dims:
        reduce_dims[dim] = True
    result = []
    for i in range(0, len(reduce_dims)):
        if not reduce_dims[i]:
            result.append(tensor_shape[i])
        elif keep_dims:
            result.append(1)
    return result


def op_reduce_impl(input: LocalTensor, keep_dims: bool, dims: Tuple[int, ...], kind: ir.ReduceKind,
                   support_dtypes: Tuple[DataType, ...],
                   support_dtypes_as_1d: Tuple[DataType, ...]) -> Union[LocalTensor, PlainValue]:
    check_type("input", input, LocalTensor)
    check_type("keep_dims", keep_dims, bool)
    input = cast_loc(input, TensorLocation.UB)
    builder = global_builder.get_ir_builder()
    if len(dims) == 0:
        if not support_dtypes_as_1d:
            raise RuntimeError("Reduction to scalar not supported")
        check_dtype("input", input, support_dtypes_as_1d)
        handle = builder.create_asctile_ReduceAs1dOp(input.dtype.to_ir(), input.to_ir(), kind)
        return PlainValue(handle)
    if not all(isinstance(dim, int) for dim in dims):
        raise TypeError("All reduction dimensions must be int")
    if len(dims) != len(set(dims)):
        raise RuntimeError("Repeating dimensions are not allowed")
    if not all(dim >= 0 and dim < input.rank for dim in dims):
        raise RuntimeError(f"All reduction dimensions must be between 0 and {input.rank - 1}")
    dims = tuple(dim for dim in dims if input.shape[dim] > 1)
    dim_ones = tuple(dim for dim in dims if input.shape[dim] == 1)
    if not keep_dims:
        input = input.squeeze(*dim_ones)
    if not dims:
        return input
    check_dtype("input", input, support_dtypes)
    target_shape = get_reduction_shape(input.shape, keep_dims, dims)
    if not target_shape:
        raise RuntimeError("Reduction along all dimensions is not supported")
    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), target_shape)
    dims_attr = global_builder.get_ir_builder().get_i32_array_attr(dims)
    handle = builder.create_asctile_ReduceOp(ir_type, input.to_ir(), dims_attr, kind)
    return cast_loc(LocalTensor(handle))


@overload
def reduce_sum(input: LocalTensor, *dims: int, keep_dims: bool = False) -> LocalTensor:
    ...


@overload
def reduce_sum(input: LocalTensor) -> PlainValue:
    ...


@bind_tensor_method(name="sum")
@require_jit
def reduce_sum(input: LocalTensor, *dims: int, keep_dims: bool = False) -> Union[LocalTensor, PlainValue]:
    """
    Returns the sum of each row of the ``input`` tensor in the given dimensions ``dims``.

    Dimensions ``dims`` are squeezed, resulting the output tensor having fewer dimensions than input,
    unless ``keep_dims=True`` is provided.
    When dimension is not specified, the entire tensor is reduced to a single scalar value.

    The supported data types for the input are: ``int32``, ``int64``, ``float32``.
    When reducing to a single scalar value, the supported data types are: ``int64``, ``float16``, ``float32``.

    Args:
        input: The input tensor
        dims: Optional, dimensions to reduce, should be in range of [0..len(input.shape)-1]
        keep_dims: If set to True, then reduced dimensions are kept in the result shape with size of 1

    Raises:
        TypeError: If input is not a ``LocalTensor``, keep_dims is not a bool, or dims contains non-integer values
        RuntimeError: If input dtype is not supported, or if ``dims`` explicitly lists all dimensions

    Examples:
        Reduce tensor by first (outermost) dimension, resulting tensor having the shape [256],
        each element is sum of 128 elements in corresponding column: ::

            input = asc2.copy_in(x, [0, 0], [128, 256])
            result = asc2.reduce_sum(input, 0)

        Compute total sum of all numbers in tensor, returns single scalar value: ::

            input = asc2.copy_in(x, [0, 0], [256, 256])
            result = asc2.reduce_sum(input)
    """
    return op_reduce_impl(input, keep_dims, dims, ir.ReduceKind.Sum, support_dtypes=(KT.int32, KT.int64, KT.float32),
                          support_dtypes_as_1d=(KT.int64, KT.float16, KT.float32))


@overload
def reduce_max(input: LocalTensor, *dims: int, keep_dims: bool = False) -> LocalTensor:
    ...


@overload
def reduce_max(input: LocalTensor) -> PlainValue:
    ...


@bind_tensor_method(name="max")
@require_jit
def reduce_max(input: LocalTensor, *dims: int, keep_dims: bool = False) -> Union[LocalTensor, PlainValue]:
    """
    Returns the maximum value of each row of the ``input`` tensor in the given dimensions ``dims``.

    Dimensions ``dims`` are squeezed, resulting the output tensor having fewer dimensions than input,
    unless ``keep_dims=True`` is provided.
    When dimension is not specified, the entire tensor is reduced to a single scalar value.

    The supported data types for the input are:
    ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.
    When reducing to a single scalar value, the supported data types are:
    ``int16``, ``int32``, ``int64``, ``float16``, ``float32``.

    Args:
        input: The input tensor
        dims: Optional, dimensions to reduce, should be in range of [0..len(input.shape)-1]
        keep_dims: If set to True, then reduced dimensions are kept in the result shape with size of 1

    Raises:
        TypeError: If input is not a ``LocalTensor``, keep_dims is not a bool, or dims contains non-integer values
        RuntimeError: If input dtype is not supported, or if ``dims`` explicitly lists all dimensions

    Examples:
        Reduce tensor by first (outermost) dimension, resulting tensor having the shape [256],
        each element is a maximum value between 128 elements in corresponding column: ::

            input = asc2.copy_in(x, [0, 0], [128, 256])
            result = asc2.reduce_max(input, 0)

        Compute the maximum value between all tensor elements, returns single scalar value: ::

            input = asc2.copy_in(x, [0, 0], [256, 256])
            result = asc2.reduce_max(input)
    """
    return op_reduce_impl(input, keep_dims, dims, ir.ReduceKind.Max,
                          support_dtypes=(KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32),
                          support_dtypes_as_1d=(KT.int16, KT.int32, KT.int64, KT.float16, KT.float32))


@overload
def reduce_min(input: LocalTensor, *dims: int, keep_dims: bool = False) -> LocalTensor:
    ...


@overload
def reduce_min(input: LocalTensor) -> PlainValue:
    ...


@bind_tensor_method(name="min")
@require_jit
def reduce_min(input: LocalTensor, *dims: int, keep_dims: bool = False) -> Union[LocalTensor, PlainValue]:
    """
    Returns the minimum value of each row of the ``input`` tensor in the given dimensions ``dims``.

    Dimensions ``dims`` are squeezed, resulting the output tensor having fewer dimensions than input,
    unless ``keep_dims=True`` is provided.
    When dimension is not specified, the entire tensor is reduced to a single scalar value.

    The supported data types for the input are:
    ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.
    When reducing to a single scalar value, the supported data types are:
    ``int16``, ``int32``, ``int64``, ``float16``, ``float32``.

    Args:
        input: The input tensor
        dims: Optional, dimensions to reduce, should be in range of [0..len(input.shape)-1]
        keep_dims: If set to True, then reduced dimensions are kept in the result shape with size of 1

    Raises:
        TypeError: If input is not a ``LocalTensor``, keep_dims is not a bool, or dims contains non-integer values
        RuntimeError: If input dtype is not supported, or if ``dims`` explicitly lists all dimensions

    Examples:
        Reduce tensor by first (outermost) dimension, resulting tensor having the shape [256],
        each element is a minimum value between 128 elements in corresponding column: ::

            input = asc2.copy_in(x, [0, 0], [128, 256])
            result = asc2.reduce_min(input, 0)

        Compute the minimum value between all tensor elements, returns single scalar value: ::

            input = asc2.copy_in(x, [0, 0], [256, 256])
            result = asc2.reduce_min(input)
    """
    return op_reduce_impl(input, keep_dims, dims, ir.ReduceKind.Min,
                          support_dtypes=(KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32),
                          support_dtypes_as_1d=(KT.int16, KT.int32, KT.int64, KT.float16, KT.float32))


@bind_tensor_method(name="prod")
@require_jit
def reduce_prod(input: LocalTensor, *dims: int, keep_dims: bool = False) -> LocalTensor:
    """
    Returns the product of each row of the ``input`` tensor in the given dimensions ``dims``.

    Dimensions ``dims`` are squeezed, resulting the output tensor having fewer dimensions than input,
    unless ``keep_dims=True`` is provided.

    The supported data types for the input are: ``float32``.
    Reduction to a single scalar value is not supported.

    Args:
        input: The input tensor
        dims: Dimensions to reduce, should be in range of [0..len(input.shape)-1]
        keep_dims: If set to True, then reduced dimensions are kept in the result shape with size of 1

    Raises:
        TypeError: If input is not a ``LocalTensor``, keep_dims is not a bool, or dims contains non-integer values
        RuntimeError: If input dtype is not supported, if ``dims`` explicitly lists all dimensions,
                      or if reducing to a scalar (no dims provided)

    Examples:
        Reduce tensor by first (outermost) dimension, resulting tensor having the shape [256],
        each element is product of 128 elements in corresponding column: ::

            input = asc2.copy_in(x, [0, 0], [128, 256])
            result = asc2.reduce_prod(0)
    """
    return op_reduce_impl(input, keep_dims, dims, ir.ReduceKind.Prod, support_dtypes=(KT.float32, ),
                          support_dtypes_as_1d=())
