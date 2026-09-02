# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
from typing import Iterable, Tuple

from asc._C import ir
from asc.language.core.dtype import KnownTypes as KT
from asc.language.core.utils import allow_jit, global_builder, require_jit

from .local_tensor import LocalTensor, bind_tensor_method
from .tensor_location import TensorLocation
from .utils import cast_tensor_location as cast_loc, infer_common_shape_impl
from .validation import check_dtype, check_type, verify_shape


def shapes_match(shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    if len(shape) > len(target_shape):
        return False
    src = shape[::-1]
    dst = target_shape[::-1]
    for i in range(0, len(dst)):
        if i < len(src) and dst[i] != src[i] and src[i] != 1:
            return False
    return True


def normalize_shape_args(args: tuple) -> Tuple[int, ...]:
    return tuple(args[0]) if len(args) == 1 and isinstance(args[0], Iterable) else args


@bind_tensor_method
@require_jit
def broadcast_to(input: LocalTensor, *shape: int) -> LocalTensor:
    """
    Creates new tensor of a given shape broadcasting data from the input tensor.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        input: The input tensor
        shape: The target shape (can be passed as separate integers or as an iterable, e.g. list or tuple)

    Returns:
        LocalTensor: A new tensor with the broadcasted shape

    Raises:
        TypeError: If input is not a LocalTensor or shape contains non-integer values
        RuntimeError: If the input tensor shape cannot be broadcasted to the target one or shape values are not positive

    Examples:
        Broadcast tensor to the provided shape: ::

            input = asc2.copy_in(x, [0], [256])
            result = input.broadcast_to([16,256])

        The code above may act as the following: ::

            input:   [0,1,2,3,4, ... 255]
            result:  [[0,1,2,...255], [0,1,2,...255] ... [0,1,2,..255]]
    """
    check_type("input", input, LocalTensor)
    check_dtype("input", input, (KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32))
    shape = normalize_shape_args(shape)
    shape = verify_shape(shape)
    if input.shape == shape:
        return input
    if not shapes_match(input.shape, shape):
        raise RuntimeError(f"Cannot broadcast tensor with shape {input.shape} to {shape}")
    input = cast_loc(input, TensorLocation.UB)
    result_type = ir.clone_shaped_type(input.to_ir().get_type(), shape)
    handle = global_builder.get_ir_builder().create_asctile_BroadcastOp(result_type, input.to_ir())
    return cast_loc(LocalTensor(handle))


@allow_jit
def broadcast_shapes(*shapes: Iterable[int]) -> Tuple[int, ...]:
    """
    Compute the common shape that all input shapes can be broadcast to.

    This function applies broadcasting rules to determine the smallest shape that is compatible with all provided
    shapes. Shorter shapes are padded with 1s on the left, then dimensions are compared element-wise, and each
    dimension must either match or be of size 1.

    Args:
        shapes: Variable number of shapes (each shape is an iterable of integers)

    Returns:
        Tuple[int, ...]: The common broadcast shape

    Raises:
        ValueError: If no shapes are provided
        RuntimeError: If the shapes are incompatible and cannot be broadcast together

    Examples:
        Compute common shape for multiple shapes: ::

            common = broadcast_shapes([256], [16, 256])   # returns (16, 256)
            common = broadcast_shapes([1, 256], [16, 1])  # returns (16, 256)
    """
    if not shapes:
        raise ValueError("'shapes' must be provided")
    shapes = tuple(verify_shape(shape, f"shapes[{i}]") for i, shape in enumerate(shapes))
    common_shape = shapes[0]
    if len(shapes) == 1:
        return common_shape
    for shape in shapes:
        common_shape = infer_common_shape_impl(common_shape, shape)
    return common_shape


@require_jit
def broadcast_tensors(*tensors: LocalTensor) -> Tuple[LocalTensor, ...]:
    """
    Broadcast all input tensors to a common shape.

    This function computes the common broadcast shape using :py:func:`broadcast_shapes` and then broadcasts each
    tensor to that shape using :py:func:`broadcast_to`.

    The supported data types are the same as for :py:func:`broadcast_to` function.

    Args:
        tensors: Variable number of LocalTensors to broadcast

    Returns:
        Tuple[LocalTensor, ...]: A tuple of tensors, all broadcasted to the common shape

    Raises:
        TypeError: If any input is not a LocalTensor or has an unsupported dtype
        RuntimeError: If the tensor shapes are incompatible and cannot be broadcast together

    Examples:
        Broadcast multiple tensors to a common shape: ::

            t1 = asc2.copy_in(x, [0], [256])          # shape [256]
            t2 = asc2.copy_in(y, [0, 0], [16, 256])   # shape [16, 256]
            t1_bc, t2_bc = broadcast_tensors(t1, t2)  # both have shape [16, 256]
    """
    if len(tensors) < 2:
        return tensors
    common_shape = broadcast_shapes(*(tensor.shape for tensor in tensors))
    return tuple(tensor.broadcast_to(common_shape) for tensor in tensors)


@bind_tensor_method
@require_jit
def reshape(input: LocalTensor, *shape: int) -> LocalTensor:
    """
    Reshape a tensor to a new shape without changing its data.

    The total number of elements in the new shape must match the total number of elements in the input tensor.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``,
    ``float64``.

    Args:
        input: The input tensor
        shape: The target shape (can be passed as separate integers or as an iterable, e.g. list or tuple)

    Returns:
        LocalTensor: A tensor with the new shape

    Raises:
        TypeError: If input is not a LocalTensor or shape contains non-integer values
        RuntimeError: If the total number of elements doesn't match or shape values are not positive

    Examples:
        Reshape a 1D tensor to 2D: ::

            input = asc2.copy_in(x, [0], [256])
            result = input.reshape([16, 16])

        Reshape a 2D tensor to 1D: ::

            input = asc2.copy_in(x, [0, 0], [32, 16])
            result = input.reshape([512])
    """
    check_type("input", input, LocalTensor)
    check_dtype("input", input,
                (KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32, KT.float64))
    shape = normalize_shape_args(shape)
    shape = verify_shape(shape)
    if input.shape == shape:
        return input
    if math.prod(input.shape) != math.prod(shape):
        raise RuntimeError(f"Reshaping tensor of shape {input.shape} with {math.prod(input.shape)} elements not match "
                           f"output shape {shape} with {math.prod(shape)} elements")
    builder = global_builder.get_ir_builder()
    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), shape)
    handle = builder.create_asctile_ReshapeOp(ir_type, input.to_ir())
    return cast_loc(LocalTensor(handle))


@bind_tensor_method
@require_jit
def ravel(input: LocalTensor) -> LocalTensor:
    """
    Flatten a tensor into a 1D tensor.

    This is equivalent to ``reshape(input, input.size)``.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``,
    ``float64``.

    Args:
        input: The input tensor

    Returns:
        LocalTensor: A 1D tensor with all elements from the input

    Raises:
        TypeError: If input is not a LocalTensor

    Examples:
        Flatten a 2D tensor to 1D: ::

            input = asc2.copy_in(x, [0, 0], [32, 16])
            result = input.ravel()
    """
    return reshape(input, math.prod(input.shape))


@bind_tensor_method
@require_jit
def expand_dims(input: LocalTensor, *axis: int) -> LocalTensor:
    """
    Insert new dimensions of size 1 at the specified positions.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``,
    ``float64``.

    Args:
        input: The input tensor
        axis: The positions where new dimensions should be inserted (0-based)

    Returns:
        LocalTensor: A tensor with the new dimensions inserted

    Raises:
        TypeError: If input is not a LocalTensor

    Note:
        Multiple axes can be specified. Axes are processed in sorted order.

    Examples:
        Insert a dimension at axis 0: ::

            input = asc2.copy_in(x, [0], [256])
            result = input.expand_dims(0)  # shape becomes [1, 256]

        Insert multiple dimensions: ::

            input = asc2.copy_in(x, [0, 0], [32, 16])
            result = input.expand_dims(0, 2)  # shape becomes [1, 32, 1, 16]
    """
    check_type("input", input, LocalTensor)
    shape = list(input.shape)
    axis = sorted(set(axis))
    for ax in axis:
        shape.insert(ax, 1)
    return reshape(input, *shape)


@bind_tensor_method
@require_jit
def squeeze(input: LocalTensor, *axis: int) -> LocalTensor:
    """
    Remove dimensions of size 1 from the tensor.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``,
    ``float64``.

    Args:
        input: The input tensor.
        axis: The positions of dimensions to remove (0-based). If not provided, all dimensions of size 1 are removed.

    Returns:
        LocalTensor: A tensor with the specified dimensions removed

    Raises:
        TypeError: If input is not a LocalTensor
        RuntimeError: If attempting to squeeze a dimension that is not of size 1

    Examples:
        Remove all dimensions of size 1: ::

            input = asc2.copy_in(x, [0, 0, 0, 0], [1, 32, 1, 16])
            result = input.squeeze()  # shape becomes [32, 16]

        Remove a specific dimension: ::

            input = asc2.copy_in(x, [0, 0, 0], [1, 32, 16])
            result = input.squeeze(0)  # shape becomes [32, 16]
    """
    check_type("input", input, LocalTensor)
    shape = []
    axis = set(axis if axis else (i for i, dim in enumerate(input.shape) if dim == 1))
    for i, dim in enumerate(input.shape):
        if i not in axis:
            shape.append(dim)
            continue
        if dim != 1:
            raise RuntimeError(f"Unable to squeeze the axis {i} since its length must be 1, got {dim}")
    return reshape(input, *shape)


@bind_tensor_method
@require_jit
def transpose(input: LocalTensor, *axis: int) -> LocalTensor:
    """
    Rearrange tensor dimensions in specific order.
    The supported data types are: ``int8``, ``int16``, ``int32``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        input: The input tensor
        axis: Order of input dimensions in result. Swaps two last dimensions when no axis provided

    Returns:
        LocalTensor: The transposed tensor with swapped dimensions

    Raises:
        TypeError: If input is not a LocalTensor
        RuntimeError: If the input tensor dtype is not supported or axis is incorrect

    Note:
        If the input tensor was created by :py:func:`copy_in` and used only as ``transpose()`` argument,
        both operations will be fused into a single **data copy operation** during the compilation.
        In this case any 2D, 3D, 4D tensor is supported.

        If the input is used by other operations or not created by :py:func:`copy_in`, a standalone transpose is used.
        In this case **only 2D tensors in UB are supported**,
        and input shape must be multiple of 16 (for 2 or 4 byte elements) or 32 (for 1 byte elements).

    Examples:
        Transpose a 2D tensor: ::

            input = asc2.copy_in(x, [0, 0, 0], [32, 16])
            result = input.transpose()  # shape becomes [32, 16], same as input.transpose(1, 0)

        Transpose a 3D tensor with specific order: ::

            input = asc2.copy_in(x, [0, 0, 0], [32, 64, 16])
            result = input.transpose(2, 0, 1)  # shape becomes [16, 32, 64]

        Transpose as a standalone operation: ::

            input = asc2.copy_in(x, [0, 0], [64, 64])
            temp = input + 2.0  # local tensor modified after the copy_in
            result = temp.transpose()
    """
    check_type("input", input, LocalTensor)
    check_dtype("input", input, (KT.int8, KT.int16, KT.int32, KT.float16, KT.bfloat16, KT.float32))
    rank = len(input.shape)
    if len(axis) == 0:
        axis = list(range(0, rank))
        axis[-1], axis[-2] = axis[-2], axis[-1]
    if len(axis) != rank:
        raise RuntimeError(f"Transpose axis count {len(axis)} should match count of tensor dimensions {rank}")
    if list(axis) == list(range(0, rank)):  # Identity transformation
        return input
    if set(axis) != set(range(0, rank)):
        raise RuntimeError(f"Wrong dimensions rearrangement {axis} for tensor of {rank} dimensions")
    result_shape = [input.shape[i] for i in axis]
    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), result_shape)
    builder = global_builder.get_ir_builder()
    handle = builder.create_asctile_TransposeOp(ir_type, input.to_ir(), builder.get_i32_array_attr(axis))
    return cast_loc(LocalTensor(handle))
