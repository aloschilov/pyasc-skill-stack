# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from numbers import Real
from typing import Iterable, Optional, Union, overload

from asc._C import ir
from asc.language.core.dtype import DataType, KnownTypes as KT
from asc.language.core.ir_value import PlainValue, RuntimeNumeric, materialize_ir_value
from asc.language.core.utils import global_builder, require_jit

from .local_tensor import LocalTensor, RoundMode
from .tensor_location import TensorLocation, TensorLocLike
from .utils import cast_tensor_location as cast_loc, check_bias, constant_tile, splat_tile
from .validation import check_dtype, check_type, verify_location, verify_shape


@require_jit
def full(shape: Iterable[int], value: RuntimeNumeric, dtype: Optional[DataType] = None,
         location: TensorLocLike = TensorLocation.UB) -> LocalTensor:
    """
    Create a tensor filled with a scalar value.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        shape: The shape of the tensor to create.
        value: The scalar value to fill the tensor with.
        dtype: The data type of the tensor. If None, inferred from the value type.
        location: The memory location for the tensor. Default is ``TensorLocation.UB``.

    Returns:
        LocalTensor: A new tensor filled with the specified value

    Raises:
        TypeError: If value is not a numeric type, dtype is not a DataType, or location is not a TensorLocation-like
        RuntimeError: If shape is invalid or dtype is not supported

    Examples:
        Create a tensor filled with a constant integer value: ::

            result = asc2.full([128], 42, dtype=asc2.int32)

        Create a tensor filled with a floating-point value: ::

            result = asc2.full([32, 16], 3.14, dtype=asc2.float16)

        Create a tensor with dtype inferred from the value: ::

            result = asc2.full([64], 0)       # inferred as int32
            result = asc2.full([64], 1.5)     # inferred as float32
    """
    check_type("value", value, RuntimeNumeric)
    check_type("dtype", dtype, Optional[DataType])
    support_dtypes = (KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32)
    check_dtype("dtype", dtype, support_dtypes, optional=True)
    location = verify_location(location, allow=TensorLocation.UB)
    shape = verify_shape(shape)
    if isinstance(value, Real):
        if dtype is None:
            dtype = KT.int32 if isinstance(value, int) else KT.float32
        return cast_loc(constant_tile(value, shape, dtype, TensorLocation.UB))
    if dtype is None:
        check_dtype("value", value, support_dtypes)
        dtype = value.dtype
    return cast_loc(splat_tile(value, shape, dtype, TensorLocation.UB))


@require_jit
def full_like(input: LocalTensor, value: RuntimeNumeric, location: Optional[TensorLocLike] = None) -> LocalTensor:
    """
    Create a tensor filled with a scalar value, with the same shape and dtype as the input tensor.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        input: The input tensor to match shape and dtype.
        value: The scalar value to fill the tensor with.
        location: The memory location for the tensor. Default is ``input.location``.

    Returns:
        LocalTensor: A new tensor filled with the specified value

    Raises:
        TypeError: If input is not a LocalTensor

    Examples:
        Create a tensor filled with a value, matching another tensor's shape and dtype: ::

            src = asc2.copy_in(x_gm, [0], [128])
            result = asc2.full_like(src, 255)
    """
    check_type("input", input, LocalTensor)
    location = input.location if location is None else location
    return full(input.shape, value, input.dtype, location)


@require_jit
def zeros(shape: Iterable[int], dtype: DataType = KT.int32, location: TensorLocLike = TensorLocation.UB) -> LocalTensor:
    """
    Create a tensor filled with zeros.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        shape: The shape of the tensor to create.
        dtype: The data type of the tensor. Default is ``int32``.
        location: The memory location for the tensor. Default is ``TensorLocation.UB``.

    Returns:
        LocalTensor: A new tensor filled with zeros

    Raises:
        TypeError: If dtype is not a DataType or location is not a TensorLocation
        RuntimeError: If shape is invalid or dtype is not supported

    Examples:
        Create a zero-filled tensor with default dtype (int32): ::

            result = asc2.zeros([128])

        Create a zero-filled tensor with a specific dtype: ::

            result = asc2.zeros([32, 16], dtype=asc2.float16)
    """
    return full(shape, 0, dtype, location)


@require_jit
def zeros_like(input: LocalTensor, location: Optional[TensorLocLike] = None) -> LocalTensor:
    """
    Create a tensor filled with zeros, with the same shape and dtype as the input tensor.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        input: The input tensor to match shape and dtype.
        location: The memory location for the tensor. Default is ``input.location``.

    Returns:
        LocalTensor: A new tensor filled with zeros

    Raises:
        TypeError: If input is not a LocalTensor

    Examples:
        Create a zero-filled tensor matching another tensor's shape and dtype: ::

            src = asc2.copy_in(x_gm, [0], [128])
            result = asc2.zeros_like(src)
    """
    check_type("input", input, LocalTensor)
    location = input.location if location is None else location
    return zeros(input.shape, input.dtype, location)


@require_jit
def zeros_acc(shape: Iterable[int], dtype: DataType, *, bias: Optional[LocalTensor] = None) -> LocalTensor:
    """
    Create a zero-initialized accumulator tensor in L0C memory for matrix multiplication.

    This tensor is specifically designed for use with :py:func:`matmul_acc` operations and is always located in
    ``TensorLocation.L0C``.

    The supported data type is: ``float32``.

    Args:
        shape: The shape of the accumulator tensor
        dtype: The data type of the accumulator
        bias: Optional initialization tensor (1D tensor in ``BT``). If provided, the accumulator will be initialized
              with this value instead of zeros. This is typically used for bias initialization in matrix multiplication.
              Supported dtypes: ``float16``, ``bfloat16``, or ``float32``. Tensors with ``float16`` or ``bfloat16`` are
              automatically promoted to ``float32``.

    Returns:
        LocalTensor: A new accumulator tensor in L0C memory (zero-initialized or initialized with the provided value)

    Raises:
        TypeError: If shape contains non-integer values
        RuntimeError: If shape contains non-positive values or bias has wrong shape/dtype


    Examples:
        Create a zero-initialized accumulator: ::

            acc = asc2.zeros_acc([64, 256], dtype=asc2.float32)
            for k in range(k_tiles):
                a_k = asc2.copy_in(a_gm, [0, k * 32], [64, 32], asc2.TensorLocation.L0A)
                b_k = asc2.copy_in(b_gm, [k * 32, 0], [32, 256], asc2.TensorLocation.L0B)
                asc2.matmul_acc(acc, a_k, b_k)
            asc2.copy_out(acc, c_gm, [0, 0])

        Create a bias-initialized accumulator: ::

            bias = asc2.copy_in(bias_gm, [0], [256], asc2.TensorLocation.BT)
            acc = asc2.zeros_acc([64, 256], dtype=asc2.float32, bias=bias)
            for k in range(k_tiles):
                a_k = asc2.copy_in(a_gm, [0, k * 32], [64, 32], asc2.TensorLocation.L0A)
                b_k = asc2.copy_in(b_gm, [k * 32, 0], [32, 256], asc2.TensorLocation.L0B)
                asc2.matmul_acc(acc, a_k, b_k)
            asc2.copy_out(acc, c_gm, [0, 0])
    """
    check_type("dtype", dtype, DataType)
    check_dtype("dtype", dtype, KT.float32)
    check_bias(bias, shape[1])
    shape = verify_shape(shape)
    if bias is not None:
        bias = cast_loc(bias, TensorLocation.BT).to_ir()
    ir_type = ir.get_asctile_LocalTensorType(list(shape), dtype.to_ir(), TensorLocation.L0C)
    handle = global_builder.get_ir_builder().create_asctile_AccumulatorOp(ir_type, bias)
    return cast_loc(LocalTensor(handle))


@overload
def cast(input: LocalTensor, dtype: DataType, round_mode: RoundMode = RoundMode.Default) -> LocalTensor:
    ...


@overload
def cast(input: RuntimeNumeric, dtype: DataType) -> PlainValue:
    ...


@require_jit
def cast(input: Union[LocalTensor, RuntimeNumeric], dtype: DataType,
         round_mode: RoundMode = RoundMode.Default) -> Union[LocalTensor, PlainValue]:
    """
    Cast a tensor or scalar value to a different data type.

    Creates a new tensor (or scalar) with the same shape but converted to the specified data type. If the input already
    has the target dtype, returns the input unchanged.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        input: The input tensor or scalar value to cast
        dtype: The target data type
        round_mode: The rounding mode for precision conversion (if ``input`` is a tensor). Supported values:
            ``RoundMode.Default`` (automatically infer rounding mode based on source and target types),
            ``RoundMode.NoRound`` (no rounding, truncate toward zero),
            ``RoundMode.Rint`` (round to nearest, ties to even),
            ``RoundMode.Floor`` (round toward negative infinity),
            ``RoundMode.Ceil`` (round toward positive infinity),
            ``RoundMode.Round`` (round half away from zero),
            ``RoundMode.Trunc`` (truncate toward zero),
            ``RoundMode.Odd`` (round to nearest odd).

    Returns:
        LocalTensor: A new tensor with the specified dtype (if input is a LocalTensor)
        PlainValue: A scalar value with the specified dtype (if input is a scalar)

    Raises:
        TypeError: If input is not a LocalTensor or numeric value, or dtype is not a DataType

    Note:
        This function is also available as the ``.to()`` method on tensors: ``result.to(dtype)``.

    Examples:
        Cast a tensor from float32 to float16: ::

            input = asc2.copy_in(x_gm, [0], [128])
            result_fp16 = asc2.cast(input, asc2.float16)

        Cast with explicit rounding mode: ::

            input = asc2.copy_in(x_gm, [0], [128])
            result_int32 = asc2.cast(input, asc2.int32, round_mode=asc2.RoundMode.Floor)

        Cast using the .to() method (equivalent): ::

            input = asc2.copy_in(x_gm, [0], [128])
            result_fp16 = input.to(asc2.float16)

        Cast a scalar value: ::

            scalar_fp16 = asc2.cast(3.14, asc2.float16)

        Chain multiple casts for quantization: ::

            acc = asc2.zeros_acc([64, 128], dtype=asc2.float32)
            # ... accumulate matmul results ...
            result_fp16 = acc.to(asc2.float16)
            asc2.copy_out(result_fp16, out_gm, [0, 0])
    """
    check_type("input", input, (LocalTensor, RuntimeNumeric))
    check_type("dtype", dtype, DataType)
    if not isinstance(input, LocalTensor):
        if round_mode != RoundMode.Default:
            raise RuntimeError("'round_mode' argument cannot be used with scalar input")
        return materialize_ir_value(input, dtype)
    if input.dtype == dtype:
        return input
    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), dtype.to_ir())
    handle = global_builder.get_ir_builder().create_asctile_CastOp(ir_type, input.to_ir(), round_mode)
    return cast_loc(LocalTensor(handle))


@require_jit
def concat(*inputs: LocalTensor) -> LocalTensor:
    """
    Concatenate tensors along the first dimension.

    All input tensors must have the same shape except for the first dimension, and must have the same data type.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        inputs: Two or more tensors to concatenate

    Returns:
        LocalTensor: A new tensor that is the concatenation of all input tensors along the first dimension

    Raises:
        TypeError: If any input is not a LocalTensor
        RuntimeError: If no inputs are provided, shapes are incompatible, dtypes don't match,
            or tensor dtype size does not fit an integer number of bytes

    Examples:
        Concatenate two tensors along the first dimension: ::

            input_a = asc2.copy_in(x_gm, [0, 0], [64, 32])
            input_b = asc2.copy_in(y_gm, [64, 0], [64, 32])
            result = asc2.concat(input_a, input_b)  # shape: [128, 32]

        Concatenate multiple tensors: ::

            tensors = [asc2.copy_in(x_gm, [0, 0], [32, 16]), asc2.copy_in(x_gm, [32, 0], [16, 16]),
                       asc2.copy_in(x_gm, [64, 0], [8, 16]), asc2.maximum(input_a, input_b)]
            result = asc2.concat(*tensors)  # shape: [120, 16]
    """
    if not inputs or not all(isinstance(inp, LocalTensor) for inp in inputs):
        raise TypeError("All input arguments must be tensors")
    same_shape = inputs[0].shape[1:]
    if not all(inp.shape[1:] == same_shape for inp in inputs):
        raise RuntimeError("All tensors must have the same shape except their first dimension")
    dtype = inputs[0].dtype
    if not all(inp.dtype == dtype for inp in inputs):
        raise RuntimeError("All tensors must have the same dtype")
    try:
        dtype.sizeof()
    except ValueError:
        raise RuntimeError("LocalTensor dtype size must fit an integer number of bytes")
    result_shape = [sum(inp.shape[0] for inp in inputs), *same_shape]
    inputs = [cast_loc(tensor, TensorLocation.UB) for tensor in inputs]
    ir_type = ir.get_asctile_LocalTensorType(result_shape, dtype.to_ir(), TensorLocation.UB)
    handle = global_builder.get_ir_builder().create_tensor_ConcatOp(ir_type, 0, [inp.to_ir() for inp in inputs])
    return cast_loc(LocalTensor(handle))
