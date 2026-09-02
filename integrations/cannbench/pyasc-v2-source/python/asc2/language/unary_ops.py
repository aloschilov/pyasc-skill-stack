# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable, Optional, Tuple, TypeVar, Union, overload

from asc._C import ir
from asc.language.core.dtype import DataType, KnownTypes as KT
from asc.language.core.ir_value import PlainValue, RuntimeFloat, RuntimeNumeric, IRHandle, materialize_ir_value as _mat
from asc.language.core.utils import global_builder, require_jit

from .local_tensor import LocalTensor, bind_tensor_method
from .tensor_location import TensorLocation
from .utils import cast_tensor_location as cast_loc
from .validation import check_dtype, check_runtime_float, check_type

T = TypeVar("T")


def op_unary_impl(
    input: Union[LocalTensor, RuntimeNumeric], build_float: Optional[Callable[..., IRHandle]] = None,
    build_int: Optional[Callable[..., IRHandle]] = None, support_dtypes: Optional[Tuple[DataType]] = None,
    support_scalar: bool = False, support_loc: Tuple[TensorLocation, ...] = (TensorLocation.UB, )
) -> Union[LocalTensor, PlainValue]:
    constraint = Union[LocalTensor, RuntimeNumeric] if support_scalar else LocalTensor
    check_type("input", input, constraint)
    is_tensor = isinstance(input, LocalTensor)
    if is_tensor:
        if support_dtypes is not None:
            check_dtype("input", input, support_dtypes)
        input = cast_loc(input, support_loc[0] if len(support_loc) == 1 else TensorLocation.Auto)
    input = input if is_tensor else _mat(input, KT.float32)  # TODO: infer dtype using builders availability
    dtype = input.dtype
    if dtype.is_float() and build_float is not None:
        handle = build_float(input.to_ir())
    elif dtype.is_signed() and build_int is not None:
        handle = build_int(input.to_ir())
    else:
        raise RuntimeError(f"Input tensor dtype is not supported: {dtype}")
    return cast_loc(LocalTensor(handle)) if is_tensor else PlainValue(handle)


def set_docstring(name: str, support_dtypes: Tuple[DataType], support_scalar: bool = False) -> Callable[[T], T]:
    dtypes_str = ", ".join(f"``{dtype}``" for dtype in support_dtypes)
    tensor_info = "tensor or scalar" if support_scalar else "tensor"
    examples = f"""
        Compute the element-wise {name} of all tensor elements: ::

            input = asc2.copy_in(tensor, [0, 0], [128, 256])
            result = asc2.{{fn_name}}(input)
        """
    if support_scalar:
        examples += f"""
        Compute the {name} of given scalar value: ::

            result = asc2.{{fn_name}}(1.0)
        """

    def decorator(fn: T) -> T:
        doc = f"""
    Computes the element-wise {name} of ``input``.

    The supported data types for the input are: {dtypes_str}.

    Args:
        input: The input value ({tensor_info})

    Returns:
        LocalTensor: The result tensor

    Raises:
        RuntimeError: If the input dtype is not supported for this operation

    Examples:
        {examples}
        """
        fn.__doc__ = doc.format(fn_name=fn.__name__)
        return fn

    return decorator


@bind_tensor_method
@require_jit
@set_docstring("cosine", support_dtypes=(KT.float16, KT.float32))
def cos(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_CosOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("sine", support_dtypes=(KT.float16, KT.float32))
def sin(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_SinOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("tangent", support_dtypes=(KT.float16, KT.float32))
def tan(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_TanOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("hyperbolic sine", support_dtypes=(KT.float16, KT.float32))
def sinh(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_SinhOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("hyperbolic cosine", support_dtypes=(KT.float16, KT.float32))
def cosh(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_CoshOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("hyperbolic tangent", support_dtypes=(KT.float16, KT.float32))
def tanh(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_TanhOp, support_dtypes=(KT.float16, KT.float32))


@overload
def exp(input: LocalTensor) -> LocalTensor:
    ...


@overload
def exp(input: RuntimeFloat) -> PlainValue:
    ...


@bind_tensor_method
@require_jit
@set_docstring("exponential", support_dtypes=(KT.float16, KT.float32), support_scalar=True)
def exp(input: Union[LocalTensor, RuntimeFloat]) -> Union[LocalTensor, PlainValue]:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_ExpOp, support_dtypes=(KT.float16, KT.float32),
                         support_scalar=True)


@bind_tensor_method
@require_jit
@set_docstring("natural logarithm", support_dtypes=(KT.float16, KT.float32))
def log(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_LogOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("logarithm (base 2)", support_dtypes=(KT.float16, KT.float32))
def log2(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_Log2Op, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("floor rounding", support_dtypes=(KT.float16, KT.float32))
def floor(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_FloorOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("ceil rounding", support_dtypes=(KT.float16, KT.float32))
def ceil(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_CeilOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("absolute value", support_dtypes=(KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.float32))
def abs(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_AbsFOp, builder.create_math_AbsIOp,
                         support_dtypes=(KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.float32))


@overload
def erf(input: LocalTensor) -> LocalTensor:
    ...


@overload
def erf(input: RuntimeFloat) -> PlainValue:
    ...


@bind_tensor_method
@require_jit
@set_docstring("error function", support_dtypes=(KT.float16, KT.float32), support_scalar=True)
def erf(input: Union[LocalTensor, RuntimeFloat]) -> Union[LocalTensor, PlainValue]:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_ErfOp, support_dtypes=(KT.float16, KT.float32),
                         support_scalar=True)


@bind_tensor_method
@require_jit
@set_docstring("exponential (base 2)", support_dtypes=(KT.float16, KT.float32))
def exp2(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_Exp2Op, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("inverse square root", support_dtypes=(KT.float16, KT.float32))
def rsqrt(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_RsqrtOp, support_dtypes=(KT.float16, KT.float32))


@overload
def sqrt(input: LocalTensor) -> LocalTensor:
    ...


@overload
def sqrt(input: RuntimeFloat) -> PlainValue:
    ...


@bind_tensor_method
@require_jit
@set_docstring("square root", support_dtypes=(KT.float16, KT.float32), support_scalar=True)
def sqrt(input: Union[LocalTensor, RuntimeFloat]) -> Union[LocalTensor, PlainValue]:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_SqrtOp, support_dtypes=(KT.float16, KT.float32),
                         support_scalar=True)


@bind_tensor_method
@require_jit
@set_docstring("ReLU value", support_dtypes=(KT.float16, KT.float32))
def relu(input: LocalTensor) -> LocalTensor:
    build = lambda handle: global_builder.get_ir_builder().create_asctile_ReluOp(handle.get_type(), handle)
    return op_unary_impl(input, build_float=build, support_dtypes=(KT.float16, KT.float32),
                         support_loc=(TensorLocation.UB, TensorLocation.L0C))


@bind_tensor_method(name="__neg__", unary_op="-")
@require_jit
@set_docstring("negation", support_dtypes=(KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32))
def negative(input: LocalTensor) -> LocalTensor:
    check_dtype("input", input, (KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32))
    return input * (-1)


@bind_tensor_method(name="__invert__", unary_op="~")
@require_jit
@set_docstring("NOT (bitwise)", support_dtypes=(KT.int8, KT.int16, KT.int32, KT.int64))
def bitwise_not(input: LocalTensor) -> LocalTensor:
    build = lambda handle: global_builder.get_ir_builder().create_asctile_BitwiseNotOp(handle.get_type(), handle)
    return op_unary_impl(input, build_int=build, support_dtypes=(KT.int8, KT.int16, KT.int32, KT.int64))


@bind_tensor_method
@require_jit
def softmax(input: LocalTensor) -> LocalTensor:
    """
    Computes the row-wise softmax of ``input``.

    For 2D tensors, softmax is applied independently along the last dimension for each row.
    For 1D tensors, softmax is applied over all elements.

    The supported data types for the input are: ``float16``, ``float32``.

    Args:
        input: The input tensor (1D or 2D)

    Returns:
        LocalTensor: The result tensor with the same shape as input

    Raises:
        RuntimeError: If the input dtype is not supported or input has more than 2 dimensions

    Examples:
        Compute row-wise softmax for a 2D tensor: ::

            input = asc2.copy_in(x_gm, [0, 0], [64, 1024])
            result = asc2.softmax(input)  # softmax applied independently to each of 64 rows

        Compute softmax for a 1D tensor: ::

            input = asc2.copy_in(x_gm, [0], [1024])
            result = asc2.softmax(input)  # softmax applied over all 1024 elements
    """
    check_type("input", input, LocalTensor)
    check_dtype("input", input, (KT.float16, KT.float32))
    if len(input.shape) > 2:
        raise RuntimeError("Tensor dimensionality greater than two is not supported")
    input = cast_loc(input, TensorLocation.UB)
    handle = global_builder.get_ir_builder().create_asctile_SoftmaxOp(input.to_ir().get_type(), input.to_ir())
    return cast_loc(LocalTensor(handle))


@require_jit
def rms_norm(input: LocalTensor, gamma: LocalTensor, epsilon: RuntimeFloat) -> LocalTensor:
    """
    Computes Root Mean Square Layer Normalization of ``input``.

    RMSNorm normalizes the input by the root mean square and scales by learnable parameters ``gamma``.
    This is commonly used in transformer architectures as an alternative to LayerNorm.

    The supported data types for the inputs are: ``float16``, ``float32``.

    Args:
        input: The input tensor to normalize (1D or 2D)
        gamma: The scale parameter tensor (1D, same length as last dimension of input)
        epsilon: Small constant added for numerical stability

    Returns:
        LocalTensor: The normalized tensor with same shape as input

    Raises:
        TypeError: If input or gamma is not a LocalTensor
        RuntimeError: If input dtype is not supported, input has more than 2 dimensions,
            or gamma dtype is not supported

    Examples:
        Apply RMSNorm to a 2D tensor: ::

            input = asc2.copy_in(x_gm, [0, 0], [32, 128])
            gamma = asc2.copy_in(gamma_gm, [0], [128])
            output = asc2.rms_norm(input, gamma, 1e-5)

        Apply RMSNorm to a 1D tensor: ::

            input = asc2.copy_in(x_gm, [0], [128])
            gamma = asc2.copy_in(gamma_gm, [0], [128])
            output = asc2.rms_norm(input, gamma, 1e-6)
    """
    for name, value in ("input", input), ("gamma", gamma):
        check_type(name, value, LocalTensor)
        check_dtype(name, value, (KT.float16, KT.float32))
    check_runtime_float("epsilon", epsilon)
    if len(input.shape) > 2:
        raise RuntimeError("Tensor dimensionality greater than two is not supported.")
    input = cast_loc(input, TensorLocation.UB)
    gamma = cast_loc(gamma, TensorLocation.UB)
    handle = global_builder.get_ir_builder().create_asctile_RmsNormOp(input.to_ir().get_type(), input.to_ir(),
                                                                      gamma.to_ir(),
                                                                      _mat(epsilon, input.dtype).to_ir())
    return cast_loc(LocalTensor(handle))


@require_jit
def layer_norm(input: LocalTensor, gamma: LocalTensor, beta: LocalTensor, epsilon: RuntimeFloat,
               output_rstd: bool = True) -> Tuple[LocalTensor, LocalTensor, LocalTensor]:
    """
    Computes Layer Normalization of ``input``.

    LayerNorm normalizes the input by subtracting the mean and dividing by the standard deviation,
    then scales by learnable parameter ``gamma`` and shifts by learnable parameter ``beta``.
    This is commonly used in transformer architectures.

    The supported data types for the inputs are: ``float16``, ``bfloat16``, ``float32``.

    Args:
        input: The input tensor to normalize with shape [A, R] for 2D or [R] for 1D
        gamma: The scale parameter tensor with shape [R] (same length as last dimension of input)
        beta: The shift parameter tensor with shape [R] (same length as last dimension of input)
        epsilon: Small constant added for numerical stability
        output_rstd: If True (default), the third output is the reciprocal of standard deviation (rstd).
            If False, the third output is the variance. Note: on C220 architecture,
            this parameter is ignored and rstd is always returned.

    Returns:
        Tuple[LocalTensor, LocalTensor, LocalTensor]: A tuple containing:
            - output: The normalized tensor with same shape and dtype as input
            - mean: The mean tensor with shape [A] for 2D input or [1] for 1D input (always float32)
            - rstd or variance: If output_rstd=True, returns reciprocal of standard deviation;
              otherwise returns variance. Shape [A] for 2D input or [1] for 1D input (always float32)

    Raises:
        TypeError: If input, gamma, or beta is not a LocalTensor
        RuntimeError: If input dtype is not supported, input has more than 2 dimensions,
            or gamma/beta dtype is not supported

    Examples:
        Apply LayerNorm to a 2D tensor: ::

            input = asc2.copy_in(x_gm, [0, 0], [32, 128])
            gamma = asc2.copy_in(gamma_gm, [0], [128])
            beta = asc2.copy_in(beta_gm, [0], [128])
            output, mean, rstd = asc2.layer_norm(input, gamma, beta, 1e-5)

        Apply LayerNorm to a 1D tensor with variance output: ::

            input = asc2.copy_in(x_gm, [0], [128])
            gamma = asc2.copy_in(gamma_gm, [0], [128])
            beta = asc2.copy_in(beta_gm, [0], [128])
            output, mean, variance = asc2.layer_norm(input, gamma, beta, 1e-6, output_rstd=False)
    """
    for name, value in ("input", input), ("gamma", gamma), ("beta", beta):
        check_type(name, value, LocalTensor)
        check_dtype(name, value, (KT.float16, KT.bfloat16, KT.float32))
    check_runtime_float("epsilon", epsilon)
    if input.rank > 2:
        raise RuntimeError("Tensor dimensionality greater than two is not supported.")
    input = cast_loc(input, TensorLocation.UB)
    gamma = cast_loc(gamma, TensorLocation.UB)
    beta = cast_loc(beta, TensorLocation.UB)
    input_type = input.to_ir().get_type()
    mean_var_shape = [input.shape[0]] if input.rank == 2 else [1]
    builder = global_builder.get_ir_builder()
    mean_var_type = ir.get_asctile_LocalTensorType(mean_var_shape, builder.get_f32_type())
    op = builder.create_asctile_LayerNormOp(input_type, mean_var_type, mean_var_type, input.to_ir(), gamma.to_ir(),
                                            beta.to_ir(),
                                            _mat(epsilon, KT.float32).to_ir(), output_rstd)
    return tuple(cast_loc(LocalTensor(op.get_result(i))) for i in range(3))
