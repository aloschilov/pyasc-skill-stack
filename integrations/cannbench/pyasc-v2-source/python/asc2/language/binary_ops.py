# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Callable, Optional, Tuple, TypeVar, Union

from asc._C import ir
from asc.common.compat import isinstance
from asc.language.core.dtype import DataType, KnownTypes as KT
from asc.language.core.ir_value import IRHandle, RuntimeInt, RuntimeNumeric
from asc.language.core.utils import global_builder, require_jit

from .local_tensor import BinaryOperandTypeError, LocalTensor, bind_tensor_method
from .tensor_location import TensorLocation
from .utils import cast_tensor_location as cast_loc, check_bias, create_tile, infer_common_dtype, infer_common_shape
from .validation import check_dtype, check_type

T = TypeVar("T")

common_support_dtypes = (KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32)
compare_support_dtypes = (KT.int8, KT.int16, KT.int32, KT.float16, KT.bfloat16, KT.float32)
bitwise_support_dtypes = (KT.int8, KT.int16, KT.int32, KT.int64)


def check_numeric_tensor_like(name: str, value: Any, support_dtypes: Tuple[DataType, ...]) -> None:
    check_type(name, value, (LocalTensor, RuntimeNumeric), BinaryOperandTypeError)
    if isinstance(value, LocalTensor):
        check_dtype(name, value, support_dtypes)


def unify_tensors(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric],
                  support_dtypes: Tuple[DataType, ...]) -> Tuple[LocalTensor, LocalTensor]:
    for name, value in ("input", input), ("other", other):
        check_numeric_tensor_like(name, value, support_dtypes)
    if not isinstance(input, LocalTensor) and not isinstance(other, LocalTensor):
        raise BinaryOperandTypeError(f"At least one operand must be tensor, got {type(input)} and {type(other)}")
    result_dtype = infer_common_dtype(input, other)
    result_shape = infer_common_shape(input, other)
    input = create_tile(input, result_dtype, result_shape, TensorLocation.UB)
    other = create_tile(other, result_dtype, result_shape, TensorLocation.UB)
    return input, other


def op_binary_impl(
    input: Union[LocalTensor, RuntimeNumeric],
    other: Union[LocalTensor, RuntimeNumeric],
    build_int: Callable[..., IRHandle],
    build_float: Callable[..., IRHandle],
    support_dtypes: Tuple[DataType, ...],
) -> LocalTensor:
    input, other = unify_tensors(input, other, support_dtypes)
    result_dtype = input.dtype
    if result_dtype.is_int():
        handle = build_int(input.to_ir(), other.to_ir())
    elif result_dtype.is_float():
        handle = build_float(input.to_ir(), other.to_ir())
    else:
        raise RuntimeError(f"Unexpected result tensor dtype: {result_dtype}")
    return cast_loc(LocalTensor(handle))


def op_compare_impl(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric],
                    pred_float: ir.CmpFPredicate, pred_int: ir.CmpIPredicate) -> LocalTensor:
    input, other = unify_tensors(input, other, compare_support_dtypes)
    builder = global_builder.get_ir_builder()
    if input.dtype.is_float():
        build = builder.create_arith_CmpFOp
        pred = pred_float
    else:
        build = builder.create_arith_CmpIOp
        pred = pred_int
    handle = build(pred, input.to_ir(), other.to_ir())
    return cast_loc(LocalTensor(handle))


def op_bitwise_impl(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric],
                    build: Callable[..., IRHandle]) -> LocalTensor:
    input, other = unify_tensors(input, other, bitwise_support_dtypes)
    handle = build(input.to_ir(), other.to_ir())
    return cast_loc(LocalTensor(handle))


def set_docstring(name: str, support_dtypes: Tuple[DataType, ...], rhs_scalar_only: bool = False) -> Callable[[T], T]:
    dtypes_str = ", ".join(f"``{dtype}``" for dtype in support_dtypes)
    other_type = "scalar" if rhs_scalar_only else "tensor or scalar"
    examples = "" if rhs_scalar_only else f"""
        Compute the {name} between elements of two tensors: ::

            input = asc2.copy_in(tensor_a, [1, 4], [32, 16])
            other = asc2.copy_in(tensor_b, [2, 8], [32, 16])
            result = asc2.{{fn_name}}(input, other)

    """
    examples += f"""
        Compute the {name} of tensor elements and a given scalar value: ::

            input = asc2.copy_in(tensor, [0, 0], [32, 16])
            result = asc2.{{fn_name}}(input, 2)
    """

    def decorator(fn: T) -> T:
        doc = f"""
    Computes the element-wise {name} of ``input`` and ``other``.

    The supported data types for the inputs are: {dtypes_str}.

    Args:
        input: The left operand (tensor or scalar)
        other: The right operand ({other_type})

    Returns:
        LocalTensor: The result of {name}

    Raises:
        RuntimeError: If neither operand is a ``LocalTensor``

    Note:
        At least one of input operands must be ``LocalTensor``.
        Operands are automatically cast to a common data type and broadcast to a common shape.
        When one operand is a ``LocalTensor`` and the other is a scalar, the tensor's dtype takes precedence.
        Inputs must either have the same shapes or be broadcastable according to NumPy broadcasting rules.

    Examples:
        {examples}
        """
        fn.__doc__ = doc.format(fn_name=fn.__name__)
        return fn

    return decorator


@bind_tensor_method(name="__eq__", binary_op="==")
@require_jit
@set_docstring("'equality' comparison", compare_support_dtypes)
def equal(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    return op_compare_impl(input, other, ir.CmpFPredicate.OEQ, ir.CmpIPredicate.eq)


@bind_tensor_method(name="__ne__", binary_op="!=")
@require_jit
@set_docstring("'inequality' comparison", compare_support_dtypes)
def not_equal(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    return op_compare_impl(input, other, ir.CmpFPredicate.ONE, ir.CmpIPredicate.ne)


@bind_tensor_method(name="__gt__", binary_op=">")
@require_jit
@set_docstring("'greater' comparison", compare_support_dtypes)
def greater(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    return op_compare_impl(input, other, ir.CmpFPredicate.OGT, ir.CmpIPredicate.sgt)


@bind_tensor_method(name="__ge__", binary_op=">=")
@require_jit
@set_docstring("'greater or equal' comparison", compare_support_dtypes)
def greater_equal(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    return op_compare_impl(input, other, ir.CmpFPredicate.OGE, ir.CmpIPredicate.sge)


@bind_tensor_method(name="__lt__", binary_op="<")
@require_jit
@set_docstring("'less' comparison", compare_support_dtypes)
def less(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    return op_compare_impl(input, other, ir.CmpFPredicate.OLT, ir.CmpIPredicate.slt)


@bind_tensor_method(name="__le__", binary_op="<=")
@require_jit
@set_docstring("'less or equal' comparison", compare_support_dtypes)
def less_equal(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    return op_compare_impl(input, other, ir.CmpFPredicate.OLE, ir.CmpIPredicate.sle)


@bind_tensor_method(name="__add__", binary_op="+")
@require_jit
@set_docstring("addition", common_support_dtypes)
def add(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_AddIOp, builder.create_arith_AddFOp, common_support_dtypes)


@bind_tensor_method(name="__sub__", binary_op="-")
@require_jit
@set_docstring("subtraction", common_support_dtypes)
def sub(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_SubIOp, builder.create_arith_SubFOp, common_support_dtypes)


@bind_tensor_method(name="__mul__", binary_op="*")
@require_jit
@set_docstring("multiplication", common_support_dtypes)
def mul(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_MulIOp, builder.create_arith_MulFOp, common_support_dtypes)


@bind_tensor_method(name="__truediv__", binary_op="/")
@require_jit
@set_docstring("division", (KT.int16, KT.int32, KT.int64, KT.float16, KT.float32))
def div(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_DivSIOp, builder.create_arith_DivFOp,
                          (KT.int16, KT.int32, KT.int64, KT.float16, KT.float32))


@require_jit
@set_docstring("maximum", common_support_dtypes)
def maximum(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_MaxSIOp, builder.create_arith_MaximumFOp,
                          common_support_dtypes)


@require_jit
@set_docstring("minimum", common_support_dtypes)
def minimum(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_MinSIOp, builder.create_arith_MinimumFOp,
                          common_support_dtypes)


@bind_tensor_method(name="__and__", binary_op="&")
@require_jit
@set_docstring("AND (bitwise)", bitwise_support_dtypes)
def bitwise_and(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    return op_bitwise_impl(input, other, global_builder.get_ir_builder().create_arith_AndIOp)


@bind_tensor_method(name="__or__", binary_op="|")
@require_jit
@set_docstring("OR (bitwise)", bitwise_support_dtypes)
def bitwise_or(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    return op_bitwise_impl(input, other, global_builder.get_ir_builder().create_arith_OrIOp)


@bind_tensor_method(name="__xor__", binary_op="^")
@require_jit
@set_docstring("XOR (bitwise)", bitwise_support_dtypes)
def bitwise_xor(input: Union[LocalTensor, RuntimeNumeric], other: Union[LocalTensor, RuntimeNumeric]) -> LocalTensor:
    return bitwise_or(input, other) & (~bitwise_and(input, other))


@bind_tensor_method(name="__lshift__", binary_op="<<")
@require_jit
@set_docstring("left shift (bitwise)", (KT.int16, KT.int32, KT.int64), rhs_scalar_only=True)
def left_shift(input: LocalTensor, other: RuntimeInt) -> LocalTensor:
    input, other = unify_tensors(input, other, (KT.int16, KT.int32, KT.int64))
    handle = global_builder.get_ir_builder().create_arith_ShLIOp(input.to_ir(), other.to_ir())
    return cast_loc(LocalTensor(handle))


@bind_tensor_method(name="__rshift__", binary_op=">>")
@require_jit
@set_docstring("right shift (bitwise)", (KT.int16, KT.int32, KT.int64), rhs_scalar_only=True)
def right_shift(input: LocalTensor, other: RuntimeInt) -> LocalTensor:
    input, other = unify_tensors(input, other, (KT.int16, KT.int32, KT.int64))
    handle = global_builder.get_ir_builder().create_arith_ShRSIOp(input.to_ir(), other.to_ir())
    return cast_loc(LocalTensor(handle))


def check_matmul_arguments(input: LocalTensor, other: LocalTensor, hf32: bool) -> None:
    for name, value, loc in ("input", input, TensorLocation.L0A), ("other", other, TensorLocation.L0B):
        check_type(name, value, LocalTensor, BinaryOperandTypeError)
        check_dtype(name, value, (KT.float16, KT.bfloat16, KT.float32))
    if input.dtype != other.dtype:
        raise RuntimeError(f"Input tensors must have the same types, got {input.dtype} and {other.dtype}")
    if len(input.shape) != 2 or len(other.shape) != 2:
        raise RuntimeError(f"Input tensors must have two dims, got {len(input.shape)} and {len(other.shape)}")
    if input.shape[1] != other.shape[0]:
        raise RuntimeError(f"Input tensors have incompatible shapes: {input.shape}, {other.shape}")
    check_type("hf32", hf32, bool)
    if hf32 and input.dtype != KT.float32:
        raise RuntimeError("HF32 mode can only be set when input tensor dtype is float32")


@bind_tensor_method(name="__matmul__", binary_op="@")
@require_jit
def matmul(input: LocalTensor, other: LocalTensor, bias: Optional[LocalTensor] = None, *,
           hf32: bool = False) -> LocalTensor:
    """
    Computes the matrix multiplication of ``input`` and ``other`` with optional ``bias``.

    Args:
        input: The left operand (2D tensor in ``L0A``)
        other: The right operand (2D tensor in ``L0B``)
        bias: Optional bias tensor (1D tensor in ``BT``)
        hf32: Enable the rounding to HF32 for input tensors with ``float32`` dtype

    Returns:
        LocalTensor: The result of the matrix multiplication (2D tensor in ``L0C``)

    Raises:
        TypeError: If input or other is not a LocalTensor
        RuntimeError: If input tensors are not 2D, have incompatible shapes, unsupported dtype,
            or bias has wrong shape/dtype

    Note:
        Input tensors must have either ``float16``, ``bfloat16``, or ``float32`` data type and compatible shapes.
        Result tensor type is always ``float32``. Bias must be a 1D tensor of ``float16``, ``bfloat16``, or ``float32``
        with shape matching the last dimension of the output.
        Bias with ``float16`` or ``bfloat16`` dtype is automatically promoted to ``float32`` to match the result type.

    Examples:
        Basic matrix multiplication using operator syntax: ::

            a = asc2.copy_in(a_gm, [0, 0], [64, 128], asc2.TensorLocation.L0A)
            b = asc2.copy_in(b_gm, [0, 0], [128, 256], asc2.TensorLocation.L0B)
            c = a @ b  # result shape: [64, 256], location: L0C

        Matrix multiplication with bias: ::

            a = asc2.copy_in(a_gm, [0, 0], [64, 128], asc2.TensorLocation.L0A)
            b = asc2.copy_in(b_gm, [0, 0], [128, 256], asc2.TensorLocation.L0B)
            bias = asc2.copy_in(bias_gm, [0], [256], asc2.TensorLocation.BT)
            c = asc2.matmul(a, b, bias)

        Matrix multiplication with HF32 mode (for float32 inputs): ::

            a = asc2.copy_in(a_gm, [0, 0], [32, 64], asc2.TensorLocation.L0A)
            b = asc2.copy_in(b_gm, [0, 0], [64, 64], asc2.TensorLocation.L0B)
            c = asc2.matmul(a, b, hf32=True)

        Store result to global memory: ::

            c = a @ b
            asc2.copy_out(c, c_gm, [0, 0])
    """
    check_matmul_arguments(input, other, hf32)
    check_bias(bias, other.shape[1])
    input = cast_loc(input, TensorLocation.L0A)
    other = cast_loc(other, TensorLocation.L0B)
    if bias is not None:
        bias = cast_loc(bias, TensorLocation.BT).to_ir()
    ir_type = ir.get_asctile_LocalTensorType([input.shape[0], other.shape[1]], KT.float32.to_ir(), TensorLocation.L0C)
    handle = global_builder.get_ir_builder().create_asctile_MatmulOp(ir_type, input.to_ir(), other.to_ir(), bias, hf32)
    return cast_loc(LocalTensor(handle))


@require_jit
def matmul_acc(acc: LocalTensor, input: LocalTensor, other: LocalTensor, *, hf32: bool = False) -> None:
    """
    Computes the matrix multiplication of ``input`` and ``other`` and accumulates the result into ``acc``.

    This function performs in-place accumulation, adding the result of ``input @ other`` to the existing accumulator
    values. Use :py:func:`asc2.zeros_acc` to create an accumulator. For simple matrix multiplication without
    accumulation, use :py:func:`matmul` which returns a new tensor.

    **Rationale:** Ascend's Cube units operate on a dedicated L0C accumulator register where the accumulator and matmul
    destination are the same physical entity—the hardware accumulates in-place as part of the matmul operation itself.
    Unlike general-purpose memory (UB, L1), L0C is a specialized register file designed for this exact use case. This
    destination-passing style makes the hardware behavior explicit: you create an accumulator with ``zeros_acc``,
    reuse it across multiple matmul operations, then read the final result. While other frameworks may use functional
    style (e.g. ``acc = matmul(acc, a, b)``), that approach would either require implicit copies from L0C (defeating
    the purpose of the dedicated accumulator) or obscure the fact that accumulation happens in specialized hardware.

    Args:
        acc: Accumulator tensor (2D tensor in ``L0C``), must be created with :py:func:`asc2.zeros_acc`
        input: The left operand (2D tensor in ``L0A``)
        other: The right operand (2D tensor in ``L0B``)
        hf32: Enable the rounding to HF32 for input tensors with ``float32`` dtype

    Raises:
        TypeError: If acc, input, or other is not a LocalTensor
        RuntimeError: If input tensors are not 2D, have incompatible shapes, unsupported dtype,
            or accumulator has wrong shape/dtype

    Note:
        Input tensors must have either ``float16``, ``bfloat16``, or ``float32`` data type and compatible shapes.
        Accumulator tensor type is always ``float32``.

    Examples:
        Accumulate multiple matrix multiplications (e.g., for K-tiled matmul): ::

            acc = asc2.zeros_acc([64, 256], dtype=asc2.float32)
            for k in range(k_tiles):
                a_k = asc2.copy(a_l1, [0, k * 32], [64, 32], asc2.TensorLocation.L0A)
                b_k = asc2.copy(b_l1, [k * 32, 0], [32, 256], asc2.TensorLocation.L0B)
                asc2.matmul_acc(acc, a_k, b_k)
            asc2.copy_out(acc, c_gm, [0, 0])

        Accumulate with bias initialization: ::

            bias = asc2.copy(bias_l1, [0], [256], asc2.TensorLocation.BT)
            acc = asc2.zeros_acc([64, 256], dtype=asc2.float32, bias=bias)
            for k in range(k_tiles):
                a_k = asc2.copy(a_l1, [0, k * 32], [64, 32], asc2.TensorLocation.L0A)
                b_k = asc2.copy(b_l1, [k * 32, 0], [32, 256], asc2.TensorLocation.L0B)
                asc2.matmul_acc(acc, a_k, b_k)
            asc2.copy_out(acc, c_gm, [0, 0])

        Accumulate with HF32 mode (for float32 inputs): ::

            acc = asc2.zeros_acc([32, 64], dtype=asc2.float32)
            asc2.matmul_acc(acc, a_l0a, b_l0b, hf32=True)
    """
    check_type("acc", acc, LocalTensor)
    check_dtype("acc", acc, KT.float32)
    check_matmul_arguments(input, other, hf32)
    if len(acc.shape) != 2:
        raise RuntimeError(f"Accumulation tensor must have two dims, got {len(acc.shape)}")
    acc = cast_loc(acc, TensorLocation.L0C)
    input = cast_loc(input, TensorLocation.L0A)
    other = cast_loc(other, TensorLocation.L0B)
    global_builder.get_ir_builder().create_asctile_MatmulAccOp(acc.to_ir(), input.to_ir(), other.to_ir(), hf32)
