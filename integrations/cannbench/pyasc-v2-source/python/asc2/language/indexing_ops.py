# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from contextlib import contextmanager
from typing import Any, Generator, Iterable, Optional, Union, overload

from asc.language.core.dtype import KnownTypes as KT
from asc.language.core.ir_value import RuntimeInt, RuntimeNumeric, materialize_ir_value as _mat
from asc.language.core.utils import global_builder, require_jit

from .local_tensor import LocalTensor
from .tensor_location import TensorLocation
from .utils import cast_tensor_location as cast_loc, create_tile, infer_common_dtype
from .validation import check_dtype, check_runtime_int, check_type, verify_runtime_ints


@require_jit
def where(mask: LocalTensor, src0: Union[LocalTensor, RuntimeNumeric], src1: Union[LocalTensor,
                                                                                   RuntimeNumeric]) -> LocalTensor:
    """
    Select elements from two sources based on a mask.

    For each element, returns the corresponding element from ``src0`` if the mask element is true (non-zero),
    otherwise returns the element from ``src1``.

    The supported data types for ``src0`` and ``src1``: ``int16``, ``int32``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        mask: A boolean tensor specifying which elements to select
        src0: The source for elements where mask is true (tensor or scalar)
        src1: The source for elements where mask is false (tensor or scalar)

    Returns:
        LocalTensor: A tensor with elements selected from ``src0`` or ``src1`` based on the mask

    Raises:
        TypeError: If mask is not a ``LocalTensor``, or if ``src0`` or ``src1`` is not a ``LocalTensor`` or scalar
        RuntimeError: If mask dtype is not ``int1``, or if ``src0`` or ``src1`` dtype is not supported

    Note:
        At least one of ``src0`` or ``src1`` must be a tensor with the same shape as the mask.
        Scalars are broadcast to the mask shape.

    Examples:
        Select elements from two tensors based on a mask: ::

            mask = tensor_a > tensor_b
            result = asc2.where(mask, tensor_a, tensor_b)

        Select elements from a tensor or a scalar based on a mask: ::

            mask = tensor > 0
            result = asc2.where(mask, tensor, 0)
    """
    check_type("mask", mask, LocalTensor)
    check_dtype("mask", mask, KT.int1)
    mask = cast_loc(mask, TensorLocation.UB)
    for name, value in ("src0", src0), ("src1", src1):
        check_type(name, value, (LocalTensor, RuntimeNumeric))
        check_dtype(name, value, (KT.int16, KT.int32, KT.float16, KT.bfloat16, KT.float32))
    src_dtype = infer_common_dtype(src0, src1)
    src0 = create_tile(src0, src_dtype, mask.shape, TensorLocation.UB)
    src1 = create_tile(src1, src_dtype, mask.shape, TensorLocation.UB)
    handle = global_builder.get_ir_builder().create_arith_SelectOp(mask.to_ir(), src0.to_ir(), src1.to_ir())
    return cast_loc(LocalTensor(handle))


@overload
def mask(*, count: RuntimeInt, other: Optional[RuntimeNumeric] = None) -> Generator[None, Any, None]:
    ...


@overload
def mask(*, bits: Iterable[RuntimeInt], other: Optional[RuntimeNumeric] = None) -> Generator[None, Any, None]:
    ...


@require_jit
@contextmanager
def mask(*, count: Optional[RuntimeInt] = None, bits: Optional[Iterable[RuntimeInt]] = None,
         other: Optional[RuntimeNumeric] = None) -> Generator[None, Any, None]:
    """
    [Experimental] A context manager for masked operations on tensors.

    Within the mask context, operations are applied only to the specified elements, with other elements optionally set
    to a different value.

    Two masking modes are supported:

    1. **Count-based masking**: Apply operations to the first ``count`` elements along the innermost dimension.
       Elements beyond ``count`` are set to ``other`` (default 0).

    2. **Bit-based masking**: Apply operations to elements where the bit index (computed from position) falls within the
       range specified by ``bits``. ``bits`` must contain exactly two integers defining the range.

    Args:
        count: The number of elements to apply operations to (from the start). Mutually exclusive with ``bits``.
        bits: A tuple of two integers defining a bit-based range for masking. Mutually exclusive with ``count``.
        other: The value to use for elements outside the mask. Default is 0.

    Raises:
        TypeError: If ``other`` is not a scalar, ``count`` is not an integer, or ``bits`` does not contain integers
        RuntimeError: If ``bits`` does not contain exactly two integers
        ValueError: If neither or both of ``count`` and ``bits`` are provided

    Warning:
        This is an experimental API which is not guaranteed to work for every relevant vector operation.
        Its interface, availability, and functional coverage may change in the future.

    Note:
        Exactly one of ``count`` or ``bits`` must be provided.
        This context manager can only be applied to vector operations (e.g., add, exp), not to load/store operations.

    Examples:
        Apply addition only to first 8 elements, others set to 0: ::

            with asc2.mask(count=8, other=0):
                result = tensor_a + tensor_b

        Apply exp only to elements within bit range, others set to -1: ::

            with asc2.mask(bits=[0, 64], other=-1):
                result = asc2.exp(tensor)
    """
    builder = global_builder.get_ir_builder()
    if other is not None:
        check_type("other", other, RuntimeNumeric)
        other = _mat(other).to_ir()
    has_count = count is not None
    has_bits = bits is not None
    if has_count and has_bits:
        raise ValueError("Only one of 'count' or 'bits' can be provided, not both")
    if has_count:
        check_runtime_int("count", count)
        mask_op = builder.create_asctile_CountMaskOp(_mat(count, KT.int64).to_ir(), other)
    elif has_bits:
        bits = verify_runtime_ints(bits, "bits", 2)
        mask_op = builder.create_asctile_BitwiseMaskOp(
            _mat(bits[0], KT.int64).to_ir(),
            _mat(bits[1], KT.int64).to_ir(), other)
    else:
        raise ValueError("One of 'count', 'bits' must be provided")
    old_insertion_point = global_builder.get_ir_builder().save_insertion_point()
    mask_region = mask_op.get_region()
    new_block = builder.create_block(mask_region)
    builder.set_insertion_point_to_start(new_block)
    try:
        yield
    finally:
        builder.create_asctile_YieldOp([])
        builder.restore_insertion_point(old_insertion_point)
