# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from typing import overload, TypeVar, Type

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.enums import RoundMode
from ..core.ir_value import PlainValue, RuntimeInt, RuntimeFloat, materialize_ir_value as _mat
from ..core.utils import require_jit, global_builder
from .utils import set_common_docstring

T = TypeVar("T", int, float)


@overload
def scalar_cast(value_in: float, dtype: Type[T], round_mode: RoundMode) -> T:
    ...


@require_jit
@set_common_docstring(api_name="scalar_cast")
def scalar_cast(value_in: RuntimeFloat, dtype: Type[T], round_mode: RoundMode) -> T:
    builder = global_builder.get_ir_builder()

    value_out = builder.create_asc_ScalarCastOp(dtype.to_ir(),
                                                _mat(value_in, KT.float_).to_ir(), dtype.to_ir(),
                                                ir.RoundMode.symbolize(round_mode))
    if dtype in (KT.int32, KT.float16, KT.half):
        return PlainValue(value_out)
    else:
        raise TypeError(f"Unsupported target dtype: {dtype}")


@overload
def scalar_get_sff_value(value_in: int, count_value: int) -> int:
    ...


@require_jit
@set_common_docstring(api_name="scalar_get_sff_value")
def scalar_get_sff_value(value_in: RuntimeInt, count_value: RuntimeInt) -> RuntimeInt:
    builder = global_builder.get_ir_builder()
    if not isinstance(count_value, int):
        raise TypeError("count_value must be a Python int (compile-time constant).")
    if count_value not in (0, 1):
        raise ValueError("count_value must be 0 or 1.")
    handle = builder.create_asc_ScalarGetSFFValueOp(KT.int64.to_ir(),
                                                    _mat(value_in, KT.uint64).to_ir(),
                                                    _mat(count_value, KT.int32).to_ir())
    return PlainValue(handle)


@overload
def scalar_get_count_of_value(value_in: int, count_value: int) -> int:
    ...


@require_jit
@set_common_docstring(api_name="scalar_get_count_of_value")
def scalar_get_count_of_value(value_in: RuntimeInt, count_value: RuntimeInt) -> RuntimeInt:
    builder = global_builder.get_ir_builder()
    handle = builder.create_asc_ScalarGetCountOfValueOp(KT.int64.to_ir(),
                                                        _mat(value_in, KT.uint64).to_ir(),
                                                        _mat(count_value, KT.int32).to_ir())
    return PlainValue(handle)


@overload
def scalar_count_leading_zero(value_in: int) -> int:
    ...


@require_jit
@set_common_docstring(api_name="scalar_count_leading_zero")
def scalar_count_leading_zero(value_in: RuntimeInt) -> RuntimeInt:
    builder = global_builder.get_ir_builder()
    handle = builder.create_asc_ScalarCountLeadingZeroOp(KT.int64.to_ir(), _mat(value_in, KT.uint64).to_ir())
    return PlainValue(handle)


@overload
def count_bits_cnt_same_as_sign_bit(value_in: int) -> int:
    ...


@require_jit
@set_common_docstring(api_name="count_bits_cnt_same_as_sign_bit")
def count_bits_cnt_same_as_sign_bit(value_in: RuntimeInt) -> RuntimeInt:
    builder = global_builder.get_ir_builder()
    handle = builder.create_asc_CountBitsCntSameAsSignBitOp(KT.int64.to_ir(), _mat(value_in, KT.int64).to_ir())
    return PlainValue(handle)
