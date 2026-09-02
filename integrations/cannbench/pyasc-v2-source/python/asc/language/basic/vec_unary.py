# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Callable, Dict, List, Tuple, overload

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.ir_value import RuntimeInt, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import DefaultValued, OverloadDispatcher, require_jit, global_builder
from ..core.types import UnaryRepeatParams
from .utils import set_unary_docstring


def op_impl(callee: str, dst: LocalTensor, src: LocalTensor, args: Tuple[Any], kwargs: Dict[str, Any],
            build_l0: Callable, build_l1: Callable, build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")
    dispatcher = OverloadDispatcher(callee)

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        build_l0(dst.to_ir(), src.to_ir(),
                 _mat(mask, KT.uint64).to_ir(),
                 _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(), mask,
                 _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register_auto
    def _(count: RuntimeInt):
        build_l2(dst.to_ir(), src.to_ir(), _mat(count, KT.int32).to_ir())

    dispatcher(*args, **kwargs)


@overload
def abs(dst: LocalTensor, src: LocalTensor, count: int) -> None:
    ...


@overload
def abs(dst: LocalTensor, src: LocalTensor, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams,
        is_set_mask: bool = True) -> None:
    ...


@overload
def abs(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams,
        is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_unary_docstring(cpp_name="Abs", append_text="按元素取绝对值。")
def abs(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("abs", dst, src, args, kwargs, builder.create_asc_AbsL0Op, builder.create_asc_AbsL1Op,
            builder.create_asc_AbsL2Op)


@overload
def exp(dst: LocalTensor, src: LocalTensor, count: int) -> None:
    ...


@overload
def exp(dst: LocalTensor, src: LocalTensor, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams,
        is_set_mask: bool = True) -> None:
    ...


@overload
def exp(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams,
        is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_unary_docstring(cpp_name="Exp", append_text="按元素取自然指数。")
def exp(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("exp", dst, src, args, kwargs, builder.create_asc_ExpL0Op, builder.create_asc_ExpL1Op,
            builder.create_asc_ExpL2Op)


@overload
def ln(dst: LocalTensor, src: LocalTensor, count: int) -> None:
    ...


@overload
def ln(dst: LocalTensor, src: LocalTensor, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams,
       is_set_mask: bool = True) -> None:
    ...


@overload
def ln(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams,
       is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_unary_docstring(cpp_name="Ln", append_text="按元素取自然对数。")
def ln(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("ln", dst, src, args, kwargs, builder.create_asc_LnL0Op, builder.create_asc_LnL1Op,
            builder.create_asc_LnL2Op)


@overload
def bitwise_not(dst: LocalTensor, src: LocalTensor, count: int) -> None:
    ...


@overload
def bitwise_not(dst: LocalTensor, src: LocalTensor, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams,
                is_set_mask: bool = True) -> None:
    ...


@overload
def bitwise_not(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_times: int,
                repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_unary_docstring(cpp_name="Not", append_text="按元素做按位取反。命名为 bitwise_not 避免与Python关键字重名。")
def bitwise_not(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("bitwise_not", dst, src, args, kwargs, builder.create_asc_NotL0Op, builder.create_asc_NotL1Op,
            builder.create_asc_NotL2Op)


@overload
def reciprocal(dst: LocalTensor, src: LocalTensor, count: int) -> None:
    ...


@overload
def reciprocal(dst: LocalTensor, src: LocalTensor, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams,
               is_set_mask: bool = True) -> None:
    ...


@overload
def reciprocal(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams,
               is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_unary_docstring(cpp_name="Reciprocal", append_text="按元素取倒数。")
def reciprocal(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("reciprocal", dst, src, args, kwargs, builder.create_asc_ReciprocalL0Op, builder.create_asc_ReciprocalL1Op,
            builder.create_asc_ReciprocalL2Op)


@overload
def relu(dst: LocalTensor, src: LocalTensor, count: int) -> None:
    ...


@overload
def relu(dst: LocalTensor, src: LocalTensor, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams,
         is_set_mask: bool = True) -> None:
    ...


@overload
def relu(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams,
         is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_unary_docstring(cpp_name="Relu", append_text="按元素做线性整流Relu。")
def relu(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("relu", dst, src, args, kwargs, builder.create_asc_ReluL0Op, builder.create_asc_ReluL1Op,
            builder.create_asc_ReluL2Op)


@overload
def rsqrt(dst: LocalTensor, src: LocalTensor, count: int) -> None:
    ...


@overload
def rsqrt(dst: LocalTensor, src: LocalTensor, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams,
          is_set_mask: bool = True) -> None:
    ...


@overload
def rsqrt(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams,
          is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_unary_docstring(cpp_name="Rsqrt", append_text="按元素进行开方后取倒数的计算。")
def rsqrt(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("rsqrt", dst, src, args, kwargs, builder.create_asc_RsqrtL0Op, builder.create_asc_RsqrtL1Op,
            builder.create_asc_RsqrtL2Op)


@overload
def sqrt(dst: LocalTensor, src: LocalTensor, count: int) -> None:
    ...


@overload
def sqrt(dst: LocalTensor, src: LocalTensor, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams,
         is_set_mask: bool = True) -> None:
    ...


@overload
def sqrt(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams,
         is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_unary_docstring(cpp_name="Sqrt", append_text="按元素做开方。")
def sqrt(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("sqrt", dst, src, args, kwargs, builder.create_asc_SqrtL0Op, builder.create_asc_SqrtL1Op,
            builder.create_asc_SqrtL2Op)
