# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List, overload

from ..._C import ir
from ..core.dtype import KnownTypes, KnownTypes as KT
from ..core.enums import RoundMode
from ..core.ir_value import RuntimeBool, RuntimeInt, RuntimeFloat, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import require_jit, global_builder, DefaultValued, OverloadDispatcher
from ..core.types import BinaryRepeatParams, UnaryRepeatParams, VdeqInfo
from .utils import op_impl, set_binary_docstring, set_common_docstring
from .vec_unary import op_impl as unary_op_impl


@overload
def add_relu_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int) -> None:
    ...


@overload
def add_relu_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
                  repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def add_relu_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
                  repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="AddReluCast", append_text="按元素求和，结果和0对比取较大值，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。")
def add_relu_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("add_relu_cast", dst, src0, src1, args, kwargs, builder.create_asc_AddReluCastL0Op,
            builder.create_asc_AddReluCastL1Op, builder.create_asc_AddReluCastL2Op)


@overload
def cast(dst: LocalTensor, src: LocalTensor, round_mode: RoundMode, count: int) -> None:
    ...


@overload
def cast(dst: LocalTensor, src: LocalTensor, round_mode: RoundMode, mask: int, repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def cast(dst: LocalTensor, src: LocalTensor, round_mode: RoundMode, mask: List[int], repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_common_docstring(api_name="cast")
def cast(dst: LocalTensor, src: LocalTensor, round_mode: RoundMode, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    dispatcher = OverloadDispatcher("cast")

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        builder.create_asc_CastL0Op(dst.to_ir(), src.to_ir(), ir.RoundMode.symbolize(round_mode),
                                    _mat(mask, KT.uint64).to_ir(),
                                    _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        builder.create_asc_CastL1Op(dst.to_ir(), src.to_ir(), ir.RoundMode.symbolize(round_mode), mask,
                                    _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register_auto
    def _(count: RuntimeInt):
        builder.create_asc_CastL2Op(dst.to_ir(), src.to_ir(), ir.RoundMode.symbolize(round_mode),
                                    _mat(count, KT.int32).to_ir())

    dispatcher(*args, **kwargs)


@overload
def cast_deq(dst: LocalTensor, src: LocalTensor, count: int, is_vec_deq: bool = True, half_block: bool = True) -> None:
    ...


@overload
def cast_deq(dst: LocalTensor, src: LocalTensor, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams,
             is_set_mask: bool = True, is_vec_deq: bool = True, half_block: bool = True) -> None:
    ...


@overload
def cast_deq(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams,
             is_set_mask: bool = True, is_vec_deq: bool = True, half_block: bool = True) -> None:
    ...


@require_jit
@set_common_docstring(api_name="cast_deq")
def cast_deq(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    unary_op_impl("cast_deq", dst, src, args, kwargs, builder.create_asc_CastDeqL0Op, builder.create_asc_CastDeqL1Op,
                  builder.create_asc_CastDeqL2Op)


@overload
def set_deq_scale(scale: float) -> None:
    ...


@overload
def set_deq_scale(scale: float, offset: int, sign_mode: bool) -> None:
    ...


@overload
def set_deq_scale(vdeq: LocalTensor, vdeq_info: VdeqInfo) -> None:
    ...


@require_jit
@set_common_docstring(api_name="set_deq_scale")
def set_deq_scale(*args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    dispatcher = OverloadDispatcher("set_deq_scale")

    @dispatcher.register(vdeq=LocalTensor, vdeq_info=VdeqInfo)
    def _(vdeq: LocalTensor, vdeq_info: VdeqInfo):
        builder.create_asc_SetDeqScaleL4Op(vdeq.to_ir(), vdeq_info.to_ir())

    @dispatcher.register(scale=RuntimeFloat)
    def _(scale: RuntimeFloat):
        builder.create_asc_SetDeqScaleOp(_mat(scale, KnownTypes.half).to_ir())

    @dispatcher.register(scale=RuntimeFloat, offset=RuntimeInt, sign_mode=RuntimeBool)
    def _(scale: RuntimeFloat, offset: RuntimeInt, sign_mode: RuntimeBool):
        builder.create_asc_SetDeqScaleOp(
            _mat(scale, KnownTypes.float32).to_ir(),
            _mat(offset, KnownTypes.int16).to_ir(),
            _mat(sign_mode, KnownTypes.bit).to_ir())

    dispatcher(*args, **kwargs)


@overload
def sub_relu_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int) -> None:
    ...


@overload
def sub_relu_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
                  repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def sub_relu_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
                  repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="SubReluCast", append_text="按元素求差，结果和0对比取较大值，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。")
def sub_relu_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("sub_relu_cast", dst, src0, src1, args, kwargs, builder.create_asc_SubReluCastL0Op,
            builder.create_asc_SubReluCastL1Op, builder.create_asc_SubReluCastL2Op)
