# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List, Union, overload

from ..core.ir_value import RuntimeNumeric
from ..core.tensor import LocalTensor
from ..core.utils import require_jit, global_builder
from ..core.types import UnaryRepeatParams
from .utils import set_binary_scalar_docstring, vec_binary_scalar_op_impl as op_impl


@overload
def adds(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def adds(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: int, repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def adds(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: List[int], repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_scalar_docstring(cpp_name="Adds", append_text="矢量内每个元素与标量求和。")
def adds(dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("adds", dst, src, scalar, args, kwargs, builder.create_asc_AddsL0Op, builder.create_asc_AddsL1Op,
            builder.create_asc_AddsL2Op)


@overload
def leaky_relu(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], count: int,
               is_set_mask: bool = True) -> None:
    ...


@overload
def leaky_relu(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: int, repeat_times: int,
               repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def leaky_relu(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: List[int], repeat_times: int,
               repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_scalar_docstring(cpp_name="LeakyRelu", append_text="按元素执行Leaky ReLU（Leaky Rectified Linear Unit）操作。")
def leaky_relu(dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("leaky_relu", dst, src, scalar, args, kwargs, builder.create_asc_LeakyReluL0Op,
            builder.create_asc_LeakyReluL1Op, builder.create_asc_LeakyReluL2Op)


@overload
def maxs(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def maxs(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: int, repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def maxs(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: List[int], repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_scalar_docstring(cpp_name="Maxs", append_text="源操作数矢量内每个元素与标量相比，如果比标量大，则取源操作数值，比标量的值小，则取标量值。")
def maxs(dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("maxs", dst, src, scalar, args, kwargs, builder.create_asc_MaxsL0Op, builder.create_asc_MaxsL1Op,
            builder.create_asc_MaxsL2Op)


@overload
def mins(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def mins(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: int, repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def mins(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: List[int], repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_scalar_docstring(cpp_name="Mins", append_text="源操作数矢量内每个元素与标量相比，如果比标量大，则取标量值，比标量的值小，则取源操作数。")
def mins(dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("mins", dst, src, scalar, args, kwargs, builder.create_asc_MinsL0Op, builder.create_asc_MinsL1Op,
            builder.create_asc_MinsL2Op)


@overload
def muls(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def muls(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: int, repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def muls(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: List[int], repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_scalar_docstring(cpp_name="Muls", append_text="矢量内每个元素与标量求积。")
def muls(dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("muls", dst, src, scalar, args, kwargs, builder.create_asc_MulsL0Op, builder.create_asc_MulsL1Op,
            builder.create_asc_MulsL2Op)


@overload
def shift_left(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], count: int,
               is_set_mask: bool = True) -> None:
    ...


@overload
def shift_left(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: int, repeat_times: int,
               repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def shift_left(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: List[int], repeat_times: int,
               repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_scalar_docstring(cpp_name="ShiftLeft", append_text="对源操作数中的每个元素进行左移操作，左移的位数由输入参数scalar决定。")
def shift_left(dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("shift_left", dst, src, scalar, args, kwargs, builder.create_asc_ShiftLeftL0Op,
            builder.create_asc_ShiftLeftL1Op, builder.create_asc_ShiftLeftL2Op)


@overload
def shift_right(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], count: int,
                is_set_mask: bool = True) -> None:
    ...


@overload
def shift_right(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: int, repeat_times: int,
                repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def shift_right(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: List[int], repeat_times: int,
                repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_scalar_docstring(cpp_name="ShiftRight", append_text="对源操作数中的每个元素进行右移操作，右移的位数由输入参数scalar决定。")
def shift_right(dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("shift_right", dst, src, scalar, args, kwargs, builder.create_asc_ShiftRightL0Op,
            builder.create_asc_ShiftRightL1Op, builder.create_asc_ShiftRightL2Op)
