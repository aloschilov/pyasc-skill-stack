# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable, List, TypeVar, Union, overload

from ..core.dtype import KnownTypes as KT
from ..core.ir_value import RuntimeBool, RuntimeInt, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import OverloadDispatcher, require_jit, global_builder
from ..core.types import BinaryRepeatParams
from .utils import check_type, op_impl, set_binary_docstring

T = TypeVar("T", bound=Callable)


@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="Add", append_text="按元素求和。")
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("add", dst, src0, src1, args, kwargs, builder.create_asc_AddL0Op, builder.create_asc_AddL1Op,
            builder.create_asc_AddL2Op)


@overload
def add_deq_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def add_deq_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
                 repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def add_deq_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
                 repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="AddDeqRelu", append_text="依次计算按元素求和、结果进行deq量化后再进行relu计算（结果和0对比取较大值）。")
def add_deq_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("add_deq_relu", dst, src0, src1, args, kwargs, builder.create_asc_AddDeqReluL0Op,
            builder.create_asc_AddDeqReluL1Op, builder.create_asc_AddDeqReluL2Op)


@overload
def add_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def add_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
             repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def add_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
             repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="AddRelu", append_text="按元素求和，再进行Relu计算（结果和0对比取较大值）。")
def add_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("add_relu", dst, src0, src1, args, kwargs, builder.create_asc_AddReluL0Op, builder.create_asc_AddReluL1Op,
            builder.create_asc_AddReluL2Op)


@overload
def bitwise_and(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def bitwise_and(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
                repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def bitwise_and(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
                repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="And", append_text="每对elements按位与运算。命名为 bitwise_and 避免与Python关键字重名。")
def bitwise_and(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    """
    Computes the element-wise and, corresponding to AscendC::And.
    Use bitwise_and to avoid conflict with python keywords.
    """
    builder = global_builder.get_ir_builder()
    op_impl("bitwise_and", dst, src0, src1, args, kwargs, builder.create_asc_AndL0Op, builder.create_asc_AndL1Op,
            builder.create_asc_AndL2Op)


@overload
def bitwise_or(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def bitwise_or(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
               repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def bitwise_or(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
               repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="Or", append_text="每对elements按位或运算。命名为 bitwise_or 避免与Python关键字重名。")
def bitwise_or(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    """
    Computes the element-wise or, corresponding to AscendC::Or. Use bitwise_or to avoid conflict with python keywords.
    """
    builder = global_builder.get_ir_builder()
    op_impl("bitwise_or", dst, src0, src1, args, kwargs, builder.create_asc_OrL0Op, builder.create_asc_OrL1Op,
            builder.create_asc_OrL2Op)


@overload
def div(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def div(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def div(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="Div", append_text="按元素求商。")
def div(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("div", dst, src0, src1, args, kwargs, builder.create_asc_DivL0Op, builder.create_asc_DivL1Op,
            builder.create_asc_DivL2Op)


@overload
def fused_mul_add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def fused_mul_add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
                  repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def fused_mul_add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
                  repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="FusedMulAdd", append_text="按元素将src0和dst相乘并加上src1，最终结果存放入dst。")
def fused_mul_add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("fused_mul_add", dst, src0, src1, args, kwargs, builder.create_asc_FusedMulAddL0Op,
            builder.create_asc_FusedMulAddL1Op, builder.create_asc_FusedMulAddL2Op)


@overload
def fused_mul_add_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int,
                       is_set_mask: bool = True) -> None:
    ...


@overload
def fused_mul_add_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
                       repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def fused_mul_add_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
                       repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="FusedMulAddRelu",
                      append_text="按元素将src0和dst相乘并加上src1，再进行Relu计算（结果和0对比取较大值），最终结果存放进dst中。")
def fused_mul_add_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("fused_mul_add_relu", dst, src0, src1, args, kwargs, builder.create_asc_FusedMulAddReluL0Op,
            builder.create_asc_FusedMulAddReluL1Op, builder.create_asc_FusedMulAddReluL2Op)


@overload
def max(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def max(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def max(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="Max", append_text="按元素求最大值。")
def max(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("max", dst, src0, src1, args, kwargs, builder.create_asc_MaxL0Op, builder.create_asc_MaxL1Op,
            builder.create_asc_MaxL2Op)


@overload
def min(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def min(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def min(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="Min", append_text="按元素求最小值。")
def min(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("min", dst, src0, src1, args, kwargs, builder.create_asc_MinL0Op, builder.create_asc_MinL1Op,
            builder.create_asc_MinL2Op)


@overload
def mul(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def mul(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def mul(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="Mul", append_text="按元素求积。")
def mul(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("mul", dst, src0, src1, args, kwargs, builder.create_asc_MulL0Op, builder.create_asc_MulL1Op,
            builder.create_asc_MulL2Op)


@overload
def mul_add_dst(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def mul_add_dst(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
                repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def mul_add_dst(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
                repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="MulAddDst", append_text="按元素将src0和src1相乘并和dst相加，将最终结果存放进dst中。")
def mul_add_dst(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("mul_add_dst", dst, src0, src1, args, kwargs, builder.create_asc_MulAddDstL0Op,
            builder.create_asc_MulAddDstL1Op, builder.create_asc_MulAddDstL2Op)


@overload
def mul_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int) -> None:
    ...


@overload
def mul_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
             repeat_params: BinaryRepeatParams) -> None:
    ...


@overload
def mul_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
             repeat_params: BinaryRepeatParams) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="MulCast", append_text="按元素求积，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。")
def mul_cast(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    dispatcher = OverloadDispatcher(__name__)
    builder = global_builder.get_ir_builder()

    check_type("mul_cast", dst, src0, src1)

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams)
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams):
        builder.create_asc_MulCastL0Op(dst.to_ir(), src0.to_ir(), src1.to_ir(),
                                       _mat(mask, KT.uint64).to_ir(),
                                       _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir())

    @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams)
    def _(mask: list, repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        builder.create_asc_MulCastL1Op(dst.to_ir(), src0.to_ir(), src1.to_ir(), mask,
                                       _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir())

    @dispatcher.register_auto
    def _(count: RuntimeInt):
        builder.create_asc_MulCastL2Op(dst.to_ir(), src0.to_ir(), src1.to_ir(), _mat(count, KT.int32).to_ir())

    dispatcher(*args, **kwargs)


@overload
def sub(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def sub(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def sub(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
        repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="Sub", append_text="按元素求差。")
def sub(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("sub", dst, src0, src1, args, kwargs, builder.create_asc_SubL0Op, builder.create_asc_SubL1Op,
            builder.create_asc_SubL2Op)


@overload
def sub_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
    ...


@overload
def sub_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
             repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def sub_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
             repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_binary_docstring(cpp_name="SubRelu", append_text="按元素求差，再进行Relu计算（结果和0对比取较大值）。")
def sub_relu(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("sub_relu", dst, src0, src1, args, kwargs, builder.create_asc_SubReluL0Op, builder.create_asc_SubReluL1Op,
            builder.create_asc_SubReluL2Op)


@overload
def bilinear_interpolation(dst: LocalTensor, src0: LocalTensor, src0_offset: LocalTensor, src1: LocalTensor, mask: int,
                           h_repeat: int, repeat_mode: bool, dst_blk_stride: int, v_r_offset: int, v_repeat: int,
                           shared_tmp_buffer: LocalTensor) -> None:
    ...


@overload
def bilinear_interpolation(dst: LocalTensor, src0: LocalTensor, src0_offset: LocalTensor, src1: LocalTensor,
                           mask: List[int], h_repeat: int, repeat_mode: bool, dst_blk_stride: int, v_r_offset: int,
                           v_repeat: int, shared_tmp_buffer: LocalTensor) -> None:
    ...


@require_jit
def bilinear_interpolation(dst: LocalTensor, src0: LocalTensor, src0_offset: LocalTensor, src1: LocalTensor,
                           mask: Union[list, RuntimeInt], h_repeat: RuntimeInt, repeat_mode: RuntimeBool,
                           dst_blk_stride: RuntimeInt, v_r_offset: RuntimeInt, v_repeat: RuntimeInt,
                           shared_tmp_buffer: LocalTensor) -> None:
    """
    分为水平迭代和垂直迭代。
    每个水平迭代顺序地从src0_offset读取8个偏移值，表示src0的偏移，每个偏移值指向src0的一个data_block的起始地址，如果repeat_mode=false，从src1中取一个值，
    与src0中8个data_block中每个值进行乘操作；如果repeat_mode=true，从src1中取8个值，按顺序与src0中8个data_block中的值进行乘操作，
    最后当前迭代的dst结果与前一个dst结果按data_block进行累加，存入目的地址，在同一个水平迭代内dst地址不变。
    然后进行垂直迭代，垂直迭代的dst起始地址为上一轮垂直迭代的dst起始地址加上v_r_offset，本轮垂直迭代占用dst空间为dst起始地址之后的8个data_block，每轮垂直迭代进行h_repeat次水平迭代。

    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void BilinearInterpolation(const LocalTensor<T> &dst, const LocalTensor<T> &src0, 
                            const LocalTensor<uint32_t> &src0Offset, const LocalTensor<T> &src1, uint64_t mask[], 
                            uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, 
                            uint8_t vRepeat, const LocalTensor<uint8_t> &sharedTmpBuffer)
                                
    .. code-block:: c++

        template <typename T>
        __aicore__ inline void BilinearInterpolation(const LocalTensor<T> &dst, const LocalTensor<T> &src0, 
                            const LocalTensor<uint32_t> &src0Offset, const LocalTensor<T> &src1, uint64_t mask,
                            uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, 
                            uint8_t vRepeat, const LocalTensor<uint8_t> &sharedTmpBuffer)


    **参数说明**
    
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0, src1：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - count：参与计算的元素个数。
    - mask：用于控制每次迭代内参与计算的元素。
    - repeat_times：重复迭代次数。
    - params：控制操作数地址步长的参数。

    **调用示例**

    - 接口样例-mask连续模式

      .. code-block:: python

          mask = 128;         # mask连续模式
          h_repeat = 2;        # 水平迭代2次
          repeat_mode = false; # 迭代模式
          dst_blk_stride = 1;   # 单次迭代内数据连续写入
          v_r_offset = 128;     # 相邻迭代间数据连续写入
          v_repeat = 2;        # 垂直迭代2次
          asc.bilinear_interpolation(dst_local, src0_local, src0_offset_local, src1_local, mask, h_repeat, repeat_mode,
          dst_blk_stride, v_r_offset, v_repeat, tmp_local)
            
    - 接口样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max];         # mask逐bit模式
          h_repeat = 2;        # 水平迭代2次
          repeat_mode = false; # 迭代模式
          dst_blk_stride = 1;   # 单次迭代内数据连续写入
          v_r_offset = 128;     # 相邻迭代间数据连续写入
          v_repeat = 2;        # 垂直迭代2次
          asc.bilinear_interpolation(dst_local, src0_local, src0_offset_local, src1_local, mask, h_repeat, repeat_mode,
          dst_blk_stride, v_r_offset, v_repeat, tmp_local)
            
    """

    builder = global_builder.get_ir_builder()

    check_type("bilinear_interpolation", dst, src0, src1)

    if isinstance(mask, list):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        builder.create_asc_BilinearInterpolationL1Op(dst.to_ir(), src0.to_ir(), src0_offset.to_ir(), src1.to_ir(), mask,
                                                     _mat(h_repeat, KT.uint8).to_ir(),
                                                     _mat(repeat_mode, KT.bool_).to_ir(),
                                                     _mat(dst_blk_stride, KT.uint16).to_ir(),
                                                     _mat(v_r_offset, KT.uint16).to_ir(),
                                                     _mat(v_repeat, KT.uint8).to_ir(), shared_tmp_buffer.to_ir())
    else:
        builder.create_asc_BilinearInterpolationL0Op(dst.to_ir(), src0.to_ir(), src0_offset.to_ir(), src1.to_ir(),
                                                     _mat(mask, KT.uint64).to_ir(),
                                                     _mat(h_repeat, KT.uint8).to_ir(),
                                                     _mat(repeat_mode, KT.bool_).to_ir(),
                                                     _mat(dst_blk_stride, KT.uint16).to_ir(),
                                                     _mat(v_r_offset, KT.uint16).to_ir(),
                                                     _mat(v_repeat, KT.uint8).to_ir(), shared_tmp_buffer.to_ir())
