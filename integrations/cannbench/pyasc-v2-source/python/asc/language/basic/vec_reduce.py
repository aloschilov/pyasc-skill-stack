# Copyright (c) 2025 AISS Group, ISE Group, AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Callable, Dict, List, Optional, Tuple, overload

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.enums import ReduceOrder
from ..core.ir_value import RuntimeInt, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import DefaultValued, OverloadDispatcher, require_jit, global_builder
from .utils import set_common_docstring


def reduce_op_impl(callee: str, dst: LocalTensor, src: LocalTensor, args: Tuple[Any], kwargs: Dict[str, Any],
                   build_l0: Callable, build_l1: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")

    dispatcher = OverloadDispatcher(callee)

    @dispatcher.register_auto
    def _(repeat: RuntimeInt, mask: RuntimeInt, dst_rep_stride: RuntimeInt, src_blk_stride: RuntimeInt,
          src_rep_stride: RuntimeInt):
        build_l0(dst.to_ir(), src.to_ir(),
                 _mat(repeat, KT.int32).to_ir(),
                 _mat(mask, KT.int32).to_ir(),
                 _mat(dst_rep_stride, KT.int32).to_ir(),
                 _mat(src_blk_stride, KT.int32).to_ir(),
                 _mat(src_rep_stride, KT.int32).to_ir())

    @dispatcher.register_auto
    def _(repeat: RuntimeInt, mask: list, dst_rep_stride: RuntimeInt, src_blk_stride: RuntimeInt,
          src_rep_stride: RuntimeInt):
        mask_vals = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(),
                 _mat(repeat, KT.int32).to_ir(), mask_vals,
                 _mat(dst_rep_stride, KT.int32).to_ir(),
                 _mat(src_blk_stride, KT.int32).to_ir(),
                 _mat(src_rep_stride, KT.int32).to_ir())

    dispatcher(*args, **kwargs)


# BlockReduceSum
@overload
def block_reduce_sum(dst: LocalTensor, src: LocalTensor, repeat: int, mask: int, dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int) -> None:
    ...


@overload
def block_reduce_sum(dst: LocalTensor, src: LocalTensor, repeat: int, mask: List[int], dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int) -> None:
    ...


@require_jit
@set_common_docstring("block_reduce_sum")
def block_reduce_sum(dst: LocalTensor, src: LocalTensor, repeat: int, mask, dst_rep_stride: int, src_blk_stride: int,
                     src_rep_stride: int) -> None:
    builder = global_builder.get_ir_builder()
    reduce_op_impl("block_reduce_sum", dst, src, (repeat, mask, dst_rep_stride, src_blk_stride, src_rep_stride), {},
                   builder.create_asc_BlockReduceSumL0Op, builder.create_asc_BlockReduceSumL1Op)


@overload
def block_reduce_max(dst: LocalTensor, src: LocalTensor, repeat: int, mask: int, dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int) -> None:
    ...


@overload
def block_reduce_max(dst: LocalTensor, src: LocalTensor, repeat: int, mask: List[int], dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int) -> None:
    ...


@require_jit
@set_common_docstring("block_reduce_max")
def block_reduce_max(dst: LocalTensor, src: LocalTensor, repeat: int, mask, dst_rep_stride: int, src_blk_stride: int,
                     src_rep_stride: int) -> None:
    builder = global_builder.get_ir_builder()
    reduce_op_impl("block_reduce_max", dst, src, (repeat, mask, dst_rep_stride, src_blk_stride, src_rep_stride), {},
                   builder.create_asc_BlockReduceMaxL0Op, builder.create_asc_BlockReduceMaxL1Op)


@overload
def block_reduce_min(dst: LocalTensor, src: LocalTensor, repeat: int, mask: int, dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int) -> None:
    ...


@overload
def block_reduce_min(dst: LocalTensor, src: LocalTensor, repeat: int, mask: List[int], dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int) -> None:
    ...


@require_jit
@set_common_docstring("block_reduce_min")
def block_reduce_min(dst: LocalTensor, src: LocalTensor, repeat: int, mask, dst_rep_stride: int, src_blk_stride: int,
                     src_rep_stride: int) -> None:
    builder = global_builder.get_ir_builder()
    reduce_op_impl("block_reduce_min", dst, src, (repeat, mask, dst_rep_stride, src_blk_stride, src_rep_stride), {},
                   builder.create_asc_BlockReduceMinL0Op, builder.create_asc_BlockReduceMinL1Op)


@overload
def pair_reduce_sum(dst: LocalTensor, src: LocalTensor, repeat_time: int, mask: int, dst_rep_stride: int,
                    src_blk_stride: int, src_rep_stride: int, is_set_mask: bool = True) -> None:
    ...


@overload
def pair_reduce_sum(dst: LocalTensor, src: LocalTensor, repeat_time: int, mask: List[int], dst_rep_stride: int,
                    src_blk_stride: int, src_rep_stride: int, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_common_docstring("pair_reduce_sum")
def pair_reduce_sum(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    dispatcher = OverloadDispatcher("pair_reduce_sum")

    @dispatcher.register(repeat_time=RuntimeInt, mask=RuntimeInt, dst_rep_stride=RuntimeInt, src_blk_stride=RuntimeInt,
                         src_rep_stride=RuntimeInt, is_set_mask=DefaultValued(bool, True))
    def _(repeat_time: RuntimeInt, mask: RuntimeInt, dst_rep_stride: RuntimeInt, src_blk_stride: RuntimeInt,
          src_rep_stride: RuntimeInt, is_set_mask: bool = True):
        builder.create_asc_PairReduceSumL0Op(dst.to_ir(), src.to_ir(),
                                             _mat(repeat_time, KT.int32).to_ir(),
                                             _mat(mask, KT.int32).to_ir(),
                                             _mat(dst_rep_stride, KT.int32).to_ir(),
                                             _mat(src_blk_stride, KT.int32).to_ir(),
                                             _mat(src_rep_stride, KT.int32).to_ir(), is_set_mask)

    @dispatcher.register(repeat_time=RuntimeInt, mask=list, dst_rep_stride=RuntimeInt, src_blk_stride=RuntimeInt,
                         src_rep_stride=RuntimeInt, is_set_mask=DefaultValued(bool, True))
    def _(repeat_time: RuntimeInt, mask: list, dst_rep_stride: RuntimeInt, src_blk_stride: RuntimeInt,
          src_rep_stride: RuntimeInt, is_set_mask: bool = True):
        mask_ir = [_mat(v, KT.uint64).to_ir() for v in mask]
        builder.create_asc_PairReduceSumL1Op(dst.to_ir(), src.to_ir(),
                                             _mat(repeat_time, KT.int32).to_ir(), mask_ir,
                                             _mat(dst_rep_stride, KT.int32).to_ir(),
                                             _mat(src_blk_stride, KT.int32).to_ir(),
                                             _mat(src_rep_stride, KT.int32).to_ir(), is_set_mask)

    dispatcher(*args, **kwargs)


@overload
def repeat_reduce_sum(dst: LocalTensor, src: LocalTensor, repeat_time: int, mask: int, dst_blk_stride: int,
                      src_blk_stride: int, dst_rep_stride: int, src_rep_stride: int, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_common_docstring("repeat_reduce_sum")
def repeat_reduce_sum(dst: LocalTensor, src: LocalTensor, repeat_time: RuntimeInt, mask: RuntimeInt,
                      dst_blk_stride: RuntimeInt, src_blk_stride: RuntimeInt, dst_rep_stride: RuntimeInt,
                      src_rep_stride: RuntimeInt, is_set_mask: bool = True) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_RepeatReduceSumL0Op(dst.to_ir(), src.to_ir(),
                                           _mat(repeat_time, KT.int32).to_ir(),
                                           _mat(mask, KT.int32).to_ir(),
                                           _mat(dst_blk_stride, KT.int32).to_ir(),
                                           _mat(src_blk_stride, KT.int32).to_ir(),
                                           _mat(dst_rep_stride, KT.int32).to_ir(),
                                           _mat(src_rep_stride, KT.int32).to_ir(), is_set_mask)


def whole_reduce_op_impl(callee: str, dst: LocalTensor, src: LocalTensor, args: Tuple[Any], kwargs: Dict[str, Any],
                         build_l0: Callable, build_l1: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")

    dispatcher = OverloadDispatcher(callee)

    @dispatcher.register(mask=RuntimeInt, repeat_time=RuntimeInt, dst_rep_stride=RuntimeInt, src_blk_stride=RuntimeInt,
                         src_rep_stride=RuntimeInt, order=DefaultValued(ReduceOrder, ReduceOrder.ORDER_VALUE_INDEX),
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_time: RuntimeInt, dst_rep_stride: RuntimeInt, src_blk_stride: RuntimeInt,
          src_rep_stride: RuntimeInt, order: ReduceOrder = ReduceOrder.ORDER_VALUE_INDEX, is_set_mask: bool = True):
        build_l0(dst.to_ir(), src.to_ir(),
                 _mat(mask, KT.int32).to_ir(),
                 _mat(repeat_time, KT.int32).to_ir(),
                 _mat(dst_rep_stride, KT.int32).to_ir(),
                 _mat(src_blk_stride, KT.int32).to_ir(),
                 _mat(src_rep_stride, KT.int32).to_ir(), ir.ReduceOrder.symbolize(order), is_set_mask)

    @dispatcher.register(mask=list, repeat_time=RuntimeInt, dst_rep_stride=RuntimeInt, src_blk_stride=RuntimeInt,
                         src_rep_stride=RuntimeInt, order=DefaultValued(ReduceOrder, ReduceOrder.ORDER_VALUE_INDEX),
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_time: RuntimeInt, dst_rep_stride: RuntimeInt, src_blk_stride: RuntimeInt,
          src_rep_stride: RuntimeInt, order: ReduceOrder = ReduceOrder.ORDER_VALUE_INDEX, is_set_mask: bool = True):
        mask_vals = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(), mask_vals,
                 _mat(repeat_time, KT.int32).to_ir(),
                 _mat(dst_rep_stride, KT.int32).to_ir(),
                 _mat(src_blk_stride, KT.int32).to_ir(),
                 _mat(src_rep_stride, KT.int32).to_ir(), ir.ReduceOrder.symbolize(order), is_set_mask)

    dispatcher(*args, **kwargs)


@overload
def whole_reduce_max(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_time: int, dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int,
                     order: Optional[ReduceOrder] = ReduceOrder.ORDER_VALUE_INDEX, is_set_mask: bool = True) -> None:
    ...


@overload
def whole_reduce_max(dst: LocalTensor, src: LocalTensor, mask: int, repeat_time: int, dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int,
                     order: Optional[ReduceOrder] = ReduceOrder.ORDER_VALUE_INDEX, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_common_docstring("whole_reduce_max")
def whole_reduce_max(dst: LocalTensor, src: LocalTensor, mask, repeat_time: RuntimeInt, dst_rep_stride: RuntimeInt,
                     src_blk_stride: RuntimeInt, src_rep_stride: RuntimeInt,
                     order: Optional[ReduceOrder] = ReduceOrder.ORDER_VALUE_INDEX) -> None:
    builder = global_builder.get_ir_builder()
    whole_reduce_op_impl("whole_reduce_max", dst, src,
                         (mask, repeat_time, dst_rep_stride, src_blk_stride, src_rep_stride, order), {},
                         builder.create_asc_WholeReduceMaxL0Op, builder.create_asc_WholeReduceMaxL1Op)


@overload
def whole_reduce_min(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_time: int, dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int,
                     order: Optional[ReduceOrder] = ReduceOrder.ORDER_VALUE_INDEX, is_set_mask: bool = True) -> None:
    ...


@overload
def whole_reduce_min(dst: LocalTensor, src: LocalTensor, mask: int, repeat_time: int, dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int,
                     order: Optional[ReduceOrder] = ReduceOrder.ORDER_VALUE_INDEX, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_common_docstring("whole_reduce_min")
def whole_reduce_min(dst: LocalTensor, src: LocalTensor, mask, repeat_time: RuntimeInt, dst_rep_stride: RuntimeInt,
                     src_blk_stride: RuntimeInt, src_rep_stride: RuntimeInt,
                     order: Optional[ReduceOrder] = ReduceOrder.ORDER_VALUE_INDEX) -> None:
    builder = global_builder.get_ir_builder()
    whole_reduce_op_impl("whole_reduce_min", dst, src,
                         (mask, repeat_time, dst_rep_stride, src_blk_stride, src_rep_stride, order), {},
                         builder.create_asc_WholeReduceMinL0Op, builder.create_asc_WholeReduceMinL1Op)


@overload
def whole_reduce_sum(dst: LocalTensor, src: LocalTensor, mask: int, repeat_time: int, dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int, is_set_mask: bool = True) -> None:
    ...


@overload
def whole_reduce_sum(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_time: int, dst_rep_stride: int,
                     src_blk_stride: int, src_rep_stride: int, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_common_docstring("whole_reduce_sum")
def whole_reduce_sum(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    dispatcher = OverloadDispatcher("whole_reduce_sum")

    @dispatcher.register(mask=RuntimeInt, repeat_time=RuntimeInt, dst_rep_stride=RuntimeInt, src_blk_stride=RuntimeInt,
                         src_rep_stride=RuntimeInt, is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_time: RuntimeInt, dst_rep_stride: RuntimeInt, src_blk_stride: RuntimeInt,
          src_rep_stride: RuntimeInt, is_set_mask: bool = True):
        builder.create_asc_WholeReduceSumL0Op(dst.to_ir(), src.to_ir(),
                                              _mat(mask, KT.int32).to_ir(),
                                              _mat(repeat_time, KT.int32).to_ir(),
                                              _mat(dst_rep_stride, KT.int32).to_ir(),
                                              _mat(src_blk_stride, KT.int32).to_ir(),
                                              _mat(src_rep_stride, KT.int32).to_ir(), is_set_mask)

    @dispatcher.register(mask=list, repeat_time=RuntimeInt, dst_rep_stride=RuntimeInt, src_blk_stride=RuntimeInt,
                         src_rep_stride=RuntimeInt, is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_time: RuntimeInt, dst_rep_stride: RuntimeInt, src_blk_stride: RuntimeInt,
          src_rep_stride: RuntimeInt, is_set_mask: bool = True):
        mask_ir = [_mat(v, KT.uint64).to_ir() for v in mask]
        builder.create_asc_WholeReduceSumL1Op(dst.to_ir(), src.to_ir(), mask_ir,
                                              _mat(repeat_time, KT.int32).to_ir(),
                                              _mat(dst_rep_stride, KT.int32).to_ir(),
                                              _mat(src_blk_stride, KT.int32).to_ir(),
                                              _mat(src_rep_stride, KT.int32).to_ir(), is_set_mask)

    dispatcher(*args, **kwargs)


@overload
def reduce_max(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, mask: int, repeat_time: int,
               src_rep_stride: int, cal_index: bool = False) -> None:
    ...


@overload
def reduce_max(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, mask: List[int], repeat_time: int,
               src_rep_stride: int, cal_index: bool = False) -> None:
    ...


@overload
def reduce_max(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, count: int,
               cal_index: bool = False) -> None:
    ...


def op_impl(callee: str, dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, args: Tuple[Any],
            kwargs: Dict[str, Any], build_l0: Callable, build_l1: Callable, build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")
    dispatcher = OverloadDispatcher(callee)

    @dispatcher.register_auto
    def _(mask: RuntimeInt, repeat_time: RuntimeInt, src_rep_stride: RuntimeInt, cal_index: bool = False):
        build_l0(dst.to_ir(), src.to_ir(), shared_tmp_buffer.to_ir(),
                 _mat(mask, KT.uint64).to_ir(),
                 _mat(repeat_time, KT.uint8).to_ir(),
                 _mat(src_rep_stride, KT.uint8).to_ir(),
                 _mat(cal_index, KT.bool_).to_ir())

    @dispatcher.register_auto
    def _(mask: list, repeat_time: RuntimeInt, src_rep_stride: RuntimeInt, cal_index: bool = False):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(), shared_tmp_buffer.to_ir(), mask,
                 _mat(repeat_time, KT.uint8).to_ir(),
                 _mat(src_rep_stride, KT.uint8).to_ir(),
                 _mat(cal_index, KT.bool_).to_ir())

    @dispatcher.register_auto
    def _(count: RuntimeInt, cal_index: bool = False):
        build_l2(dst.to_ir(), src.to_ir(), shared_tmp_buffer.to_ir(),
                 _mat(count, KT.uint32).to_ir(),
                 _mat(cal_index, KT.bool_).to_ir())

    dispatcher(*args, **kwargs)


@require_jit
@set_common_docstring("reduce_max")
def reduce_max(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("reduce_max", dst, src, shared_tmp_buffer, args, kwargs, builder.create_asc_ReduceMaxL0Op,
            builder.create_asc_ReduceMaxL1Op, builder.create_asc_ReduceMaxL2Op)


@overload
def reduce_min(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, mask: int, repeat_time: int,
               src_rep_stride: int, cal_index: bool = False) -> None:
    ...


@overload
def reduce_min(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, mask: List[int], repeat_time: int,
               src_rep_stride: int, cal_index: bool = False) -> None:
    ...


@overload
def reduce_min(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, count: int,
               cal_index: bool = False) -> None:
    ...


@require_jit
@set_common_docstring("reduce_min")
def reduce_min(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("reduce_min", dst, src, shared_tmp_buffer, args, kwargs, builder.create_asc_ReduceMinL0Op,
            builder.create_asc_ReduceMinL1Op, builder.create_asc_ReduceMinL2Op)


@overload
def reduce_sum(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, mask: int, repeat_time: int,
               src_rep_stride: int) -> None:
    ...


@overload
def reduce_sum(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, mask: List[int], repeat_time: int,
               src_rep_stride: int) -> None:
    ...


@overload
def reduce_sum(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, count: int) -> None:
    ...


def op_impl_sum(callee: str, dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, args: Tuple[Any],
                kwargs: Dict[str, Any], build_l0: Callable, build_l1: Callable, build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")
    dispatcher = OverloadDispatcher(callee)

    @dispatcher.register_auto
    def _(mask: RuntimeInt, repeat_time: RuntimeInt, src_rep_stride: RuntimeInt):
        build_l0(dst.to_ir(), src.to_ir(), shared_tmp_buffer.to_ir(),
                 _mat(mask, KT.uint64).to_ir(),
                 _mat(repeat_time, KT.uint8).to_ir(),
                 _mat(src_rep_stride, KT.uint8).to_ir())

    @dispatcher.register_auto
    def _(mask: list, repeat_time: RuntimeInt, src_rep_stride: RuntimeInt):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(), shared_tmp_buffer.to_ir(), mask,
                 _mat(repeat_time, KT.uint8).to_ir(),
                 _mat(src_rep_stride, KT.uint8).to_ir())

    @dispatcher.register_auto
    def _(count: RuntimeInt):
        build_l2(dst.to_ir(), src.to_ir(), shared_tmp_buffer.to_ir(), _mat(count, KT.uint32).to_ir())

    dispatcher(*args, **kwargs)


@require_jit
@set_common_docstring("reduce_sum")
def reduce_sum(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl_sum("reduce_sum", dst, src, shared_tmp_buffer, args, kwargs, builder.create_asc_ReduceSumL0Op,
                builder.create_asc_ReduceSumL1Op, builder.create_asc_ReduceSumL2Op)
