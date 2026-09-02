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
from ..core.utils import OverloadDispatcher, require_jit, global_builder
from .utils import set_common_docstring


def check_type(dst_offset: LocalTensor) -> None:
    if dst_offset.dtype != KT.uint32:
        raise TypeError(f"Invalid dst_offset data type, got {dst_offset.dtype}, expect uint32.")


def op_impl(callee: str, dst: LocalTensor, src: LocalTensor, dst_offset: LocalTensor, dst_base: int, args: Tuple[Any],
            kwargs: Dict[str, Any], build_l0: Callable, build_l1: Callable, build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")
    dispatcher = OverloadDispatcher(callee)

    check_type(dst_offset)

    @dispatcher.register_auto
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, src_rep_stride: RuntimeInt):
        build_l0(dst.to_ir(), src.to_ir(), dst_offset.to_ir(),
                 _mat(dst_base, KT.uint32).to_ir(),
                 _mat(mask, KT.uint64).to_ir(),
                 _mat(repeat_times, KT.uint8).to_ir(),
                 _mat(src_rep_stride, KT.uint8).to_ir())

    @dispatcher.register_auto
    def _(mask: list, repeat_times: RuntimeInt, src_rep_stride: RuntimeInt):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(), dst_offset.to_ir(),
                 _mat(dst_base, KT.uint32).to_ir(), mask,
                 _mat(repeat_times, KT.uint8).to_ir(),
                 _mat(src_rep_stride, KT.uint8).to_ir())

    @dispatcher.register_auto
    def _(count: RuntimeInt):
        build_l2(dst.to_ir(), src.to_ir(), dst_offset.to_ir(),
                 _mat(dst_base, KT.uint32).to_ir(),
                 _mat(count, KT.uint32).to_ir())

    dispatcher(*args, **kwargs)


@overload
def scatter(dst: LocalTensor, src: LocalTensor, dst_offset: LocalTensor, dst_base: int, mask: int, repeat_times: int,
            src_rep_stride: int) -> None:
    ...


@overload
def scatter(dst: LocalTensor, src: LocalTensor, dst_offset: LocalTensor, dst_base: int, mask: List[int],
            repeat_times: int, src_rep_stride: int) -> None:
    ...


@overload
def scatter(dst: LocalTensor, src: LocalTensor, dst_offset: LocalTensor, dst_base: int, count: int) -> None:
    ...


@require_jit
@set_common_docstring(api_name="scatter")
def scatter(dst: LocalTensor, src: LocalTensor, dst_offset: LocalTensor, dst_base: int, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("scatter", dst, src, dst_offset, dst_base, args, kwargs, builder.create_asc_ScatterL0Op,
            builder.create_asc_ScatterL1Op, builder.create_asc_ScatterL2Op)
