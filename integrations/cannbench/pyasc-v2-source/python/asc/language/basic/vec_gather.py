# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
from ..core.types import GatherRepeatParams
from ..core.utils import OverloadDispatcher, require_jit, global_builder
from .utils import set_common_docstring


@require_jit
@set_common_docstring(api_name="gatherb")
def gatherb(dst: LocalTensor, src0: LocalTensor, offset: LocalTensor, repeat_times: int,
            repeat_params: GatherRepeatParams) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_GatherbL0Op(dst.to_ir(), src0.to_ir(), offset.to_ir(),
                                   _mat(repeat_times, KT.uint8).to_ir(), repeat_params.to_ir())


def op_impl(callee: str, dst: LocalTensor, src: LocalTensor, src_offset: LocalTensor, src_base: int, args: Tuple[Any],
            kwargs: Dict[str, Any], build_l0: Callable, build_l1: Callable, build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")

    dispatcher = OverloadDispatcher(callee)

    @dispatcher.register_auto
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, dst_rep_stride: RuntimeInt):
        build_l0(dst.to_ir(), src.to_ir(), src_offset.to_ir(),
                 _mat(src_base, KT.uint32).to_ir(),
                 _mat(mask, KT.uint64).to_ir(),
                 _mat(repeat_times, KT.uint8).to_ir(),
                 _mat(dst_rep_stride, KT.uint16).to_ir())

    @dispatcher.register_auto
    def _(mask: list, repeat_times: RuntimeInt, dst_rep_stride: RuntimeInt):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(), src_offset.to_ir(),
                 _mat(src_base, KT.uint32).to_ir(), mask,
                 _mat(repeat_times, KT.uint8).to_ir(),
                 _mat(dst_rep_stride, KT.uint16).to_ir())

    @dispatcher.register_auto
    def _(count: RuntimeInt):
        build_l2(dst.to_ir(), src.to_ir(), src_offset.to_ir(),
                 _mat(src_base, KT.uint32).to_ir(),
                 _mat(count, KT.uint32).to_ir())

    dispatcher(*args, **kwargs)


@overload
def gather(dst: LocalTensor, src: LocalTensor, src_offset: LocalTensor, src_base: int, mask: int, repeat_times: int,
           dst_rep_stride: int) -> None:
    ...


@overload
def gather(dst: LocalTensor, src: LocalTensor, src_offset: LocalTensor, src_base: int, mask: List[int],
           repeat_times: int, dst_rep_stride: int) -> None:
    ...


@overload
def gather(dst: LocalTensor, src: LocalTensor, src_offset: LocalTensor, src_base: int, count: int) -> None:
    ...


@require_jit
@set_common_docstring(api_name="gather")
def gather(dst: LocalTensor, src: LocalTensor, src_offset: LocalTensor, src_base: int, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    dispatcher = OverloadDispatcher(__name__)

    @dispatcher.register_auto
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, dst_rep_stride: RuntimeInt):
        builder.create_asc_GatherL0Op(dst.to_ir(), src.to_ir(), src_offset.to_ir(),
                                      _mat(src_base, KT.uint32).to_ir(),
                                      _mat(mask, KT.uint64).to_ir(),
                                      _mat(repeat_times, KT.uint8).to_ir(),
                                      _mat(dst_rep_stride, KT.uint16).to_ir())

    @dispatcher.register_auto
    def _(mask: list, repeat_times: RuntimeInt, dst_rep_stride: RuntimeInt):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        builder.create_asc_GatherL1Op(dst.to_ir(), src.to_ir(), src_offset.to_ir(),
                                      _mat(src_base, KT.uint32).to_ir(), mask,
                                      _mat(repeat_times, KT.uint8).to_ir(),
                                      _mat(dst_rep_stride, KT.uint16).to_ir())

    @dispatcher.register_auto
    def _(count: RuntimeInt):
        builder.create_asc_GatherL2Op(dst.to_ir(), src.to_ir(), src_offset.to_ir(),
                                      _mat(src_base, KT.uint32).to_ir(),
                                      _mat(count, KT.uint32).to_ir())

    dispatcher(*args, **kwargs)
