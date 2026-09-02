# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List, Union, overload
from ..core.dtype import KnownTypes as KT
from ..core.ir_value import RuntimeInt, RuntimeNumeric, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import DefaultValued, OverloadDispatcher, require_jit, global_builder
from .utils import set_common_docstring


@overload
def duplicate(dst: LocalTensor, scalar: Union[int, float], count: int) -> None:
    ...


@overload
def duplicate(dst: LocalTensor, scalar: Union[int, float], mask: int, repeat_times: int, dst_block_stride: int,
              dst_repeat_stride: int, is_set_mask: bool = True) -> None:
    ...


@overload
def duplicate(dst: LocalTensor, scalar: Union[int, float], mask: List[int], repeat_times: int, dst_block_stride: int,
              dst_repeat_stride: int, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_common_docstring(api_name="duplicate")
def duplicate(dst: LocalTensor, scalar: RuntimeNumeric, *args, **kwargs) -> None:
    dispatcher = OverloadDispatcher(__name__)
    builder = global_builder.get_ir_builder()

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, dst_block_stride=RuntimeInt,
                         dst_repeat_stride=RuntimeInt, is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, dst_block_stride: RuntimeInt, dst_repeat_stride: RuntimeInt,
          is_set_mask: bool = True):
        builder.create_asc_DuplicateL0Op(dst.to_ir(),
                                         _mat(scalar, dst.dtype).to_ir(),
                                         _mat(mask, KT.uint64).to_ir(),
                                         _mat(repeat_times, KT.int8).to_ir(),
                                         _mat(dst_block_stride, KT.int8).to_ir(),
                                         _mat(dst_repeat_stride, KT.int8).to_ir(), is_set_mask)

    @dispatcher.register(mask=list, repeat_times=RuntimeInt, dst_block_stride=RuntimeInt, dst_repeat_stride=RuntimeInt,
                         is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_times: RuntimeInt, dst_block_stride: RuntimeInt, dst_repeat_stride: RuntimeInt,
          is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        builder.create_asc_DuplicateL1Op(dst.to_ir(),
                                         _mat(scalar, dst.dtype).to_ir(), mask,
                                         _mat(repeat_times, KT.int8).to_ir(),
                                         _mat(dst_block_stride, KT.int8).to_ir(),
                                         _mat(dst_repeat_stride, KT.int8).to_ir(), is_set_mask)

    @dispatcher.register(count=RuntimeInt, is_set_mask=DefaultValued(bool, True))
    def _(count: RuntimeInt, is_set_mask: bool = True):
        builder.create_asc_DuplicateL2Op(dst.to_ir(), _mat(scalar, dst.dtype).to_ir(), _mat(count, KT.int32).to_ir())

    dispatcher(*args, **kwargs)
