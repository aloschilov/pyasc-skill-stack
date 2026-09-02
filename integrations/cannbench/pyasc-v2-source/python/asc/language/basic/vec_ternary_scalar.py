# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List, Union, overload

from .utils import vec_ternary_scalar_op_impl as op_impl
from ..core.ir_value import RuntimeNumeric
from ..core.tensor import LocalTensor
from ..core.utils import require_jit, global_builder
from ..core.types import UnaryRepeatParams
from .utils import set_common_docstring


@overload
def axpy(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: int, repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def axpy(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], mask: List[int], repeat_times: int,
         repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def axpy(dst: LocalTensor, src: LocalTensor, scalar: Union[int, float], count: int) -> None:
    ...


@require_jit
@set_common_docstring("axpy")
def axpy(dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("axpy", dst, src, scalar, args, kwargs, builder.create_asc_AxpyL0Op, builder.create_asc_AxpyL1Op,
            builder.create_asc_AxpyL2Op)
