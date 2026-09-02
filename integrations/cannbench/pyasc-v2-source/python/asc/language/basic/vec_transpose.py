# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload, List

from ..core.ir_value import RuntimeInt, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import require_jit, global_builder, OverloadDispatcher
from ..core.types import TransposeParamsExt, TransDataTo5HDParams, KnownTypes
from .utils import check_type_transpose, check_type_5hd, set_common_docstring


@overload
def transpose(dst: LocalTensor, src: LocalTensor) -> None:
    ...


@overload
def transpose(dst: LocalTensor, src: LocalTensor, shared_tmp_buffer: LocalTensor, params: TransposeParamsExt) -> None:
    ...


@require_jit
@set_common_docstring(api_name="transpose")
def transpose(dst: LocalTensor, src: LocalTensor, *args, **kwargs) -> None:
    check_type_transpose("transpose", dst, src, *args)
    builder = global_builder.get_ir_builder()

    if not args and not kwargs:
        builder.create_asc_TransposeOp(dst.to_ir(), src.to_ir())
        return

    dispatcher = OverloadDispatcher("transpose")

    @dispatcher.register_auto
    def _(shared_tmp_buffer: LocalTensor, params: TransposeParamsExt):
        builder.create_asc_TransposeExtOp(dst.to_ir(), src.to_ir(), shared_tmp_buffer.to_ir(), params.to_ir())

    dispatcher(*args, **kwargs)


TensorList = List[LocalTensor]
AddrList = List[RuntimeInt]


@overload
def trans_data_to_5hd(dst_list: TensorList, src_list: TensorList, params: TransDataTo5HDParams) -> None:
    ...


@overload
def trans_data_to_5hd(dst_list: AddrList, src_list: AddrList, params: TransDataTo5HDParams) -> None:
    ...


@overload
def trans_data_to_5hd(dst: LocalTensor, src: LocalTensor, params: TransDataTo5HDParams) -> None:
    ...


@require_jit
@set_common_docstring(api_name="trans_data_to_5hd")
def trans_data_to_5hd(dst_or_list, src_or_list, params: TransDataTo5HDParams, *args, **kwargs) -> None:
    check_type_5hd("trans_data_to_5hd", dst_or_list, src_or_list)
    builder = global_builder.get_ir_builder()

    if isinstance(dst_or_list, LocalTensor):
        builder.create_asc_TransDataTo5HDOp(dst_or_list.to_ir(), src_or_list.to_ir(), params.to_ir())
    elif isinstance(dst_or_list, list):
        if not dst_or_list:
            return
        if isinstance(dst_or_list[0], LocalTensor):
            dst_ir_list = [t.to_ir() for t in dst_or_list]
            src_ir_list = [t.to_ir() for t in src_or_list]
            builder.create_asc_TransDataTo5HDTensorListOp(dst_ir_list, src_ir_list, params.to_ir())
        else:
            dst_ir_list = [_mat(addr, KnownTypes.uint64).to_ir() for addr in dst_or_list]
            src_ir_list = [_mat(addr, KnownTypes.uint64).to_ir() for addr in src_or_list]
            builder.create_asc_TransDataTo5HDUintListOp(dst_ir_list, src_ir_list, params.to_ir())
