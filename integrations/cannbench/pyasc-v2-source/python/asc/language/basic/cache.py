# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.tensor import GlobalTensor
from ..core.enums import CacheLine, DcciDst
from ..core.ir_value import PlainValue, RuntimeInt, materialize_ir_value as _mat
from ..core.utils import OverloadDispatcher, require_jit, global_builder
from .utils import set_common_docstring


@overload
def data_cache_clean_and_invalid(entire_type: CacheLine, dcci_dst: DcciDst, dst: GlobalTensor) -> None:
    ...


@overload
def data_cache_clean_and_invalid(entire_type: CacheLine, dst: GlobalTensor) -> None:
    ...


@require_jit
@set_common_docstring(api_name="data_cache_clean_and_invalid")
def data_cache_clean_and_invalid(entire_type: CacheLine, dst: GlobalTensor, dcci_dst: DcciDst = None) -> None:
    dispatcher = OverloadDispatcher(__name__)
    builder = global_builder.get_ir_builder()

    @dispatcher.register_auto
    def _(entire_type: CacheLine, dcci_dst: DcciDst, dst: GlobalTensor):
        builder.create_asc_DataCacheCleanAndInvalidGlobalOp(dst.to_ir(), ir.CacheLine.symbolize(entire_type),
                                                            ir.DcciDst.symbolize(dcci_dst))

    @dispatcher.register_auto
    def _(entire_type: CacheLine, dst: GlobalTensor):
        builder.create_asc_DataCacheCleanAndInvalidGlobalNoDcciDstOp(dst.to_ir(), ir.CacheLine.symbolize(entire_type))

    if dcci_dst is not None:
        dispatcher(entire_type, dcci_dst, dst)
    else:
        dispatcher(entire_type, dst)


@overload
def get_icache_preload_status() -> int:
    ...


@require_jit
@set_common_docstring(api_name="get_icache_preload_status")
def get_icache_preload_status() -> RuntimeInt:
    builder = global_builder.get_ir_builder()
    return PlainValue(builder.create_asc_GetICachePreloadStatusOp(KT.int64.to_ir()))


@overload
def icache_preload(pre_fetch_len: int) -> None:
    ...


@require_jit
@set_common_docstring(api_name="icache_preload")
def icache_preload(pre_fetch_len: RuntimeInt) -> None:
    builder = global_builder.get_ir_builder()
    if not isinstance(builder, ir.Builder):
        raise TypeError("global_builder must provide an ir.Builder")
    builder.create_asc_ICachePreLoadOp(_mat(pre_fetch_len, KT.int64).to_ir())
