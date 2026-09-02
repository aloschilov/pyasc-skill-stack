# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Union, overload

from ..core.dtype import KnownTypes as KT
from ..core.enums import GatherMaskMode
from ..core.ir_value import materialize_ir_value as _mat, PlainValue, RuntimeBool, RuntimeInt
from ..core.tensor import LocalTensor
from ..core.utils import require_jit, global_builder, DefaultValued
from ..core.types import GatherMaskParams
from .utils import OverloadDispatcher, set_common_docstring


def check_type_gather_mask(dst: LocalTensor, src0: LocalTensor, src1_pattern: Union[LocalTensor, int]) -> None:
    """
    Check data type constraints for GatherMask operation.
    
    According to GatherMask specification:
    - T (dst and src0 data types):
        Atlas inference AI Core: half/uint16_t/int16_t/float/uint32_t/int32_t
        Atlas A2 training/Atlas 800I A2 inference/A200I A2 Box: half/bfloat16_t/uint16_t/int16_t/float/uint32_t/int32_t
        Atlas 200I/500 A2 inference: half/uint16_t/int16_t/float/uint32_t/int32_t
        Atlas A3 training/A3 inference: half/bfloat16_t/uint16_t/int16_t/float/uint32_t/int32_t
    
    - U (src1_pattern data type when LocalTensor):
        uint16_t/uint32_t
        When dst is half/uint16_t/int16_t: src1_pattern should be uint16_t
        When dst is float/uint32_t/int32_t: src1_pattern should be uint32_t
    """
    valid_dst_src0_types = [
        KT.half,
        KT.uint16,
        KT.int16,
        KT.float_,
        KT.uint32,
        KT.int32,
    ]
    if dst.dtype not in valid_dst_src0_types:
        raise TypeError(f"Invalid dst data type for GatherMask: {dst.dtype}. "
                        f"Supported types: half, uint16, int16, float, uint32, int32")
    if src0.dtype not in valid_dst_src0_types:
        raise TypeError(f"Invalid src0 data type for GatherMask: {src0.dtype}. "
                        f"Supported types: half, uint16, int16, float, uint32, int32")
    if dst.dtype != src0.dtype:
        raise TypeError(f"dst and src0 must have same data type. Got dst={dst.dtype}, src0={src0.dtype}")
    if isinstance(src1_pattern, LocalTensor):
        if dst.dtype in [KT.half, KT.uint16, KT.int16]:
            if src1_pattern.dtype != KT.uint16:
                raise TypeError(f"For dst data type {dst.dtype}, src1_pattern must be uint16. Got {src1_pattern.dtype}")
        elif dst.dtype in [KT.float_, KT.uint32, KT.int32]:
            if src1_pattern.dtype != KT.uint32:
                raise TypeError(f"For dst data type {dst.dtype}, src1_pattern must be uint32. Got {src1_pattern.dtype}")
        else:
            raise TypeError(f"Unsupported dst data type for src1_pattern validation: {dst.dtype}")
    elif isinstance(src1_pattern, int):
        if not (1 <= src1_pattern <= 7):
            raise ValueError(f"Built-in src1_pattern must be between 1 and 7. Got {src1_pattern}")
    else:
        raise TypeError(f"src1_pattern must be either LocalTensor or int. Got {type(src1_pattern)}")


@overload
def gather_mask(dst: LocalTensor, src0: LocalTensor, src1_pattern: LocalTensor, reduce_mode: bool, mask: int,
                params: GatherMaskParams, gather_mask_mode=GatherMaskMode.DEFAULT) -> int:
    ...


@overload
def gather_mask(dst: LocalTensor, src0: LocalTensor, src1_pattern: int, reduce_mode: bool, mask: int,
                params: GatherMaskParams, gather_mask_mode=GatherMaskMode.DEFAULT) -> int:
    ...


@require_jit
@set_common_docstring("gather_mask")
def gather_mask(dst: LocalTensor, src0: LocalTensor, *args, **kwargs) -> RuntimeInt:
    builder = global_builder.get_ir_builder()

    dispatcher = OverloadDispatcher("gather_mask")

    @dispatcher.register(src1_pattern=LocalTensor, reduce_mode=RuntimeBool, mask=RuntimeInt, params=GatherMaskParams,
                         gather_mask_mode=DefaultValued(GatherMaskMode, GatherMaskMode.DEFAULT))
    def _(src1_pattern: LocalTensor, reduce_mode: RuntimeBool, mask: RuntimeInt, params: GatherMaskParams,
          gather_mask_mode: GatherMaskMode):
        check_type_gather_mask(dst, src0, src1_pattern)
        rsvd_cnt = builder.create_asc_GatherMaskAndResult(KT.uint64.to_ir(), dst.to_ir(), src0.to_ir(),
                                                          src1_pattern.to_ir(),
                                                          _mat(reduce_mode, KT.bool_).to_ir(),
                                                          _mat(mask, KT.uint32).to_ir(), params.to_ir(),
                                                          gather_mask_mode)
        return PlainValue(rsvd_cnt)

    @dispatcher.register(src1_pattern=RuntimeInt, reduce_mode=RuntimeBool, mask=RuntimeInt, params=GatherMaskParams,
                         gather_mask_mode=DefaultValued(GatherMaskMode, GatherMaskMode.DEFAULT))
    def _(src1_pattern: RuntimeInt, reduce_mode: RuntimeBool, mask: RuntimeInt, params: GatherMaskParams,
          gather_mask_mode: GatherMaskMode):
        check_type_gather_mask(dst, src0, src1_pattern)

        rsvd_cnt = builder.create_asc_GatherMaskAndResult(KT.uint64.to_ir(), dst.to_ir(), src0.to_ir(),
                                                          _mat(src1_pattern, KT.uint8).to_ir(),
                                                          _mat(reduce_mode, KT.bool_).to_ir(),
                                                          _mat(mask, KT.uint32).to_ir(), params.to_ir(),
                                                          gather_mask_mode)
        return PlainValue(rsvd_cnt)

    return dispatcher(*args, **kwargs)


@require_jit
def get_gather_mask_remain_count() -> int:
    builder = global_builder.get_ir_builder()
    result = builder.create_asc_GetGatherMaskRemainCountOp(builder.get_ui64_type())
    return result
