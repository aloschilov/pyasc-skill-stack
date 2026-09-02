# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List, Union, overload

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.ir_value import RuntimeInt, RuntimeNumeric, RuntimeFloat, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import OverloadDispatcher, require_jit, global_builder
from ..core.types import BinaryRepeatParams, UnaryRepeatParams
from ..core.enums import CMPMODE, SelMode
from .utils import set_common_docstring


@overload
def compare(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, cmp_mode: CMPMODE, count: int) -> None:
    ...


@overload
def compare(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, cmp_mode: CMPMODE, mask: int, repeat_times: int,
            repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def compare(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, cmp_mode: CMPMODE, mask: List[int],
            repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def compare(src0: LocalTensor, src1: LocalTensor, cmp_mode: CMPMODE, mask: List[int], repeat_params: BinaryRepeatParams,
            is_set_mask: bool = True) -> None:
    ...


@overload
def compare(src0: LocalTensor, src1: LocalTensor, cmp_mode: CMPMODE, mask: int, repeat_params: BinaryRepeatParams,
            is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_common_docstring(api_name="compare")
def compare(*args, **kwargs) -> None:
    dispatcher = OverloadDispatcher(__name__)
    builder = global_builder.get_ir_builder()

    @dispatcher.register(dst=LocalTensor, src0=LocalTensor, src1=LocalTensor, cmp_mode=CMPMODE, mask=RuntimeInt,
                         repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams)
    def _(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, cmp_mode: CMPMODE, mask: RuntimeInt,
          repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams, is_set_mask: bool = True):
        builder.create_asc_CompareL0Op(dst.to_ir(), src0.to_ir(), src1.to_ir(), ir.CMPMODE.symbolize(cmp_mode),
                                       _mat(mask, KT.uint64).to_ir(),
                                       _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(dst=LocalTensor, src0=LocalTensor, src1=LocalTensor, cmp_mode=CMPMODE, mask=list,
                         repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams)
    def _(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, cmp_mode: CMPMODE, mask: list,
          repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams, is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        builder.create_asc_CompareL1Op(dst.to_ir(), src0.to_ir(), src1.to_ir(), ir.CMPMODE.symbolize(cmp_mode), mask,
                                       _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register_auto
    def _(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, cmp_mode: CMPMODE, count: RuntimeInt):
        builder.create_asc_CompareL2Op(dst.to_ir(), src0.to_ir(), src1.to_ir(), ir.CMPMODE.symbolize(cmp_mode),
                                       _mat(count, KT.int32).to_ir())

    @dispatcher.register(src0=LocalTensor, src1=LocalTensor, cmp_mode=CMPMODE, mask=RuntimeInt,
                         repeat_params=BinaryRepeatParams)
    def _(src0: LocalTensor, src1: LocalTensor, cmp_mode: CMPMODE, mask: RuntimeInt, repeat_params: BinaryRepeatParams,
          is_set_mask: bool = True):
        builder.create_asc_CompareRL0Op(src0.to_ir(), src1.to_ir(), ir.CMPMODE.symbolize(cmp_mode),
                                        _mat(mask, KT.uint64).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(src0=LocalTensor, src1=LocalTensor, cmp_mode=CMPMODE, mask=list,
                         repeat_params=BinaryRepeatParams)
    def _(src0: LocalTensor, src1: LocalTensor, cmp_mode: CMPMODE, mask: list, repeat_params: BinaryRepeatParams,
          is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        builder.create_asc_CompareRL1Op(src0.to_ir(), src1.to_ir(), ir.CMPMODE.symbolize(cmp_mode), mask,
                                        repeat_params.to_ir(), is_set_mask)

    dispatcher(*args, **kwargs)


@overload
def compare_scalar(dst: LocalTensor, src0: LocalTensor, src1_scalar: Union[int, float], cmp_mode: CMPMODE,
                   count: int) -> None:
    ...


@overload
def compare_scalar(dst: LocalTensor, src0: LocalTensor, src1_scalar: Union[int, float], cmp_mode: CMPMODE, mask: int,
                   repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def compare_scalar(dst: LocalTensor, src0: LocalTensor, src1_scalar: Union[int,
                                                                           float], cmp_mode: CMPMODE, mask: List[int],
                   repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@require_jit
@set_common_docstring(api_name="compare_scalar")
def compare_scalar(dst: LocalTensor, src0: LocalTensor, src1_scalar: RuntimeNumeric, cmp_mode: CMPMODE, *args,
                   **kwargs) -> None:
    dispatcher = OverloadDispatcher(__name__)
    builder = global_builder.get_ir_builder()
    src1_scalar = _mat(src1_scalar, src0.dtype).to_ir()

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams)
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        builder.create_asc_CompareScalarL0Op(dst.to_ir(), src0.to_ir(), src1_scalar, ir.CMPMODE.symbolize(cmp_mode),
                                             _mat(mask, KT.uint64).to_ir(),
                                             _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams)
    def _(mask: list, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        builder.create_asc_CompareScalarL1Op(dst.to_ir(), src0.to_ir(), src1_scalar, ir.CMPMODE.symbolize(cmp_mode),
                                             mask,
                                             _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)

    @dispatcher.register_auto
    def _(count: RuntimeInt):
        builder.create_asc_CompareScalarL2Op(dst.to_ir(), src0.to_ir(), src1_scalar, ir.CMPMODE.symbolize(cmp_mode),
                                             _mat(count, KT.int32).to_ir())

    dispatcher(*args, **kwargs)


@require_jit
@set_common_docstring(api_name="get_cmp_mask")
def get_cmp_mask(dst: LocalTensor) -> None:
    build = global_builder.get_ir_builder()
    build.create_asc_GetCmpMaskOp(dst.to_ir())


@require_jit
@set_common_docstring(api_name="set_cmp_mask")
def set_cmp_mask(src: LocalTensor) -> None:
    build = global_builder.get_ir_builder()
    build.create_asc_SetCmpMaskOp(src.to_ir())


@overload
def select(dst: LocalTensor, sel_mask: LocalTensor, src0: LocalTensor, src1: float, sel_mode: SelMode,
           count: int) -> None:
    ...


@overload
def select(dst: LocalTensor, sel_mask: LocalTensor, src0: LocalTensor, src1: LocalTensor, sel_mode: SelMode,
           count: int) -> None:
    ...


@overload
def select(dst: LocalTensor, sel_mask: LocalTensor, src0: LocalTensor, src1: float, sel_mode: SelMode, mask: List[int],
           repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def select(dst: LocalTensor, sel_mask: LocalTensor, src0: LocalTensor, src1: float, sel_mode: SelMode, mask: int,
           repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def select(dst: LocalTensor, sel_mask: LocalTensor, src0: LocalTensor, repeat_times: int,
           repeat_params: BinaryRepeatParams) -> None:
    ...


@overload
def select(dst: LocalTensor, sel_mask: LocalTensor, src0: LocalTensor, src1: LocalTensor, sel_mode: SelMode,
           mask: List[int], repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def select(dst: LocalTensor, sel_mask: LocalTensor, src0: LocalTensor, src1: LocalTensor, sel_mode: SelMode, mask: int,
           repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
    ...


@overload
def select(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, repeat_times: int, repeat_params: BinaryRepeatParams,
           sel_mode: SelMode) -> None:
    ...


@require_jit
@set_common_docstring(api_name="select")
def select(dst: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()

    if len(args) < 2:
        raise TypeError("select: invalid arguments")
    if not hasattr(builder, "_select_mask_list_cache"):
        builder._select_mask_list_cache = {}
    if (len(args) == 2 and "repeat_times" in kwargs and "repeat_params" in kwargs and "sel_mode" in kwargs):
        src0, src1 = args
        builder.create_asc_SelectRegMaskOp(
            dst.to_ir(),
            src0.to_ir(),
            src1.to_ir(),
            _mat(kwargs["repeat_times"], KT.int8).to_ir(),
            kwargs["repeat_params"].to_ir(),
            ir.SELMODE.symbolize(kwargs["sel_mode"]),
        )
        return
    sel_mask = args[0]
    src0 = args[1]
    rest = args[2:]
    dispatcher = OverloadDispatcher(__name__)

    @dispatcher.register(src1=RuntimeFloat, sel_mode=SelMode, count=RuntimeInt)
    def _(src1, sel_mode, count):
        builder.create_asc_SelectScalarL2Op(
            dst.to_ir(),
            sel_mask.to_ir(),
            src0.to_ir(),
            _mat(src1, src0.dtype).to_ir(),
            ir.SELMODE.symbolize(sel_mode),
            _mat(count, KT.uint32).to_ir(),
        )

    @dispatcher.register(src1=LocalTensor, sel_mode=SelMode, count=RuntimeInt)
    def _(src1, sel_mode, count):
        builder.create_asc_SelectL2Op(
            dst.to_ir(),
            sel_mask.to_ir(),
            src0.to_ir(),
            src1.to_ir(),
            ir.SELMODE.symbolize(sel_mode),
            _mat(count, KT.uint32).to_ir(),
        )

    @dispatcher.register(
        src1=RuntimeFloat,
        sel_mode=SelMode,
        mask=list,
        repeat_times=RuntimeInt,
        repeat_params=BinaryRepeatParams,
    )
    def _(src1, sel_mode, mask, repeat_times, repeat_params, is_set_mask: bool = True):
        key = tuple(int(v) for v in mask)
        mask_ir = builder._select_mask_list_cache.get(key)
        if mask_ir is None:
            mask_ir = [_mat(v, KT.uint64).to_ir() for v in mask]
            builder._select_mask_list_cache[key] = mask_ir
        builder.create_asc_SelectScalarL1Op(
            dst.to_ir(),
            sel_mask.to_ir(),
            src0.to_ir(),
            _mat(src1, src0.dtype).to_ir(),
            ir.SELMODE.symbolize(sel_mode),
            mask_ir,
            _mat(repeat_times, KT.int8).to_ir(),
            repeat_params.to_ir(),
            is_set_mask,
        )

    @dispatcher.register(
        src1=RuntimeFloat,
        sel_mode=SelMode,
        mask=RuntimeInt,
        repeat_times=RuntimeInt,
        repeat_params=BinaryRepeatParams,
    )
    def _(src1, sel_mode, mask, repeat_times, repeat_params, is_set_mask: bool = True):
        builder.create_asc_SelectScalarL0Op(
            dst.to_ir(),
            sel_mask.to_ir(),
            src0.to_ir(),
            _mat(src1, src0.dtype).to_ir(),
            ir.SELMODE.symbolize(sel_mode),
            _mat(mask, KT.uint64).to_ir(),
            _mat(repeat_times, KT.int8).to_ir(),
            repeat_params.to_ir(),
            is_set_mask,
        )

    @dispatcher.register(repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams)
    def _(repeat_times, repeat_params):
        builder.create_asc_SelectScalarRegMaskOp(
            dst.to_ir(),
            sel_mask.to_ir(),
            src0.to_ir(),
            _mat(repeat_times, KT.int8).to_ir(),
            repeat_params.to_ir(),
        )

    @dispatcher.register(
        src1=LocalTensor,
        sel_mode=SelMode,
        mask=list,
        repeat_times=RuntimeInt,
        repeat_params=BinaryRepeatParams,
    )
    def _(src1, sel_mode, mask, repeat_times, repeat_params, is_set_mask: bool = True):
        key = tuple(int(v) for v in mask)
        mask_ir = builder._select_mask_list_cache.get(key)
        if mask_ir is None:
            mask_ir = [_mat(v, KT.uint64).to_ir() for v in mask]
            builder._select_mask_list_cache[key] = mask_ir
        builder.create_asc_SelectL1Op(
            dst.to_ir(),
            sel_mask.to_ir(),
            src0.to_ir(),
            src1.to_ir(),
            ir.SELMODE.symbolize(sel_mode),
            mask_ir,
            _mat(repeat_times, KT.int8).to_ir(),
            repeat_params.to_ir(),
            is_set_mask,
        )

    @dispatcher.register(
        src1=LocalTensor,
        sel_mode=SelMode,
        mask=RuntimeInt,
        repeat_times=RuntimeInt,
        repeat_params=BinaryRepeatParams,
    )
    def _(src1, sel_mode, mask, repeat_times, repeat_params, is_set_mask: bool = True):
        builder.create_asc_SelectL0Op(
            dst.to_ir(),
            sel_mask.to_ir(),
            src0.to_ir(),
            src1.to_ir(),
            ir.SELMODE.symbolize(sel_mode),
            _mat(mask, KT.uint64).to_ir(),
            _mat(repeat_times, KT.int8).to_ir(),
            repeat_params.to_ir(),
            is_set_mask,
        )

    dispatcher(*rest, **kwargs)
