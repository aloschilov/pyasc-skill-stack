# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload, Union, List, Optional

from ..._C import ir
from ..core.dtype import KnownTypes
from ..core.enums import TPosition
from ..core.ir_value import RuntimeInt, materialize_ir_value as _mat
from ..core.tensor import BaseTensor, GlobalTensor, LocalTensor
from ..core.types import (CopyRepeatParams, DataCopyEnhancedParams, DataCopyParams, DataCopyCO12DstParams,
                          DataCopyExtParams, DataCopyPadExtParams, DataCopyPadParams, LoadImageToLocalParams,
                          Nd2NzParams, Nz2NdParamsFull)
from ..core.utils import OverloadDispatcher, require_jit, global_builder
from .utils import set_common_docstring


@overload
def copy(dst: LocalTensor, src: LocalTensor, mask: int, repeat_time: int, repeat_params: CopyRepeatParams) -> None:
    ...


@overload
def copy(dst: LocalTensor, src: LocalTensor, mask: List[int], repeat_time: int,
         repeat_params: CopyRepeatParams) -> None:
    ...


@require_jit
@set_common_docstring(api_name="copy")
def copy(dst: BaseTensor, src: BaseTensor, mask: Union[list, RuntimeInt], repeat_time: RuntimeInt,
         repeat_params: CopyRepeatParams, is_set_mask: bool = True) -> None:

    if is_set_mask not in (True, False):
        raise TypeError(f"The 'is_set_mask' argument must be a boolean literal (True or False), "
                        f"but got {is_set_mask} of type {type(is_set_mask).__name__}. "
                        f"This parameter must be a compile-time constant.")

    builder = global_builder.get_ir_builder()

    is_set_mask_val = _mat(is_set_mask, KnownTypes.bool_).to_ir()
    repeat_time_val = _mat(repeat_time, KnownTypes.uint8).to_ir()

    if isinstance(mask, list):
        mask_val = [_mat(v, KnownTypes.uint64).to_ir() for v in mask]
        builder.create_asc_CopyL0Op(dst.to_ir(), src.to_ir(), mask_val, repeat_time_val, repeat_params.to_ir(),
                                    is_set_mask_val)
    elif isinstance(mask, int):
        mask_val = _mat(mask, KnownTypes.uint64).to_ir()
        builder.create_asc_CopyL1Op(dst.to_ir(), src.to_ir(), mask_val, repeat_time_val, repeat_params.to_ir(),
                                    is_set_mask_val)
    else:
        raise TypeError(f"Unsupported type for mask: {type(mask)}")


@overload
def data_copy(dst: LocalTensor, src: GlobalTensor, count: int) -> None:
    ...


@overload
def data_copy(dst: LocalTensor, src: LocalTensor, count: int) -> None:
    ...


@overload
def data_copy(dst: GlobalTensor, src: LocalTensor, count: int) -> None:
    ...


@overload
def data_copy(dst: LocalTensor, src: GlobalTensor, repeat_params: DataCopyParams) -> None:
    ...


@overload
def data_copy(dst: LocalTensor, src: LocalTensor, repeat_params: DataCopyParams) -> None:
    ...


@overload
def data_copy(dst: GlobalTensor, src: LocalTensor, repeat_params: DataCopyParams) -> None:
    ...


@overload
def data_copy(dst: LocalTensor, src: GlobalTensor, intri_params: DataCopyParams,
              enhanced_params: DataCopyEnhancedParams) -> None:
    ...


@overload
def data_copy(dst: LocalTensor, src: LocalTensor, intri_params: DataCopyParams,
              enhanced_params: DataCopyEnhancedParams) -> None:
    ...


@overload
def data_copy(dst: GlobalTensor, src: LocalTensor, intri_params: DataCopyParams,
              enhanced_params: DataCopyEnhancedParams) -> None:
    ...


@overload
def data_copy(dst: GlobalTensor, src: LocalTensor, slice_list1: list, slice_list2: list, dim_value: int) -> None:
    ...


@overload
def data_copy(dst: LocalTensor, src: GlobalTensor, slice_list1: list, slice_list2: list, dim_value: int) -> None:
    ...


@overload
def data_copy(dst: LocalTensor, src: GlobalTensor, intri_params: Nd2NzParams) -> None:
    ...


@overload
def data_copy(dst: LocalTensor, src: LocalTensor, intri_params: Nd2NzParams) -> None:
    ...


@overload
def data_copy(dst: GlobalTensor, src: LocalTensor, intri_params: Nz2NdParamsFull) -> None:
    ...


@overload
def data_copy(dst: GlobalTensor, src: LocalTensor, intri_params: DataCopyCO12DstParams) -> None:
    ...


@overload
def data_copy(dst: LocalTensor, src: LocalTensor, intri_params: DataCopyCO12DstParams) -> None:
    ...


@require_jit
@set_common_docstring(api_name="data_copy")
def data_copy(dst: BaseTensor, src: BaseTensor, *args, **kwargs) -> None:
    dispatcher = OverloadDispatcher(__name__)
    builder = global_builder.get_ir_builder()

    @dispatcher.register_auto
    def _(repeat_params: DataCopyParams):
        builder.create_asc_DataCopyL0Op(dst.to_ir(), src.to_ir(), repeat_params.to_ir())

    @dispatcher.register_auto
    def _(count: RuntimeInt):
        builder.create_asc_DataCopyL2Op(dst.to_ir(), src.to_ir(), _mat(count, KnownTypes.int_).to_ir())

    @dispatcher.register_auto
    def _(repeat_params: DataCopyParams, enhanced_params: DataCopyEnhancedParams):
        builder.create_asc_DataCopyEnhancedOp(dst.to_ir(), src.to_ir(), repeat_params.to_ir(), enhanced_params.to_ir())

    @dispatcher.register_auto
    def _(slice_list1: list, slice_list2: list, dim_value: RuntimeInt):
        slice_list1 = [value.to_ir() for value in slice_list1]
        slice_list2 = [value.to_ir() for value in slice_list2]
        builder.create_asc_DataCopySliceOp(dst.to_ir(), src.to_ir(), slice_list1, slice_list2,
                                           _mat(dim_value, KnownTypes.uint32).to_ir())

    @dispatcher.register_auto
    def _(intri_params: Nd2NzParams):
        builder.create_asc_DataCopyNd2NzOp(dst.to_ir(), src.to_ir(), intri_params.to_ir())

    @dispatcher.register_auto
    def _(intri_params: Nz2NdParamsFull):
        builder.create_asc_DataCopyNz2NdOp(dst.to_ir(), src.to_ir(), intri_params.to_ir())

    @dispatcher.register_auto
    def _(intri_params: DataCopyCO12DstParams):
        builder.create_asc_DataCopyCO12DstOp(dst.to_ir(), src.to_ir(), intri_params.to_ir())

    dispatcher(*args, **kwargs)


@overload
def data_copy_pad(dst: LocalTensor, src: GlobalTensor, data_copy_params: DataCopyExtParams,
                  pad_params: DataCopyPadExtParams) -> None:
    ...


@overload
def data_copy_pad(dst: GlobalTensor, src: LocalTensor, data_copy_params: DataCopyExtParams) -> None:
    ...


@overload
def data_copy_pad(dst: LocalTensor, src: LocalTensor, data_copy_params: DataCopyExtParams,
                  nd2nz_params: Nd2NzParams) -> None:
    ...


@overload
def data_copy_pad(dst: LocalTensor, src: GlobalTensor, data_copy_params: DataCopyParams,
                  pad_params: DataCopyPadParams) -> None:
    ...


@overload
def data_copy_pad(dst: GlobalTensor, src: LocalTensor, data_copy_params: DataCopyParams) -> None:
    ...


@overload
def data_copy_pad(dst: LocalTensor, src: LocalTensor, data_copy_params: DataCopyParams,
                  nd2nz_params: Nd2NzParams) -> None:
    ...


@require_jit
@set_common_docstring(api_name="data_copy_pad")
def data_copy_pad(dst: BaseTensor, src: BaseTensor, *args, **kwargs) -> None:
    dispatcher = OverloadDispatcher(__name__)
    builder = global_builder.get_ir_builder()

    @dispatcher.register_auto
    def _(data_copy_params: DataCopyExtParams, pad_params: DataCopyPadExtParams):
        builder.create_asc_DataCopyPadExtL0Op(dst.to_ir(), src.to_ir(), data_copy_params.to_ir(), pad_params.to_ir())

    @dispatcher.register_auto
    def _(data_copy_params: DataCopyExtParams):
        builder.create_asc_DataCopyPadExtL2Op(dst.to_ir(), src.to_ir(), data_copy_params.to_ir())

    @dispatcher.register_auto
    def _(data_copy_params: DataCopyExtParams, nd2nz_params: Nd2NzParams):
        builder.create_asc_DataCopyPadExtNd2NzOp(dst.to_ir(), src.to_ir(), data_copy_params.to_ir(),
                                                 nd2nz_params.to_ir())

    @dispatcher.register_auto
    def _(data_copy_params: DataCopyParams, pad_params: DataCopyPadParams):
        builder.create_asc_DataCopyPadL0Op(dst.to_ir(), src.to_ir(), data_copy_params.to_ir(), pad_params.to_ir())

    @dispatcher.register_auto
    def _(data_copy_params: DataCopyParams):
        builder.create_asc_DataCopyPadL2Op(dst.to_ir(), src.to_ir(), data_copy_params.to_ir())

    @dispatcher.register_auto
    def _(data_copy_params: DataCopyParams, nd2nz_params: Nd2NzParams):
        builder.create_asc_DataCopyPadNd2NzOp(dst.to_ir(), src.to_ir(), data_copy_params.to_ir(), nd2nz_params.to_ir())

    dispatcher(*args, **kwargs)


@overload
def load_image_to_local(dst: LocalTensor, load_data_params: LoadImageToLocalParams) -> None:
    ...


@require_jit
@set_common_docstring(api_name="load_image_to_local")
def load_image_to_local(dst: LocalTensor, load_data_params: LoadImageToLocalParams) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_LoadImageToLocalOp(dst.to_ir(), load_data_params.to_ir())


@overload
def set_pad_value(padding_value: Union[int, float], pos: Optional[TPosition] = TPosition.MAX) -> None:
    ...


@require_jit
@set_common_docstring(api_name="set_pad_value")
def set_pad_value(padding_value: Union[int, float], pos: Optional[TPosition] = TPosition.MAX) -> None:
    if pos is not None and pos not in (
            TPosition.MAX,
            TPosition.VECIN,
            TPosition.VECOUT,
    ):
        raise ValueError("set_pad_value(): pos must be one of [TPosition.MAX, TPosition.VECIN, TPosition.VECOUT]")
    builder = global_builder.get_ir_builder()
    builder.create_asc_SetPadValueOp(_mat(padding_value).to_ir(), ir.TPosition.symbolize(pos))
