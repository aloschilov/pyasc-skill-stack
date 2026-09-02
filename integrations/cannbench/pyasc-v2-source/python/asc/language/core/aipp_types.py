# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from .ir_value import IRValue, IRHandle, RuntimeInt, materialize_ir_value as _mat
from .dtype import DataType, KnownTypes as KT, float16, int8
from .utils import global_builder


def _get_mlir_type_from_dtype(dtype: DataType, builder):
    if dtype == int8:
        return builder.get_i8_type()
    if dtype == float16:
        return builder.get_f16_type()
    raise TypeError(f"Unsupported DataType '{dtype}' for AIPP parameter construction.")


@dataclass
class AippPaddingParams:
    padding_mode: RuntimeInt = 0
    padding_value_ch0: int | float = 0
    padding_value_ch1: int | float = 0
    padding_value_ch2: int | float = 0
    padding_value_ch3: int | float = 0

    def _to_ir(self, dtype: DataType) -> IRHandle:
        builder = global_builder.get_ir_builder()
        ir_dtype = _get_mlir_type_from_dtype(dtype, builder)
        return builder.create_asc_ConstructOp(
            builder.get_asc_AippPaddingParamsType(), [
                _mat(self.padding_mode).to_ir(),
                _mat(self.padding_value_ch0, dtype).to_ir(),
                _mat(self.padding_value_ch1, dtype).to_ir(),
                _mat(self.padding_value_ch2, dtype).to_ir(),
                _mat(self.padding_value_ch3, dtype).to_ir()
            ], builder.get_type_array_attr([builder.get_ui32_type(), ir_dtype, ir_dtype, ir_dtype, ir_dtype]))


@dataclass
class AippSwapParams:
    is_swap_rb: bool = False
    is_swap_uv: bool = False
    is_swap_ax: bool = False

    def _to_ir(self) -> IRHandle:
        builder = global_builder.get_ir_builder()
        return builder.create_asc_ConstructOp(
            builder.get_asc_AippSwapParamsType(),
            [_mat(self.is_swap_rb).to_ir(),
             _mat(self.is_swap_uv).to_ir(),
             _mat(self.is_swap_ax).to_ir()],
            builder.get_type_array_attr([builder.get_i8_type(),
                                         builder.get_i8_type(),
                                         builder.get_i8_type()]))


@dataclass
class AippSingleLineParams:
    is_single_line_copy: bool = False

    def _to_ir(self) -> IRHandle:
        builder = global_builder.get_ir_builder()
        return builder.create_asc_ConstructOp(builder.get_asc_AippSingleLineParamsType(),
                                              [_mat(self.is_single_line_copy).to_ir()],
                                              builder.get_type_array_attr([builder.get_i8_type()]))


@dataclass
class AippDataTypeConvParams:
    dtc_mean_ch0: int = 0
    dtc_mean_ch1: int = 0
    dtc_mean_ch2: int = 0
    dtc_min_ch0: float = 0.0
    dtc_min_ch1: float = 0.0
    dtc_min_ch2: float = 0.0
    dtc_var_ch0: float = 1.0
    dtc_var_ch1: float = 1.0
    dtc_var_ch2: float = 1.0
    dtc_round_mode: int = 0

    def _to_ir(self) -> IRHandle:
        builder = global_builder.get_ir_builder()
        return builder.create_asc_ConstructOp(
            builder.get_asc_AippDataTypeConvParamsType(), [
                _mat(self.dtc_mean_ch0, KT.uint8).to_ir(),
                _mat(self.dtc_mean_ch1, KT.uint8).to_ir(),
                _mat(self.dtc_mean_ch2, KT.uint8).to_ir(),
                _mat(self.dtc_min_ch0, KT.float16).to_ir(),
                _mat(self.dtc_min_ch1, KT.float16).to_ir(),
                _mat(self.dtc_min_ch2, KT.float16).to_ir(),
                _mat(self.dtc_var_ch0, KT.float16).to_ir(),
                _mat(self.dtc_var_ch1, KT.float16).to_ir(),
                _mat(self.dtc_var_ch2, KT.float16).to_ir(),
                _mat(self.dtc_round_mode, KT.uint32).to_ir()
            ],
            builder.get_type_array_attr([
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_f16_type(),
                builder.get_f16_type(),
                builder.get_f16_type(),
                builder.get_f16_type(),
                builder.get_f16_type(),
                builder.get_f16_type(),
                builder.get_ui32_type()
            ]))


@dataclass
class AippChannelPaddingParams:
    c_padding_mode: RuntimeInt = 0
    c_padding_value: int | float = 0

    def _to_ir(self, dtype: DataType) -> IRHandle:
        builder = global_builder.get_ir_builder()
        ir_dtype = _get_mlir_type_from_dtype(dtype, builder)
        return builder.create_asc_ConstructOp(
            builder.get_asc_AippChannelPaddingParamsType(),
            [_mat(self.c_padding_mode).to_ir(),
             _mat(self.c_padding_value, dtype).to_ir()],
            builder.get_type_array_attr([builder.get_ui32_type(), ir_dtype]))


@dataclass
class AippColorSpaceConvParams:
    is_enable_csc: bool = False
    csc_matrix_r0_c0: int = 0
    csc_matrix_r0_c1: int = 0
    csc_matrix_r0_c2: int = 0
    csc_matrix_r1_c0: int = 0
    csc_matrix_r1_c1: int = 0
    csc_matrix_r1_c2: int = 0
    csc_matrix_r2_c0: int = 0
    csc_matrix_r2_c1: int = 0
    csc_matrix_r2_c2: int = 0
    csc_bias_in_0: int = 0
    csc_bias_in_1: int = 0
    csc_bias_in_2: int = 0
    csc_bias_out_0: int = 0
    csc_bias_out_1: int = 0
    csc_bias_out_2: int = 0

    def _to_ir(self) -> IRHandle:
        builder = global_builder.get_ir_builder()
        params_ir = [
            _mat(self.is_enable_csc).to_ir(),
            _mat(self.csc_matrix_r0_c0, KT.int16).to_ir(),
            _mat(self.csc_matrix_r0_c1, KT.int16).to_ir(),
            _mat(self.csc_matrix_r0_c2, KT.int16).to_ir(),
            _mat(self.csc_matrix_r1_c0, KT.int16).to_ir(),
            _mat(self.csc_matrix_r1_c1, KT.int16).to_ir(),
            _mat(self.csc_matrix_r1_c2, KT.int16).to_ir(),
            _mat(self.csc_matrix_r2_c0, KT.int16).to_ir(),
            _mat(self.csc_matrix_r2_c1, KT.int16).to_ir(),
            _mat(self.csc_matrix_r2_c2, KT.int16).to_ir(),
            _mat(self.csc_bias_in_0, KT.uint8).to_ir(),
            _mat(self.csc_bias_in_1, KT.uint8).to_ir(),
            _mat(self.csc_bias_in_2, KT.uint8).to_ir(),
            _mat(self.csc_bias_out_0, KT.uint8).to_ir(),
            _mat(self.csc_bias_out_1, KT.uint8).to_ir(),
            _mat(self.csc_bias_out_2, KT.uint8).to_ir()
        ]
        types_attr = [
            builder.get_i8_type(),
            builder.get_i16_type(),
            builder.get_i16_type(),
            builder.get_i16_type(),
            builder.get_i16_type(),
            builder.get_i16_type(),
            builder.get_i16_type(),
            builder.get_i16_type(),
            builder.get_i16_type(),
            builder.get_i16_type(),
            builder.get_ui8_type(),
            builder.get_ui8_type(),
            builder.get_ui8_type(),
            builder.get_ui8_type(),
            builder.get_ui8_type(),
            builder.get_ui8_type()
        ]
        return builder.create_asc_ConstructOp(builder.get_asc_AippColorSpaceConvParamsType(), params_ir,
                                              builder.get_type_array_attr(types_attr))


class AippParams(IRValue):

    def __init__(self, dtype: DataType, padding_params: Optional[AippPaddingParams] = None,
                 swap_params: Optional[AippSwapParams] = None,
                 single_line_params: Optional[AippSingleLineParams] = None,
                 dtc_params: Optional[AippDataTypeConvParams] = None,
                 c_padding_params: Optional[AippChannelPaddingParams] = None,
                 csc_params: Optional[AippColorSpaceConvParams] = None, handle: Optional[IRHandle] = None) -> None:
        if handle:
            self.handle = handle
            return

        if dtype not in (int8, float16):
            raise TypeError(f"AippParams dtype must be asc.int8 or asc.float16, but got {dtype}")

        if padding_params is None:
            padding_params = AippPaddingParams()
        if swap_params is None:
            swap_params = AippSwapParams()
        if single_line_params is None:
            single_line_params = AippSingleLineParams()
        if dtc_params is None:
            dtc_params = AippDataTypeConvParams()
        if c_padding_params is None:
            c_padding_params = AippChannelPaddingParams()
        if csc_params is None:
            csc_params = AippColorSpaceConvParams()

        builder = global_builder.get_ir_builder()

        padding_params_ir = padding_params._to_ir(dtype)
        swap_params_ir = swap_params._to_ir()
        single_line_params_ir = single_line_params._to_ir()
        dtc_params_ir = dtc_params._to_ir()
        c_padding_params_ir = c_padding_params._to_ir(dtype)
        csc_params_ir = csc_params._to_ir()

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_AippParamsType(), [
                padding_params_ir, swap_params_ir, single_line_params_ir, dtc_params_ir, c_padding_params_ir,
                csc_params_ir
            ],
            builder.get_type_array_attr([
                builder.get_asc_AippPaddingParamsType(),
                builder.get_asc_AippSwapParamsType(),
                builder.get_asc_AippSingleLineParamsType(),
                builder.get_asc_AippDataTypeConvParamsType(),
                builder.get_asc_AippChannelPaddingParamsType(),
                builder.get_asc_AippColorSpaceConvParamsType()
            ]))

    @classmethod
    def from_ir(cls, handle: IRHandle) -> AippParams:
        return cls(dtype=int8, handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle
