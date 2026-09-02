# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ..core.dtype import KnownTypes as KT
from ..core.struct import Field, Struct
from ..core.utils import global_builder
from .types import MatmulConfig


class MatmulApiStaticTiling(Struct):
    used_core_num = Field(dtype=KT.int32, default=-1, name="usedCoreNum")
    m = Field(dtype=KT.int32, default=-1, name="M")
    n = Field(dtype=KT.int32, default=-1, name="N")
    k_a = Field(dtype=KT.int32, default=-1, name="Ka")
    k_b = Field(dtype=KT.int32, default=-1, name="Kb")
    single_core_m = Field(dtype=KT.int32, default=-1, name="singleCoreM")
    single_core_n = Field(dtype=KT.int32, default=-1, name="singleCoreN")
    single_core_k = Field(dtype=KT.int32, default=-1, name="singleCoreK")
    base_m = Field(dtype=KT.int32, default=-1, name="baseM")
    base_n = Field(dtype=KT.int32, default=-1, name="baseN")
    base_k = Field(dtype=KT.int32, default=-1, name="baseK")
    depth_a1 = Field(dtype=KT.int32, default=-1, name="depthA1")
    depth_b1 = Field(dtype=KT.int32, default=-1, name="depthB1")
    step_m = Field(dtype=KT.int32, default=-1, name="stepM")
    step_n = Field(dtype=KT.int32, default=-1, name="stepN")
    is_bias = Field(dtype=KT.int32, default=-1, name="isBias")
    trans_length = Field(dtype=KT.int32, default=-1, name="transLength")
    iterate_order = Field(dtype=KT.int32, default=-1, name="iterateOrder")
    share_mode = Field(dtype=KT.int32, default=-1, name="shareMode")
    share_l1_size = Field(dtype=KT.int32, default=-1, name="shareL1Size")
    share_l0c_size = Field(dtype=KT.int32, default=-1, name="shareL0CSize")
    share_ub_size = Field(dtype=KT.int32, default=-1, name="shareUbSize")
    step_k_a = Field(dtype=KT.int32, default=-1, name="stepKa")
    step_k_b = Field(dtype=KT.int32, default=-1, name="stepKb")
    depth_a_l1_cache_ub = Field(dtype=KT.int32, default=-1, name="depthAL1CacheUB")
    depth_b_l1_cache_ub = Field(dtype=KT.int32, default=-1, name="depthBL1CacheUB")
    db_l0a = Field(dtype=KT.int32, default=-1, name="dbL0A")
    db_l0b = Field(dtype=KT.int32, default=-1, name="dbL0B")
    db_l0c = Field(dtype=KT.int32, default=-1, name="dbL0C")
    a_layout_info_b = Field(dtype=KT.int32, default=-1, name="ALayoutInfoB")
    a_layout_info_s = Field(dtype=KT.int32, default=-1, name="ALayoutInfoS")
    a_layout_info_n = Field(dtype=KT.int32, default=-1, name="ALayoutInfoN")
    a_layout_info_g = Field(dtype=KT.int32, default=-1, name="ALayoutInfoG")
    a_layout_info_d = Field(dtype=KT.int32, default=-1, name="ALayoutInfoD")
    b_layout_info_b = Field(dtype=KT.int32, default=-1, name="BLayoutInfoB")
    b_layout_info_s = Field(dtype=KT.int32, default=-1, name="BLayoutInfoS")
    b_layout_info_n = Field(dtype=KT.int32, default=-1, name="BLayoutInfoN")
    b_layout_info_g = Field(dtype=KT.int32, default=-1, name="BLayoutInfoG")
    b_layout_info_d = Field(dtype=KT.int32, default=-1, name="BLayoutInfoD")
    c_layout_info_b = Field(dtype=KT.int32, default=-1, name="CLayoutInfoB")
    c_layout_info_s1 = Field(dtype=KT.int32, default=-1, name="CLayoutInfoS1")
    c_layout_info_n = Field(dtype=KT.int32, default=-1, name="CLayoutInfoN")
    c_layout_info_g = Field(dtype=KT.int32, default=-1, name="CLayoutInfoG")
    c_layout_info_s2 = Field(dtype=KT.int32, default=-1, name="CLayoutInfoS2")
    batch_num = Field(dtype=KT.int32, default=-1, name="BatchNum")
    mx_type_para = Field(dtype=KT.int32, default=-1, name="mxTypePara")
    cfg = MatmulConfig

    @classmethod
    def get_ir_type(cls):
        return global_builder.get_ir_builder().get_asc_MatmulApiStaticTilingType()


class RmsNormTiling(Struct):
    b_length = Field(dtype=KT.int32, default=0, name="bLength")
    s_length = Field(dtype=KT.int32, default=0, name="sLength")
    h_length = Field(dtype=KT.int32, default=0, name="hLength")
    original_h_length = Field(dtype=KT.int32, default=0, name="originalHLength")
    reciprocal_of_h_length = Field(dtype=KT.float32, default=0.0, name="reciprocalOfHLength")
    main_bsh_length = Field(dtype=KT.int32, default=0, name="mainBshLength")
    main_bs_length = Field(dtype=KT.int32, default=0, name="mainBsLength")
    main_bs_length_align = Field(dtype=KT.int32, default=0, name="mainBsLengthAlign")
    loop_round = Field(dtype=KT.int32, default=0, name="loopRound")
    input_tail_pos = Field(dtype=KT.int32, default=0, name="inputTailPos")
    tail_bsh_length = Field(dtype=KT.int32, default=0, name="tailBshLength")
    tail_bs_length = Field(dtype=KT.int32, default=0, name="tailBsLength")

    @classmethod
    def get_ir_type(cls):
        return global_builder.get_ir_builder().get_asc_RmsNormTilingType()


class SoftmaxTiling(Struct):
    src_m = Field(dtype=KT.int32, default=0, name="srcM")
    src_k = Field(dtype=KT.int32, default=0, name="srcK")
    src_size = Field(dtype=KT.int32, default=0, name="srcSize")
    out_max_m = Field(dtype=KT.int32, default=0, name="outMaxM")
    out_max_k = Field(dtype=KT.int32, default=0, name="outMaxK")
    out_max_size = Field(dtype=KT.int32, default=0, name="outMaxSize")
    split_m = Field(dtype=KT.int32, default=0, name="splitM")
    split_k = Field(dtype=KT.int32, default=0, name="splitK")
    split_size = Field(dtype=KT.int32, default=0, name="splitSize")
    reduce_m = Field(dtype=KT.int32, default=0, name="reduceM")
    reduce_k = Field(dtype=KT.int32, default=0, name="reduceK")
    reduce_size = Field(dtype=KT.int32, default=0, name="reduceSize")
    range_m = Field(dtype=KT.int32, default=0, name="rangeM")
    tail_m = Field(dtype=KT.int32, default=0, name="tailM")
    tail_split_size = Field(dtype=KT.int32, default=0, name="tailSplitSize")
    tail_reduce_size = Field(dtype=KT.int32, default=0, name="tailReduceSize")

    @classmethod
    def get_ir_type(cls):
        return global_builder.get_ir_builder().get_asc_SoftMaxTilingType()


class TCubeTiling(Struct):
    used_core_num = Field(dtype=KT.int32, default=0, name="usedCoreNum")
    m = Field(dtype=KT.int32, default=0, name="M")
    n = Field(dtype=KT.int32, default=0, name="N")
    k_a = Field(dtype=KT.int32, default=0, name="Ka")
    k_b = Field(dtype=KT.int32, default=0, name="Kb")
    single_core_m = Field(dtype=KT.int32, default=0, name="singleCoreM")
    single_core_n = Field(dtype=KT.int32, default=0, name="singleCoreN")
    single_core_k = Field(dtype=KT.int32, default=0, name="singleCoreK")
    base_m = Field(dtype=KT.int32, default=0, name="baseM")
    base_n = Field(dtype=KT.int32, default=0, name="baseN")
    base_k = Field(dtype=KT.int32, default=0, name="baseK")
    depth_a1 = Field(dtype=KT.int32, default=0, name="depthA1")
    depth_b1 = Field(dtype=KT.int32, default=0, name="depthB1")
    step_m = Field(dtype=KT.int32, default=0, name="stepM")
    step_n = Field(dtype=KT.int32, default=0, name="stepN")
    is_bias = Field(dtype=KT.int32, default=0, name="isBias")
    trans_length = Field(dtype=KT.int32, default=0, name="trans_length")
    iterate_order = Field(dtype=KT.int32, default=0, name="iterateOrder")
    share_mode = Field(dtype=KT.int32, default=0, name="shareMode")
    share_l1_size = Field(dtype=KT.int32, default=0, name="shareL1Size")
    share_l0c_size = Field(dtype=KT.int32, default=0, name="shareL0CSize")
    share_ub_size = Field(dtype=KT.int32, default=0, name="shareUbSize")
    batch_m = Field(dtype=KT.int32, default=0, name="batchM")
    batch_n = Field(dtype=KT.int32, default=0, name="batchN")
    single_batch_m = Field(dtype=KT.int32, default=0, name="singleBatchM")
    single_batch_n = Field(dtype=KT.int32, default=0, name="singleBatchN")
    step_k_a = Field(dtype=KT.int32, default=0, name="stepKa")
    step_k_b = Field(dtype=KT.int32, default=0, name="stepKb")
    depth_a_l1_cache_ub = Field(dtype=KT.int32, default=0, name="depthAL1CacheUB")
    depth_b_l1_cache_ub = Field(dtype=KT.int32, default=0, name="depthBL1CacheUB")
    db_l0a = Field(dtype=KT.int32, default=0, name="dbL0A")
    db_l0b = Field(dtype=KT.int32, default=0, name="dbL0B")
    db_l0c = Field(dtype=KT.int32, default=0, name="dbL0C")
    a_layout_info_b = Field(dtype=KT.int32, default=0, name="ALayoutInfoB")
    a_layout_info_s = Field(dtype=KT.int32, default=0, name="ALayoutInfoS")
    a_layout_info_n = Field(dtype=KT.int32, default=0, name="ALayoutInfoN")
    a_layout_info_g = Field(dtype=KT.int32, default=0, name="ALayoutInfoG")
    a_layout_info_d = Field(dtype=KT.int32, default=0, name="ALayoutInfoD")
    b_layout_info_b = Field(dtype=KT.int32, default=0, name="BLayoutInfoB")
    b_layout_info_s = Field(dtype=KT.int32, default=0, name="BLayoutInfoS")
    b_layout_info_n = Field(dtype=KT.int32, default=0, name="BLayoutInfoN")
    b_layout_info_g = Field(dtype=KT.int32, default=0, name="BLayoutInfoG")
    b_layout_info_d = Field(dtype=KT.int32, default=0, name="BLayoutInfoD")
    c_layout_info_b = Field(dtype=KT.int32, default=0, name="CLayoutInfoB")
    c_layout_info_s1 = Field(dtype=KT.int32, default=0, name="CLayoutInfoS1")
    c_layout_info_n = Field(dtype=KT.int32, default=0, name="CLayoutInfoN")
    c_layout_info_g = Field(dtype=KT.int32, default=0, name="CLayoutInfoG")
    c_layout_info_s2 = Field(dtype=KT.int32, default=0, name="CLayoutInfoS2")
    batch_num = Field(dtype=KT.int32, default=0, name="BatchNum")

    @classmethod
    def get_ir_type(cls):
        return global_builder.get_ir_builder().get_asc_TCubeTilingType()
