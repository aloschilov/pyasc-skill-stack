# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from typing import Optional, overload

from ..core.dtype import KnownTypes as KT
from ..core.enums import BatchMode, IterateMode, IterateOrder, ScheduleType, BatchOutMode
from ..core.ir_value import IRHandle, IRValue, materialize_ir_value as _mat
from ..core.utils import global_builder


class MatmulConfig(IRValue):

    @overload
    def __init__(self, do_norm: bool = True, do_basic_block: bool = False, do_multi_data_load: bool = False,
                 basic_m: int = 0, basic_n: int = 0, basic_k: int = 0, intrinsics_check: bool = False,
                 is_n_batch: bool = False, en_vec_nd2nz: bool = False, do_special_basic_block: bool = False,
                 do_mte2_preload: int = 0, single_core_m: int = 0, single_core_n: int = 0, single_core_k: int = 0,
                 step_m: int = 0, step_n: int = 0, base_mn: int = 0, single_core_mn: int = 0, en_unit_flag: bool = True,
                 is_per_tensor: bool = False, has_anti_quant_offset: bool = False, do_ib_share_norm: bool = False,
                 do_special_mdl: bool = False, enable_init: bool = True,
                 batch_mode: BatchMode = BatchMode.BATCH_LESS_THAN_L1.value, enable_end: bool = True,
                 enable_get_tensor_c: bool = True, enable_set_org_shape: bool = True, enable_set_bias: bool = True,
                 enable_set_tail: bool = True, enable_quant_vector: bool = True, enable_set_define_data: bool = True,
                 iterate_mode: IterateMode = IterateMode.ITERATE_MODE_DEFAULT.value, enable_reuse: bool = True,
                 enable_ub_reuse: bool = False, enable_l1_cache_ub: bool = False, intra_block_part_sum: bool = False,
                 iterate_order: IterateOrder = IterateOrder.UNDEF,
                 schedule_type: ScheduleType = ScheduleType.INNER_PRODUCT, enable_double_cache: int = 0,
                 is_bias_batch: bool = True, enable_static_pad_zeros: bool = False, is_partial_output: bool = False,
                 enable_mix_dual_master: bool = False, is_a2b2_shared: bool = False,
                 is_enable_channel_split: bool = False, enable_kdim_reorder_load: bool = False,
                 is_co1_shared: bool = False, shared_co1_buffer_size: int = 64 * 1024,
                 batch_out_mode: BatchOutMode = BatchOutMode.SINGLE_BATCH) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(
            self, do_norm: Optional[bool] = True, do_basic_block: Optional[bool] = False,
            do_multi_data_load: Optional[bool] = False, basic_m: Optional[int] = 0, basic_n: Optional[int] = 0,
            basic_k: Optional[int] = 0, intrinsics_check: Optional[bool] = False, is_n_batch: Optional[bool] = False,
            en_vec_nd2nz: Optional[bool] = False, do_special_basic_block: Optional[bool] = False,
            do_mte2_preload: Optional[int] = 0, single_core_m: Optional[int] = 0, single_core_n: Optional[int] = 0,
            single_core_k: Optional[int] = 0, step_m: Optional[int] = 0, step_n: Optional[int] = 0,
            base_mn: Optional[int] = 0, single_core_mn: Optional[int] = 0, en_unit_flag: Optional[bool] = True,
            is_per_tensor: Optional[bool] = False, has_anti_quant_offset: bool = False,
            do_ib_share_norm: Optional[bool] = False, do_special_mdl: Optional[bool] = False,
            enable_init: Optional[bool] = True, batch_mode: Optional[BatchMode] = BatchMode.BATCH_LESS_THAN_L1.value,
            enable_end: Optional[bool] = True, enable_get_tensor_c: Optional[bool] = True,
            enable_set_org_shape: Optional[bool] = True, enable_set_bias: Optional[bool] = True,
            enable_set_tail: Optional[bool] = True, enable_quant_vector: Optional[bool] = True,
            enable_set_define_data: Optional[bool] = True,
            iterate_mode: Optional[IterateMode] = IterateMode.ITERATE_MODE_DEFAULT.value,
            enable_reuse: Optional[bool] = True, enable_ub_reuse: Optional[bool] = False,
            enable_l1_cache_ub: Optional[bool] = False, intra_block_part_sum: Optional[bool] = False,
            iterate_order: Optional[IterateOrder] = IterateOrder.UNDEF,
            schedule_type: Optional[ScheduleType] = ScheduleType.INNER_PRODUCT, enable_double_cache: Optional[int] = 0,
            is_bias_batch: Optional[bool] = True, enable_static_pad_zeros: Optional[bool] = False,
            is_partial_output: Optional[bool] = False, enable_mix_dual_master: Optional[bool] = False,
            is_a2b2_shared: Optional[bool] = False, is_enable_channel_split: Optional[bool] = False,
            enable_kdim_reorder_load: Optional[bool] = False, is_co1_shared: Optional[bool] = False,
            shared_co1_buffer_size: Optional[int] = 64 * 1024,
            batch_out_mode: Optional[BatchOutMode] = BatchOutMode.SINGLE_BATCH, handle: Optional[IRHandle] = None):
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_MatmulConfigType(), [
                _mat(do_norm).to_ir(),
                _mat(do_basic_block).to_ir(),
                _mat(do_multi_data_load).to_ir(),
                _mat(basic_m).to_ir(),
                _mat(basic_n).to_ir(),
                _mat(basic_k).to_ir(),
                _mat(intrinsics_check).to_ir(),
                _mat(is_n_batch).to_ir(),
                _mat(en_vec_nd2nz).to_ir(),
                _mat(do_special_basic_block).to_ir(),
                _mat(do_mte2_preload).to_ir(),
                _mat(single_core_m).to_ir(),
                _mat(single_core_n).to_ir(),
                _mat(single_core_k).to_ir(),
                _mat(step_m).to_ir(),
                _mat(step_n).to_ir(),
                _mat(base_mn).to_ir(),
                _mat(single_core_mn).to_ir(),
                _mat(en_unit_flag).to_ir(),
                _mat(is_per_tensor).to_ir(),
                _mat(has_anti_quant_offset).to_ir(),
                _mat(do_ib_share_norm).to_ir(),
                _mat(do_special_mdl).to_ir(),
                _mat(enable_init).to_ir(),
                _mat(batch_mode).to_ir(),
                _mat(enable_end).to_ir(),
                _mat(enable_get_tensor_c).to_ir(),
                _mat(enable_set_org_shape).to_ir(),
                _mat(enable_set_bias).to_ir(),
                _mat(enable_set_tail).to_ir(),
                _mat(enable_quant_vector).to_ir(),
                _mat(enable_set_define_data).to_ir(),
                _mat(iterate_mode).to_ir(),
                _mat(enable_reuse).to_ir(),
                _mat(enable_ub_reuse).to_ir(),
                _mat(enable_l1_cache_ub).to_ir(),
                _mat(intra_block_part_sum).to_ir(),
                _mat(iterate_order).to_ir(),
                _mat(schedule_type).to_ir(),
                _mat(enable_double_cache).to_ir(),
                _mat(is_bias_batch).to_ir(),
                _mat(enable_static_pad_zeros).to_ir(),
                _mat(is_partial_output).to_ir(),
                _mat(enable_mix_dual_master).to_ir(),
                _mat(is_a2b2_shared).to_ir(),
                _mat(is_enable_channel_split).to_ir(),
                _mat(enable_kdim_reorder_load).to_ir(),
                _mat(is_co1_shared).to_ir(),
                _mat(shared_co1_buffer_size).to_ir(),
                _mat(batch_out_mode).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_asc_BatchModeType(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_ui8_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_asc_IterateOrderType(),
                builder.get_asc_ScheduleTypeType(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_ui32_type(),
                builder.get_asc_BatchOutModeType()
            ]), isConstexpr=True, isStatic=True)
        self.do_norm = do_norm
        self.do_basic_block = do_basic_block
        self.do_multi_data_load = do_multi_data_load
        self.basic_m = basic_m
        self.basic_n = basic_n
        self.basic_k = basic_k
        self.intrinsics_check = intrinsics_check
        self.is_n_batch = is_n_batch
        self.en_vec_nd2nz = en_vec_nd2nz
        self.do_special_basic_block = do_special_basic_block
        self.do_mte2_preload = do_mte2_preload
        self.single_core_m = single_core_m
        self.single_core_n = single_core_n
        self.single_core_k = single_core_k
        self.step_m = step_m
        self.step_n = step_n
        self.base_mn = base_mn
        self.single_core_mn = single_core_mn
        self.en_unit_flag = en_unit_flag
        self.is_per_tensor = is_per_tensor
        self.has_anti_quant_offset = has_anti_quant_offset
        self.do_ib_share_norm = do_ib_share_norm
        self.do_special_mdl = do_special_mdl
        self.enable_init = enable_init
        self.batch_mode = batch_mode
        self.enable_end = enable_end
        self.enable_get_tensor_c = enable_get_tensor_c
        self.enable_set_org_shape = enable_set_org_shape
        self.enable_set_bias = enable_set_bias
        self.enable_set_tail = enable_set_tail
        self.enable_quant_vector = enable_quant_vector
        self.enable_set_define_data = enable_set_define_data
        self.iterate_mode = iterate_mode
        self.enable_reuse = enable_reuse
        self.enable_ub_reuse = enable_ub_reuse
        self.enable_l1_cache_ub = enable_l1_cache_ub
        self.intra_block_part_sum = intra_block_part_sum
        self.iterate_order = iterate_order
        self.schedule_type = schedule_type
        self.enable_double_cache = enable_double_cache
        self.is_bias_batch = is_bias_batch
        self.enable_static_pad_zeros = enable_static_pad_zeros
        self.is_partial_output = is_partial_output
        self.enable_mix_dual_master = enable_mix_dual_master
        self.is_a2b2_shared = is_a2b2_shared
        self.is_enable_channel_split = is_enable_channel_split
        self.enable_kdim_reorder_load = enable_kdim_reorder_load
        self.is_co1_shared = is_co1_shared
        self.shared_co1_buffer_size = shared_co1_buffer_size
        self.batch_out_mode = batch_out_mode

    @classmethod
    def from_ir(cls, handle: IRHandle) -> MatmulConfig:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class QuantConfig(IRValue):

    @overload
    def __init__(self, calc_count: int = 0, offset_count: int = 0, scale_count: int = 0,
                 work_local_size: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(self, calc_count: int = 0, offset_count: int = 0, scale_count: int = 0, work_local_size: int = 0,
                 handle: Optional[IRHandle] = None):
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(builder.get_asc_AscendQuantConfigType(), [
            _mat(calc_count, KT.int32).to_ir(),
            _mat(offset_count, KT.int32).to_ir(),
            _mat(scale_count, KT.int32).to_ir(),
            _mat(work_local_size, KT.int32).to_ir(),
        ], builder.get_type_array_attr([builder.get_ui32_type()] * 4), isConstexpr=True, isStatic=True)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> QuantConfig:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class MatmulShapeParams:

    def __init__(self, single_core_m: int = 0, single_core_n: int = 0, single_core_k: int = 0, basic_m: int = 0,
                 basic_n: int = 0, basic_k: int = 0) -> None:
        self.single_core_m = single_core_m
        self.single_core_n = single_core_n
        self.single_core_k = single_core_k
        self.basic_m = basic_m
        self.basic_n = basic_n
        self.basic_k = basic_k


class MatmulQuantParams:

    def __init__(self, is_per_tensor: bool = False, has_anti_quant_offset: bool = False) -> None:
        self.is_per_tensor = is_per_tensor
        self.has_anti_quant_offset = has_anti_quant_offset


class MatmulBatchParams:

    def __init__(self, is_b_batch: bool = False, batch_mode: BatchMode = 1, is_bias_batch: bool = False) -> None:
        self.is_n_batch = is_b_batch
        self.batch_mode = batch_mode
        self.is_bias_batch = is_bias_batch


class MatmulFuncParams:

    def __init__(self, intrinsics_limit: bool = False, en_vec_nd2_nz: bool = False, enable_double_cache: bool = False,
                 enable_l1_cache: bool = False, do_mte2_pre_load: int = 0, iterate_order: IterateOrder = 0,
                 schedule_type: ScheduleType = 0, enable_reuse: bool = True, enable_ub_reuse: bool = False,
                 is_partial_output: bool = False, is_a2_b2_shared: bool = False, is_enable_channel_split: bool = False,
                 enable_kdim_reorder_load: bool = False) -> None:
        self.intrinsics_limit = intrinsics_limit
        self.en_vec_nd2_nz = en_vec_nd2_nz
        self.enable_double_cache = enable_double_cache
        self.enable_l1_cache = enable_l1_cache
        self.do_mte2_pre_load = do_mte2_pre_load
        self.iterate_order = iterate_order
        self.schedule_type = schedule_type
        self.enable_reuse = enable_reuse
        self.enable_ub_reuse = enable_ub_reuse
        self.is_partial_output = is_partial_output
        self.is_a2_b2_shared = is_a2_b2_shared
        self.is_enable_channel_split = is_enable_channel_split
        self.enable_kdim_reorder_load = enable_kdim_reorder_load
