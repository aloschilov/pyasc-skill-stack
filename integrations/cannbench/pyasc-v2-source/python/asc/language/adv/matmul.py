# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union, overload

from ..._C import ir
from ..core.dtype import DataType, KnownTypes as KT
from ..core.enums import CubeFormat, TPosition, LayoutMode, BatchMode, IterateOrder, ScheduleType, MatmulConfigMode, \
    MatmulPolicy
from ..core.ir_value import GlobalAddress, IRHandle, IRValue, PlainValue, \
                        RuntimeBool, RuntimeInt, materialize_ir_value as _mat
from ..core.properties import property, TOTAL_L1_SIZE
from ..core.tensor import BaseTensor, GlobalTensor, LocalTensor
from ..core.utils import require_jit, global_builder, OverloadDispatcher, DefaultValued
from ..fwk.tpipe import TPipe
from .tiling import MatmulApiStaticTiling, TCubeTiling
from .types import MatmulConfig, MatmulShapeParams, MatmulQuantParams, MatmulBatchParams, MatmulFuncParams
from .utils import set_matmul_docstring


def check_type(target_type: Any, type_union: List[Any], msg: str):
    if target_type not in type_union:
        raise ValueError(msg)


@require_jit
@set_matmul_docstring(api_name="register_matmul")
def register_matmul(pipe: TPipe, workspace: GlobalAddress, matmul: Matmul, \
                    tiling: Optional[TCubeTiling] = None) -> None:
    ir_tiling = tiling.to_ir() if tiling is not None else None
    builder = global_builder.get_ir_builder()
    builder.create_asc_RegistMatmulObjOp(pipe.to_ir(), workspace.to_ir(), matmul.to_ir(), ir_tiling)


@dataclass(frozen=True)
class MatmulType:
    position: TPosition
    format: CubeFormat
    dtype: DataType
    is_trans: bool = False
    layout: LayoutMode = LayoutMode.NONE


class Matmul(IRValue):
    """
    Ascend C提供一组Matmul高阶API，方便用户快速实现Matmul矩阵乘法的运算操作。
    Matmul的计算公式为：C = A * B + Bias。
    """

    @overload
    def __init__(
        self,
        a: MatmulType,
        b: MatmulType,
        c: MatmulType,
        bias: Optional[MatmulType] = None,
        matmul_config: Optional[MatmulConfig] = None,
        matmul_policy: Optional[MatmulPolicy] = 0,
    ) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(self, a: Optional[MatmulType] = None, b: Optional[MatmulType] = None, c: Optional[MatmulType] = None,
                 bias: Optional[MatmulType] = None, matmul_config: Optional[MatmulConfig] = None,
                 matmul_policy: Optional[MatmulPolicy] = 0, handle: Optional[IRHandle] = None):
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        for name, value in [('a', a), ('b', b), ('c', c)]:
            if not isinstance(value, MatmulType):
                raise ValueError(f"{name} must be MatmulType and cannot be None.")
        for name, value, param_type in [('bias', bias, MatmulType), ('matmul_config', matmul_config, MatmulConfig)]:
            if value is not None and not isinstance(value, param_type):
                raise ValueError(f"{name} must be {param_type.__name__} or None.")
        bias_pos = c.position
        bias_format = c.format
        bias_type = c.dtype
        if bias is not None:
            bias_pos = bias.position
            bias_format = bias.format
            bias_type = bias.dtype
        if matmul_config is None:
            matmul_config = MatmulConfig()
        ir_type = builder.get_matmul_type(
            a.position, a.format, a.dtype.to_ir(), a.is_trans, a.layout,  # a
            b.position, b.format, b.dtype.to_ir(), b.is_trans, b.layout,  # b
            c.position, c.format, c.dtype.to_ir(), c.is_trans, c.layout,  # c
            bias_pos, bias_format, bias_type.to_ir(), matmul_config.do_norm, matmul_config.do_basic_block,
            matmul_config.do_multi_data_load, matmul_config.basic_m, matmul_config.basic_n, matmul_config.basic_k,
            matmul_config.intrinsics_check, matmul_config.is_n_batch, matmul_config.en_vec_nd2nz,
            matmul_config.do_special_basic_block, matmul_config.do_mte2_preload, matmul_config.single_core_m,
            matmul_config.single_core_n, matmul_config.single_core_k, matmul_config.step_m, matmul_config.step_n,
            matmul_config.base_mn, matmul_config.single_core_mn, matmul_config.en_unit_flag,
            matmul_config.is_per_tensor, matmul_config.has_anti_quant_offset, matmul_config.do_ib_share_norm,
            matmul_config.do_special_mdl, matmul_config.enable_init, matmul_config.batch_mode, matmul_config.enable_end,
            matmul_config.enable_get_tensor_c, matmul_config.enable_set_org_shape, matmul_config.enable_set_bias,
            matmul_config.enable_set_tail, matmul_config.enable_quant_vector, matmul_config.enable_set_define_data,
            matmul_config.iterate_mode, matmul_config.enable_reuse, matmul_config.enable_ub_reuse,
            matmul_config.enable_l1_cache_ub, matmul_config.intra_block_part_sum, matmul_config.iterate_order,
            matmul_config.schedule_type, matmul_config.enable_double_cache, matmul_config.is_bias_batch,
            matmul_config.enable_static_pad_zeros, matmul_config.is_partial_output,
            matmul_config.enable_mix_dual_master, matmul_config.is_a2b2_shared, matmul_config.is_enable_channel_split,
            matmul_config.enable_kdim_reorder_load, matmul_config.is_co1_shared, matmul_config.shared_co1_buffer_size,
            matmul_config.batch_out_mode)
        self.handle = builder.create_asc_ConstructOp(ir_type, [])
        self.c_dtype = c.dtype
        self.a_dtype = a.dtype
        self.b_dtype = b.dtype

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Matmul:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    @require_jit
    @set_matmul_docstring(api_name="end")
    def end(self) -> None:
        global_builder.get_ir_builder().create_asc_MatmulEndOp(self.to_ir())

    @overload
    def get_tensor_c(self, tensor: BaseTensor, en_atomic: int = 0, en_sequential_write: bool = False, sync: bool = True,
                     optional_tensor: Optional[BaseTensor] = None) -> None:
        ...

    @overload
    def get_tensor_c(self, en_atomic: int = 0, en_sequential_write: bool = False, sync: bool = True) -> GlobalTensor:
        ...

    @require_jit
    @set_matmul_docstring(api_name="get_tensor_c")
    def get_tensor_c(self, *args, **kwargs) -> None:
        dispatcher = OverloadDispatcher(__name__)
        builder = global_builder.get_ir_builder()

        @dispatcher.register(tensor=BaseTensor, en_atomic=DefaultValued(RuntimeInt, 0),
                             en_sequential_write=DefaultValued(RuntimeBool,
                                                               False), sync=DefaultValued(RuntimeBool, True),
                             optional_tensor=DefaultValued(Optional[BaseTensor], None))
        def _(tensor: BaseTensor, en_atomic: RuntimeInt = 0, en_sequential_write: RuntimeBool = False,
              sync: RuntimeBool = True, optional_tensor: Optional[BaseTensor] = None):
            check_type(tensor.dtype, [KT.int32, KT.int_, KT.float_, KT.float32, KT.half, KT.float16, KT.int8],
                       "Tensor type is not supported in get_tensor_c")
            check_type(en_atomic, [0, 1, 2, 3], "Params en_atomic should be in [0, 1, 2, 3]")
            en_atomic = _mat(en_atomic, KT.int8)
            en_sequential_write = _mat(en_sequential_write, KT.bit)
            if optional_tensor is not None:
                if not isinstance(tensor, GlobalTensor) or not isinstance(optional_tensor, LocalTensor):
                    raise TypeError("When use get_tensor_c to fetch output to both GM and VECIN,\
                        the first input should be GlobalTensor, and the last input should be LocalTensor")
                check_type(optional_tensor.dtype,
                           [KT.int32, KT.int_, KT.float_, KT.float32, KT.half, KT.float16, KT.int8],
                           "Optional tensor type is not supported in get_tensor_c")
            tensor2 = optional_tensor.to_ir() if optional_tensor is not None else None
            global_builder.get_ir_builder().create_asc_MatmulGetTensorCOp(self.to_ir(), tensor.to_ir(), tensor2,
                                                                          en_atomic.to_ir(),
                                                                          en_sequential_write.to_ir(),
                                                                          _mat(sync, KT.bit).to_ir())

        @dispatcher.register(en_atomic=DefaultValued(RuntimeInt,
                                                     0), en_sequential_write=DefaultValued(RuntimeBool, False),
                             sync=DefaultValued(RuntimeBool, True))
        def _(en_atomic: int = 0, en_sequential_write: bool = False, sync: bool = True):
            if sync:
                raise ValueError("Matmul get_tensor_c with return value only support params sync=False")
            check_type(en_atomic, [0, 1, 2, 3], "Params en_atomic should be in [0, 1, 2, 3]")
            en_atomic = _mat(en_atomic, KT.int8)
            en_sequential_write = _mat(en_sequential_write, KT.bit)
            res = builder.create_asc_MatmulGetTensorCReturnOp(ir.get_global_tensor_type(self.c_dtype.to_ir()),
                                                              self.to_ir(), en_atomic.to_ir(),
                                                              en_sequential_write.to_ir(),
                                                              _mat(sync, KT.bit).to_ir())
            return GlobalTensor(res)

        dispatcher(*args, **kwargs)

    @require_jit
    @set_matmul_docstring(api_name="init")
    def init(self, tiling: TCubeTiling) -> None:
        global_builder.get_ir_builder().create_asc_MatmulInitOp(self.to_ir(), tiling.to_ir())

    @overload
    def iterate(self, en_partial_sum: bool = False, sync: bool = True,
                local_c_matrix: Optional[BaseTensor] = None) -> MatmulIterator:
        ...

    @require_jit
    @set_matmul_docstring(api_name="iterate")
    def iterate(self, en_partial_sum: RuntimeBool = False, sync: RuntimeBool = True,
                local_c_matrix: Optional[BaseTensor] = None) -> MatmulIterator:
        if local_c_matrix is not None:
            check_type(local_c_matrix.dtype, [KT.int32, KT.int_, KT.float_, KT.float32],
                       "Local_c_matrix Tensor type is not supported in iterate")
        return MatmulIterator(self, en_partial_sum, local_c_matrix, sync)

    @overload
    def iterate_all(self, tensor: BaseTensor, en_atomic: int = 0, sync: bool = True,
                    en_sequential_write: Optional[bool] = None, wait_iterate_all: Optional[bool] = None,
                    fake_msg: Optional[bool] = None) -> None:
        ...

    @require_jit
    @set_matmul_docstring(api_name="iterate_all")
    def iterate_all(self, tensor: BaseTensor, en_atomic: RuntimeInt = 0, sync: RuntimeBool = True,
                    en_sequential_write: Optional[RuntimeBool] = None, wait_iterate_all: Optional[RuntimeBool] = None,
                    fake_msg: Optional[RuntimeBool] = None) -> None:
        if isinstance(tensor, GlobalTensor):
            if en_sequential_write is None:
                en_sequential_write = False
            if wait_iterate_all is None:
                wait_iterate_all = False
            if fake_msg is None:
                fake_msg = False
        else:
            if en_sequential_write is not None:
                raise ValueError("When iterate_all output to TSCM, param en_sequential_write is not supported.")
            if wait_iterate_all is not None:
                raise ValueError("When iterate_all output to TSCM, param wait_iterate_all is not supported.")
            if fake_msg is not None:
                raise ValueError("When iterate_all output to TSCM, param fake_msg is not supported.")
        check_type(tensor.dtype, [KT.int32, KT.int_, KT.float_, KT.float32, KT.half, KT.float16, KT.int8],
                   "Tensor type is not supported in iterate_all")
        check_type(en_atomic, [0, 1, 2, 3], "Params en_atomic should be in [0, 1, 2, 3]")
        if wait_iterate_all is True and sync:
            raise ValueError("Param wait_iterate_all can be True only when sync is False")
        en_sequential_write_value = _mat(en_sequential_write, KT.bit).to_ir() \
            if en_sequential_write is not None else None
        wait_iterate_all_value = _mat(wait_iterate_all, KT.bit).to_ir() if wait_iterate_all is not None else None
        fake_msg_value = _mat(fake_msg, KT.bit).to_ir() if fake_msg is not None else None
        sync_value = _mat(sync, KT.bit).to_ir()
        global_builder.get_ir_builder().create_asc_MatmulIterateAllOp(self.to_ir(), tensor.to_ir(),
                                                                      _mat(en_atomic, KT.int8).to_ir(),
                                                                      en_sequential_write_value, wait_iterate_all_value,
                                                                      fake_msg_value, sync_value)

    @require_jit
    @set_matmul_docstring(api_name="wait_iterate_all")
    def wait_iterate_all(self) -> None:
        global_builder.get_ir_builder().create_asc_MatmulWaitIterateAllOp(self.to_ir())

    @overload
    def iterate_batch(self, tensor: BaseTensor, batch_a: int, batch_b: int, en_sequential_write: bool,
                      matrix_stride_a: int = 0, matrix_stride_b: int = 0, matrix_stride_c: int = 0,
                      en_partial_sum: bool = False, en_atomic: int = 0, sync: bool = True,
                      wait_iterate_batch: Optional[bool] = None) -> None:
        ...

    @overload
    def iterate_batch(self, tensor: BaseTensor, en_partial_sum, en_atomic, en_sequential_write: bool,
                      matrix_stride_a: int = 0, matrix_stride_b: int = 0, matrix_stride_c: int = 0,
                      sync: bool = True) -> None:
        ...

    @require_jit
    @set_matmul_docstring(api_name="iterate_batch")
    def iterate_batch(self, *args, **kwargs) -> None:
        dispatcher = OverloadDispatcher(__name__)
        builder = global_builder.get_ir_builder()

        @dispatcher.register(tensor=BaseTensor, batch_a=RuntimeInt, batch_b=RuntimeInt, en_sequential_write=RuntimeBool,
                             matrix_stride_a=DefaultValued(RuntimeInt, 0), matrix_stride_b=DefaultValued(RuntimeInt, 0),
                             matrix_stride_c=DefaultValued(RuntimeInt, 0),
                             en_partial_sum=DefaultValued(Optional[RuntimeBool],
                                                          None), en_atomic=DefaultValued(Optional[RuntimeInt], None),
                             sync=DefaultValued(RuntimeBool,
                                                True), wait_iterate_batch=DefaultValued(Optional[RuntimeBool], None))
        def _(tensor: BaseTensor, batch_a: RuntimeInt, batch_b: RuntimeInt, en_sequential_write: RuntimeBool,
              matrix_stride_a: RuntimeInt = 0, matrix_stride_b: RuntimeInt = 0, matrix_stride_c: RuntimeInt = 0,
              en_partial_sum: Optional[RuntimeBool] = None, en_atomic: Optional[RuntimeInt] = None,
              sync: RuntimeBool = True, wait_iterate_batch: Optional[RuntimeBool] = None):
            check_type(tensor.dtype, [KT.int32, KT.int_, KT.float_, KT.float32, KT.half, KT.float16],
                       "Tensor type is not supported in iterate_batch")
            if isinstance(tensor, GlobalTensor):
                check_type(en_sequential_write, [False],
                           "When output to GM, en_sequential_write should be False in iterate_batch")
            else:
                check_type(en_sequential_write, [True],
                           "When output to GM, en_sequential_write should be True in iterate_batch")
            check_type(en_atomic, [None, 0, 1, 2, 3], "Params en_atomic should be in [None, 0, 1, 2, 3]")
            batch_a_value = _mat(batch_a, KT.uint32).to_ir()
            batch_b_value = _mat(batch_b, KT.uint32).to_ir()
            en_sequential_write_value = _mat(en_sequential_write, KT.bit).to_ir()
            matrix_stride_a_value = _mat(matrix_stride_a, KT.uint32).to_ir()
            matrix_stride_b_value = _mat(matrix_stride_b, KT.uint32).to_ir()
            matrix_stride_c_value = _mat(matrix_stride_c, KT.uint32).to_ir()
            en_partial_sum_value = None
            if en_partial_sum is not None:
                en_partial_sum_value = _mat(en_partial_sum, KT.bit).to_ir()
            en_atomic_value = None
            if en_atomic is not None:
                en_atomic_value = _mat(en_atomic, KT.uint8).to_ir()
            sync_value = _mat(sync, KT.bit).to_ir()
            wait_iterate_batch_value = None
            if isinstance(tensor, GlobalTensor):
                if wait_iterate_batch is True and sync:
                    raise ValueError("Param wait_iterate_batch can be True only when sync is False")
                if wait_iterate_batch is None:
                    wait_iterate_batch_value = _mat(False, KT.bit).to_ir()
                else:
                    wait_iterate_batch_value = _mat(wait_iterate_batch, KT.bit).to_ir()
            builder.create_asc_MatmulIterateBatchOp(self.to_ir(), tensor.to_ir(), batch_a_value, batch_b_value,
                                                    en_sequential_write_value, matrix_stride_a_value,
                                                    matrix_stride_b_value, matrix_stride_c_value, en_partial_sum_value,
                                                    en_atomic_value, sync_value, wait_iterate_batch_value)

        @dispatcher.register(tensor=BaseTensor, en_partial_sum=RuntimeBool, en_atomic=RuntimeInt,
                             en_sequential_write=RuntimeBool, matrix_stride_a=DefaultValued(RuntimeInt, 0),
                             matrix_stride_b=DefaultValued(RuntimeInt, 0), matrix_stride_c=DefaultValued(RuntimeInt, 0))
        def _(tensor: BaseTensor, en_partial_sum: RuntimeBool, en_atomic: RuntimeInt, en_sequential_write: RuntimeBool,
              matrix_stride_a: RuntimeInt = 0, matrix_stride_b: RuntimeInt = 0, matrix_stride_c: RuntimeInt = 0):
            if not isinstance(tensor, GlobalTensor):
                raise TypeError("iterate_batch interface under cube-only sence only support output to GM.")
            check_type(tensor.dtype, [KT.int32, KT.int_, KT.float_, KT.float32, KT.half, KT.float16],
                       "Tensor type is not supported in iterate_batch")
            check_type(en_atomic, [0, 1, 2, 3], "Params en_atomic should be in [0, 1, 2, 3]")
            check_type(en_sequential_write, [False],
                       "When output to GM, en_sequential_write should be False in iterate_batch")
            en_partial_sum_value = _mat(en_partial_sum, KT.bit).to_ir()
            en_atomic_value = _mat(en_atomic, KT.uint8).to_ir()
            en_sequential_write_value = _mat(en_sequential_write, KT.bit).to_ir()
            matrix_stride_a_value = _mat(matrix_stride_a, KT.uint32).to_ir()
            matrix_stride_b_value = _mat(matrix_stride_b, KT.uint32).to_ir()
            matrix_stride_c_value = _mat(matrix_stride_c, KT.uint32).to_ir()
            builder.create_asc_MatmulIterateBatchCubeOnlyOp(self.to_ir(), en_partial_sum_value, en_atomic_value,
                                                            en_sequential_write_value, matrix_stride_a_value,
                                                            matrix_stride_b_value, matrix_stride_c_value)

        dispatcher(*args, **kwargs)

    @require_jit
    @set_matmul_docstring(api_name="wait_iterate_batch")
    def wait_iterate_batch(self) -> None:
        global_builder.get_ir_builder().create_asc_MatmulWaitIterateBatchOp(self.to_ir())

    @overload
    def get_batch_tensor_c(self, batch_a: int, batch_b: int, en_sequential_write: bool = False,
                           sync: bool = True) -> GlobalTensor:
        ...

    @overload
    def get_batch_tensor_c(self, tensor: LocalTensor, batch_a: int, batch_b: int, en_sequential_write: bool = False,
                           sync: bool = True) -> None:
        ...

    @require_jit
    @set_matmul_docstring(api_name="get_batch_tensor_c")
    def get_batch_tensor_c(self, *args, **kwargs):
        dispatcher = OverloadDispatcher(__name__)
        builder = global_builder.get_ir_builder()

        @dispatcher.register(batch_a=RuntimeInt, batch_b=RuntimeInt,
                             en_sequential_write=DefaultValued(RuntimeBool,
                                                               False), sync=DefaultValued(RuntimeBool, True))
        def _(batch_a: RuntimeInt, batch_b: RuntimeInt, en_sequential_write: RuntimeBool = False,
              sync: RuntimeBool = True):
            if sync:
                raise ValueError("Matmul get_batch_tensor_c only support params sync=False")
            batch_a_value = _mat(batch_a, KT.uint32).to_ir()
            batch_b_value = _mat(batch_b, KT.uint32).to_ir()
            en_sequential_write_value = _mat(en_sequential_write, KT.bit).to_ir()
            sync_value = _mat(sync, KT.bit).to_ir()
            res = builder.create_asc_MatmulGetBatchTensorCOp(ir.get_global_tensor_type(self.c_dtype.to_ir()),
                                                             self.to_ir(), batch_a_value, batch_b_value,
                                                             en_sequential_write_value, sync_value)
            return GlobalTensor(res)

        @dispatcher.register(tensor=LocalTensor, batch_a=RuntimeInt, batch_b=RuntimeInt,
                             en_sequential_write=DefaultValued(RuntimeBool,
                                                               False), sync=DefaultValued(RuntimeBool, True))
        def _(tensor: LocalTensor, batch_a: RuntimeInt, batch_b: RuntimeInt, en_sequential_write: RuntimeBool = False,
              sync: RuntimeBool = True):
            if sync:
                raise ValueError("Matmul get_batch_tensor_c only support params sync=False")
            batch_a_value = _mat(batch_a, KT.uint32).to_ir()
            batch_b_value = _mat(batch_b, KT.uint32).to_ir()
            en_sequential_write_value = _mat(en_sequential_write, KT.bit).to_ir()
            sync_value = _mat(sync, KT.bit).to_ir()
            builder.create_asc_MatmulGetBatchTensorCLocalMemOp(self.to_ir(), tensor.to_ir(), batch_a_value,
                                                               batch_b_value, en_sequential_write_value, sync_value)

        dispatcher(*args, **kwargs)

    @overload
    def set_tensor_a(self, scalar: int) -> None:
        ...

    @overload
    def set_tensor_a(self, tensor: BaseTensor, transpose: bool = False) -> None:
        ...

    @require_jit
    @set_matmul_docstring(api_name="set_tensor_a")
    def set_tensor_a(self, *args, **kwargs) -> None:
        dispatcher = OverloadDispatcher(__name__)
        builder = global_builder.get_ir_builder()

        @dispatcher.register(scalar=RuntimeInt)
        def _(scalar: RuntimeInt):
            check_type(self.a_dtype, [KT.half, KT.float_, KT.float16, KT.float32],
                       "Scalar type is not supported in set_tensor_a")
            builder.create_asc_MatmulSetTensorAScalarOp(self.to_ir(), _mat(scalar, self.a_dtype).to_ir())

        @dispatcher.register(tensor=BaseTensor, transpose=DefaultValued(RuntimeBool, False))
        def _(tensor: BaseTensor, transpose: RuntimeBool = False):
            check_type(tensor.dtype, [KT.half, KT.float_, KT.int8, KT.float16, KT.float32],
                       "Tensor type is not supported in set_tensor_a")
            transpose = _mat(transpose, KT.bit)
            builder.create_asc_MatmulSetTensorAOp(self.to_ir(), tensor.to_ir(), transpose.to_ir())

        dispatcher(*args, **kwargs)

    @overload
    def set_tensor_b(self, scalar: int) -> None:
        ...

    @overload
    def set_tensor_b(self, tensor: BaseTensor, transpose: bool = False) -> None:
        ...

    @require_jit
    @set_matmul_docstring(api_name="set_tensor_b")
    def set_tensor_b(self, *args, **kwargs) -> None:
        dispatcher = OverloadDispatcher(__name__)
        builder = global_builder.get_ir_builder()

        @dispatcher.register(scalar=RuntimeInt)
        def _(scalar: RuntimeInt):
            check_type(self.b_dtype, [KT.half, KT.float_, KT.float16, KT.float32],
                       "Scalar type is not supported in set_tensor_b")
            builder.create_asc_MatmulSetTensorBScalarOp(self.to_ir(), _mat(scalar, self.b_dtype).to_ir())

        @dispatcher.register(tensor=BaseTensor, transpose=DefaultValued(RuntimeBool, False))
        def _(tensor: BaseTensor, transpose: RuntimeBool = False):
            check_type(tensor.dtype, [KT.half, KT.float_, KT.int8, KT.float16, KT.float32],
                       "Tensor type is not supported in set_tensor_b")
            transpose = _mat(transpose, KT.bit)
            builder.create_asc_MatmulSetTensorBOp(self.to_ir(), tensor.to_ir(), transpose.to_ir())

        dispatcher(*args, **kwargs)

    @require_jit
    @set_matmul_docstring(api_name="set_bias")
    def set_bias(self, tensor: BaseTensor) -> None:
        if self.a_dtype == KT.int8 and self.b_dtype == KT.int8:
            check_type(tensor.dtype, [KT.int32, KT.int_], "Bias type is not supported when input A and B is int8")
        else:
            check_type(tensor.dtype, [KT.half, KT.float_, KT.float16, KT.float32],
                       "Bias type is not supported in set_bias")
        global_builder.get_ir_builder().create_asc_MatmulSetBiasOp(self.to_ir(), tensor.to_ir())

    @require_jit
    @set_matmul_docstring(api_name="disable_bias")
    def disable_bias(self) -> None:
        global_builder.get_ir_builder().create_asc_MatmulDisableBiasOp(self.to_ir())

    @require_jit
    @set_matmul_docstring(api_name="set_self_define_data")
    def set_self_define_data(self, data_ptr: Union[GlobalAddress, RuntimeInt]) -> None:
        if isinstance(data_ptr, GlobalAddress):
            data_ptr_ir = data_ptr.to_ir()
        else:
            data_ptr_ir = _mat(data_ptr, KT.uint64).to_ir()

        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_MatmulSetSelfDefineDataOp(self.to_ir(), data_ptr_ir)
        return GlobalAddress(handle)

    @require_jit
    @set_matmul_docstring(api_name="set_sparse_index")
    def set_sparse_index(self, index_global: GlobalTensor) -> None:
        builder = global_builder.get_ir_builder()
        builder.create_asc_MatmulSetSparseIndexOp(self.to_ir(), index_global.to_ir())

    @require_jit
    @set_matmul_docstring(api_name="set_user_def_info")
    def set_user_def_info(self, tiling_ptr: GlobalAddress) -> None:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_MatmulSetUserDefInfoOp(self.to_ir(), tiling_ptr.to_ir())
        return GlobalAddress(handle)

    @require_jit
    @set_matmul_docstring(api_name="iterate_n_batch")
    def iterate_n_batch(self, batch_loop: RuntimeInt, batch_a: RuntimeInt, batch_b: RuntimeInt, \
                        en_sequential_write: RuntimeBool, matrix_stride_a: RuntimeInt = 0, \
                        matrix_stride_b: RuntimeInt = 0, matrix_stride_c: RuntimeInt = 0, \
                        sync: RuntimeBool = True, wait_iterate_batch: RuntimeBool = False) -> None:
        batch_loop = _mat(batch_loop, KT.uint32)
        batch_a = _mat(batch_a, KT.uint32)
        batch_b = _mat(batch_b, KT.uint32)
        wait_iterate_batch = _mat(wait_iterate_batch, KT.bit)
        matrix_stride_a = _mat(matrix_stride_a, KT.uint32)
        matrix_stride_b = _mat(matrix_stride_b, KT.uint32)
        matrix_stride_c = _mat(matrix_stride_c, KT.uint32)
        en_sequential_write = _mat(en_sequential_write, KT.bit)
        sync = _mat(sync, KT.bit)
        builder = global_builder.get_ir_builder()
        builder.create_asc_MatmulIterateNBatchOp(self.to_ir(), batch_loop.to_ir(), batch_a.to_ir(), \
            batch_b.to_ir(), en_sequential_write.to_ir(), matrix_stride_a.to_ir(), \
            matrix_stride_b.to_ir(), matrix_stride_c.to_ir(), sync.to_ir(), wait_iterate_batch.to_ir())

    @require_jit
    @set_matmul_docstring(api_name="set_hf32")
    def set_hf32(self, enable_hf32: RuntimeBool = False, trans_mode: RuntimeInt = 0) -> None:
        enable_hf32 = _mat(enable_hf32, KT.bit)
        trans_mode = _mat(trans_mode, KT.int32)
        builder = global_builder.get_ir_builder()
        builder.create_asc_MatmulSetHF32Op(self.to_ir(), enable_hf32.to_ir(), trans_mode.to_ir())

    @require_jit
    @set_matmul_docstring(api_name="set_tail")
    def set_tail(self, tail_m: RuntimeInt = -1, tail_n: RuntimeInt = -1, tail_k: RuntimeInt = -1) -> None:
        tail_m = _mat(tail_m, KT.int_)
        tail_n = _mat(tail_n, KT.int_)
        tail_k = _mat(tail_k, KT.int_)
        builder = global_builder.get_ir_builder()
        builder.create_asc_MatmulSetTailOp(self.to_ir(), tail_m.to_ir(), tail_n.to_ir(), tail_k.to_ir())

    @require_jit
    @set_matmul_docstring(api_name="set_batch_num")
    def set_batch_num(self, batch_a: RuntimeInt, batch_b: RuntimeInt) -> None:
        batch_a = _mat(batch_a, KT.int32)
        batch_b = _mat(batch_b, KT.int32)
        builder = global_builder.get_ir_builder()
        builder.create_asc_MatmulSetBatchNumOp(self.to_ir(), batch_a.to_ir(), batch_b.to_ir())

    @require_jit
    @set_matmul_docstring(api_name="set_workspace")
    def set_workspace(self, addr: Union[GlobalTensor, GlobalAddress], size: Optional[RuntimeInt] = None) -> None:
        builder = global_builder.get_ir_builder()
        if size is not None:
            size = _mat(size, KT.int_)
            builder.create_asc_MatmulSetWorkspaceOp(self.to_ir(), addr.to_ir(), size.to_ir())
        else:
            builder.create_asc_MatmulSetWorkspaceOp(self.to_ir(), addr.to_ir())

    @require_jit
    @set_matmul_docstring(api_name="wait_get_tensor_c")
    def wait_get_tensor_c(self) -> None:
        builder = global_builder.get_ir_builder()
        builder.create_asc_MatmulWaitGetTensorCOp(self.to_ir())

    @require_jit
    @set_matmul_docstring(api_name="get_offset_c")
    def get_offset_c(self) -> MatrixOffset:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_MatmulGetOffsetCOp(builder.get_asc_MatrixOffsetType(), self.to_ir())
        return MatrixOffset(handle=handle)

    @require_jit
    @set_matmul_docstring(api_name="async_get_tensor_c")
    def async_get_tensor_c(self, c: LocalTensor) -> None:
        builder = global_builder.get_ir_builder()
        builder.create_asc_MatmulAsyncGetTensorCOp(self.to_ir(), c.to_ir())

    @overload
    def set_quant_scalar(self, quant_scalar: int) -> None:
        ...

    @require_jit
    @set_matmul_docstring(api_name="set_quant_scalar")
    def set_quant_scalar(self, quant_scalar: RuntimeInt) -> None:
        global_builder.get_ir_builder().create_asc_MatmulSetQuantScalarOp(self.to_ir(),
                                                                          _mat(quant_scalar, KT.uint64).to_ir())

    @require_jit
    @set_matmul_docstring(api_name="set_quant_vector")
    def set_quant_vector(self, quant_vector: GlobalTensor) -> None:
        global_builder.get_ir_builder().create_asc_MatmulSetQuantVectorOp(self.to_ir(), quant_vector.to_ir())

    @overload
    def set_org_shape(self, org_m: int, org_n: int, org_ka: int, \
        org_kb: Optional[int] = None, org_kc: Optional[int] = None) -> None:
        ...

    @require_jit
    @set_matmul_docstring(api_name="set_org_shape")
    def set_org_shape(self, org_m: RuntimeInt, org_n: RuntimeInt, org_ka: RuntimeInt, \
        org_kb: Optional[RuntimeInt] = None, org_kc: Optional[RuntimeInt] = None) -> None:
        if org_kb is None:
            global_builder.get_ir_builder().create_asc_MatmulSetOrgShapeOp(self.to_ir(),
                                                                           _mat(org_m, KT.int32).to_ir(),
                                                                           _mat(org_n, KT.int32).to_ir(),
                                                                           _mat(org_ka, KT.int32).to_ir())
        else:
            org_kc_val = _mat(org_kc, KT.int32).to_ir() if org_kc is not None else None
            global_builder.get_ir_builder().create_asc_MatmulSetOrgShapeOp(self.to_ir(),
                                                                           _mat(org_m, KT.int32).to_ir(),
                                                                           _mat(org_n, KT.int32).to_ir(),
                                                                           _mat(org_ka, KT.int32).to_ir(),
                                                                           _mat(org_kb, KT.int32).to_ir(), org_kc_val)

    @overload
    def set_single_shape(self, single_m: int, single_n: int, single_k: int) -> None:
        ...

    @require_jit
    @set_matmul_docstring(api_name="set_single_shape")
    def set_single_shape(self, single_m: RuntimeInt, single_n: RuntimeInt, single_k: RuntimeInt) -> None:
        global_builder.get_ir_builder().create_asc_MatmulSetSingleShapeOp(self.to_ir(),
                                                                          _mat(single_m, KT.int32).to_ir(),
                                                                          _mat(single_n, KT.int32).to_ir(),
                                                                          _mat(single_k, KT.int32).to_ir())


class MatmulIterator:

    def __init__(self, matmul: Matmul, partial_sum: RuntimeBool, local_c_matrix: BaseTensor, sync: RuntimeBool):
        self.matmul = matmul
        self.partial_sum = partial_sum
        self.local_c_matrix = local_c_matrix
        self.sync = sync
        self.insert_point = None
        self.count = None

    def __enter__(self) -> RuntimeInt:
        builder = global_builder.get_ir_builder()
        zero = builder.get_i32(0)
        op = builder.create_scf_WhileOp([zero.get_type()], [zero])
        self.insert_point = builder.save_insertion_point()
        before = builder.create_block(op.get_before())
        before.add_argument(zero.get_type())
        builder.set_insertion_point_to_start(before)
        partial_sum = _mat(self.partial_sum, KT.bit)
        sync = _mat(self.sync, KT.bit)
        local_c_matrix = self.local_c_matrix.to_ir() if self.local_c_matrix is not None else None
        it = builder.create_asc_MatmulIterateOp(builder.get_i1_type(), self.matmul.to_ir(), partial_sum.to_ir(),
                                                local_c_matrix, sync.to_ir())
        builder.create_scf_ConditionOp(it, [before.get_argument(0)])
        after = builder.create_block(op.get_after())
        after.add_argument(zero.get_type())
        builder.set_insertion_point_to_start(after)
        self.count = PlainValue(after.get_argument(0), KT.int32)
        return self.count

    def __exit__(self, *args) -> None:
        self.count = self.count.__add__(1)
        builder = global_builder.get_ir_builder()
        builder.create_scf_YieldOp([self.count.to_ir()])
        builder.restore_insertion_point(self.insert_point)


class MatrixOffset(IRValue):

    @overload
    def __init__(self, offset: int, row: int, col: int, height: int, width: int) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(self, offset: Optional[RuntimeInt] = None, row: Optional[RuntimeInt] = None, \
                col: Optional[RuntimeInt] = None, height: Optional[RuntimeInt] = None, \
                width: Optional[RuntimeInt] = None, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_MatrixOffsetType(),
            [
                _mat(offset, KT.int32).to_ir(),
                _mat(row, KT.int32).to_ir(),
                _mat(col, KT.int32).to_ir(),
                _mat(height, KT.int32).to_ir(),
                _mat(width, KT.int32).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_i32_type()] * 5),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> MatrixOffset:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


@overload
def get_basic_config(basic_m: int, basic_n: int, basic_k: int, intrinsics_limit: Optional[bool] = False,
                     batch_loop: Optional[bool] = False, bmm_mode: Optional[BatchMode] = BatchMode.BATCH_LESS_THAN_L1)\
                     -> MatmulConfig:
    ...


@require_jit
@set_matmul_docstring(api_name="get_basic_config")
def get_basic_config(basic_m: RuntimeInt, basic_n: RuntimeInt, basic_k: RuntimeInt,
                     intrinsics_limit: RuntimeBool = False, batch_loop: RuntimeBool = False,
                     bmm_mode: BatchMode = BatchMode.BATCH_LESS_THAN_L1) -> MatmulConfig:
    if intrinsics_limit is None:
        intrinsics_limit = False
    if batch_loop is None:
        batch_loop = False
    if bmm_mode is None:
        bmm_mode = BatchMode.BATCH_LESS_THAN_L1
    mm_config = MatmulConfig(basic_m=basic_m, basic_n=basic_n, basic_k=basic_k, intrinsics_check=intrinsics_limit,
                             is_n_batch=batch_loop, batch_mode=bmm_mode)
    return mm_config


@overload
def get_ib_share_norm_config(intrinsics_limit: Optional[bool] = False, batch_loop: Optional[bool] = False,
                             is_vec_nd2_nz: Optional[bool] = False,
                             bmm_mode: Optional[BatchMode] = BatchMode.BATCH_LESS_THAN_L1,
                             is_double_cache: Optional[bool] = False, en_unit_flag: Optional[bool] = True) \
        -> MatmulConfig:
    ...


@require_jit
@set_matmul_docstring(api_name="get_ib_share_norm_config")
def get_ib_share_norm_config(intrinsics_limit: RuntimeBool = False, batch_loop: RuntimeBool = False,
                             is_vec_nd2_nz: RuntimeBool = False,
                             bmm_mode: BatchMode = BatchMode.BATCH_LESS_THAN_L1,
                             is_double_cache: RuntimeBool = False, en_unit_flag: RuntimeBool = True) \
        -> MatmulConfig:
    if intrinsics_limit is None:
        intrinsics_limit = False
    if batch_loop is None:
        batch_loop = False
    if is_vec_nd2_nz is None:
        is_vec_nd2_nz = False
    if bmm_mode is None:
        bmm_mode = BatchMode.BATCH_LESS_THAN_L1
    if is_double_cache is None:
        is_double_cache = False
    if en_unit_flag is None:
        en_unit_flag = True
    mm_config = MatmulConfig(intrinsics_check=intrinsics_limit, is_n_batch=batch_loop, en_vec_nd2nz=is_vec_nd2_nz,
                             batch_mode=bmm_mode, enable_double_cache=is_double_cache, en_unit_flag=en_unit_flag)
    return mm_config


@overload
def get_matmul_api_tiling(mm_cfg: MatmulConfig, l1_size: int, a_type: MatmulType, b_type: MatmulType,
                          c_type: MatmulType, bias_type: Optional[MatmulType] = None) -> MatmulApiStaticTiling:
    ...


@require_jit
@set_matmul_docstring(api_name="get_matmul_api_tiling")
def get_matmul_api_tiling(mm_cfg: MatmulConfig, a_type: MatmulType, b_type: MatmulType, c_type: MatmulType,
                          bias_type: MatmulType, l1_size: Optional[RuntimeInt] = None) -> MatmulApiStaticTiling:
    builder = global_builder.get_ir_builder()
    ir_type = builder.get_matmul_type(
        a_type.position, a_type.format, a_type.dtype.to_ir(), a_type.is_trans, a_type.layout,  # a
        b_type.position, b_type.format, b_type.dtype.to_ir(), b_type.is_trans, b_type.layout,  # b
        c_type.position, c_type.format, c_type.dtype.to_ir(), c_type.is_trans, c_type.layout,  # c
        bias_type.position, bias_type.format, bias_type.dtype.to_ir(), mm_cfg.do_norm, mm_cfg.do_basic_block,
        mm_cfg.do_multi_data_load, mm_cfg.basic_m, mm_cfg.basic_n, mm_cfg.basic_k, mm_cfg.intrinsics_check,
        mm_cfg.is_n_batch, mm_cfg.en_vec_nd2nz, mm_cfg.do_special_basic_block, mm_cfg.do_mte2_preload,
        mm_cfg.single_core_m, mm_cfg.single_core_n, mm_cfg.single_core_k, mm_cfg.step_m, mm_cfg.step_n, mm_cfg.base_mn,
        mm_cfg.single_core_mn, mm_cfg.en_unit_flag, mm_cfg.is_per_tensor, mm_cfg.has_anti_quant_offset,
        mm_cfg.do_ib_share_norm, mm_cfg.do_special_mdl, mm_cfg.enable_init, mm_cfg.batch_mode, mm_cfg.enable_end,
        mm_cfg.enable_get_tensor_c, mm_cfg.enable_set_org_shape, mm_cfg.enable_set_bias, mm_cfg.enable_set_tail,
        mm_cfg.enable_quant_vector, mm_cfg.enable_set_define_data, mm_cfg.iterate_mode, mm_cfg.enable_reuse,
        mm_cfg.enable_ub_reuse, mm_cfg.enable_l1_cache_ub, mm_cfg.intra_block_part_sum, mm_cfg.iterate_order,
        mm_cfg.schedule_type, mm_cfg.enable_double_cache, mm_cfg.is_bias_batch, mm_cfg.enable_static_pad_zeros,
        mm_cfg.is_partial_output, mm_cfg.enable_mix_dual_master, mm_cfg.is_a2b2_shared, mm_cfg.is_enable_channel_split,
        mm_cfg.enable_kdim_reorder_load, mm_cfg.is_co1_shared, mm_cfg.shared_co1_buffer_size, mm_cfg.batch_out_mode)

    if not l1_size:
        l1_size = property(prop=TOTAL_L1_SIZE, builder=builder)

    builder.create_asc_MatmulGetMatmulApiTilingOp(builder.get_asc_MatmulApiStaticTilingType(), mm_cfg.to_ir(),
                                                  _mat(l1_size).to_ir(), ir_type)


@overload
def get_mdl_config(intrinsics_limit: Optional[bool] = False, batch_loop: Optional[bool] = False,
                   do_mte2_preload: Optional[int] = 0, is_vec_nd2_nz: Optional[bool] = False,
                   is_per_tensor: Optional[bool] = False, has_anti_quant_offset: Optional[bool] = False,
                   en_unit_flag: Optional[bool] = False, is_msg_reuse: Optional[bool] = True,
                   enable_ub_reuse: Optional[bool] = True, enable_l1_cache_ub: Optional[bool] = False,
                   enable_mix_dual_master: Optional[bool] = False,
                   enable_kdim_reorder_load: Optional[bool] = False) -> MatmulConfig:
    ...


@require_jit
@set_matmul_docstring(api_name="get_mdl_config")
def get_mdl_config(intrinsics_limit: RuntimeBool = False, batch_loop: RuntimeBool = False,
                   do_mte2_preload: RuntimeInt = 0, is_vec_nd2_nz: RuntimeBool = False,
                   is_per_tensor: RuntimeBool = False, has_anti_quant_offset: RuntimeBool = False,
                   en_unit_flag: RuntimeBool = False, is_msg_reuse: RuntimeBool = True,
                   enable_ub_reuse: RuntimeBool = True, enable_l1_cache_ub: RuntimeBool = False,
                   enable_mix_dual_master: RuntimeBool = False,
                   enable_kdim_reorder_load: RuntimeBool = False) -> MatmulConfig:
    mm_config = MatmulConfig(intrinsics_check=intrinsics_limit, is_n_batch=batch_loop, do_mte2_preload=do_mte2_preload,
                             en_vec_nd2nz=is_vec_nd2_nz, is_per_tensor=is_per_tensor,
                             has_anti_quant_offset=has_anti_quant_offset, en_unit_flag=en_unit_flag,
                             enable_reuse=is_msg_reuse, enable_ub_reuse=enable_ub_reuse,
                             enable_l1_cache_ub=enable_l1_cache_ub, enable_mix_dual_master=enable_mix_dual_master,
                             enable_kdim_reorder_load=enable_kdim_reorder_load)
    return mm_config


@overload
def get_mm_config(*args, **kwargs) -> MatmulConfig:
    ...


@require_jit
@set_matmul_docstring(api_name="get_mm_config")
def get_mm_config(*args, **kwargs) -> MatmulConfig:
    norm = True
    mdl = False
    special_mdl = False
    ib_share = False
    single_core_m = 0
    single_core_n = 0
    single_core_k = 0
    basic_m = 0
    basic_n = 0
    basic_k = 0
    is_per_tensor = False
    has_anti_quant_offset = False
    is_n_batch = False
    batch_mode = BatchMode.BATCH_LESS_THAN_L1
    is_bias_batch = False
    intrinsics_limit = False
    en_vec_nd2_nz = False
    enable_l1_cache_ub = False
    do_mte2_preload = 0
    iterate_order = IterateOrder.ORDER_M
    schedule_type = ScheduleType.INNER_PRODUCT
    enable_reuse = False
    enable_ub_reuse = False
    is_partial_output = False
    is_a2_b2_shared = False
    is_enable_channel_split = False
    enable_kdim_reorder_load = False
    for arg in [args, kwargs]:
        if isinstance(arg, MatmulShapeParams):
            single_core_m = arg.single_core_m
            single_core_n = arg.single_core_n
            single_core_k = arg.single_core_k
            basic_m = arg.basic_m
            basic_n = arg.basic_n
            basic_k = arg.basic_k
        if isinstance(arg, MatmulQuantParams):
            is_per_tensor = arg.is_per_tensor
            has_anti_quant_offset = arg.has_anti_quant_offset
        if isinstance(arg, MatmulBatchParams):
            is_n_batch = arg.is_n_batch
            batch_mode = arg.batch_mode
            is_bias_batch = arg.is_bias_batch
        if isinstance(arg, MatmulFuncParams):
            intrinsics_limit = arg.intrinsics_limit
            en_vec_nd2_nz = arg.intrinsics_limit
            enable_l1_cache_ub = arg.enable_l1_cache
            do_mte2_preload = arg.do_mte2_pre_load
            iterate_order = arg.iterate_order
            schedule_type = arg.schedule_type
            enable_reuse = arg.enable_reuse
            enable_ub_reuse = arg.enable_ub_reuse
            is_partial_output = arg.is_partial_output
            is_a2_b2_shared = arg.is_a2_b2_shared
            enable_kdim_reorder_load = arg.enable_kdim_reorder_load
        if isinstance(arg, MatmulConfigMode):
            if arg == MatmulConfigMode.CONFIG_NORM:
                norm = True
            if arg == MatmulConfigMode.CONFIG_MDL:
                mdl = True
            if arg == MatmulConfigMode.CONFIG_SPECIALMDL:
                special_mdl = True
            if arg == MatmulConfigMode.CONFIG_IBSHARE:
                ib_share = True
        check_type(batch_mode, [1, 2, 3], "Params batch_mode should be in [1, 2, 3]")
        check_type(iterate_order, [0, 1, 2], "Params iterate_order should be in [0, 1, 2]")
        check_type(schedule_type, [0, 1], "Params schedule_type should be in [0, 1]")
        check_type(do_mte2_preload, [0, 1, 2], "Params do_mte2_preload should be in [0, 1, 2]")

    return MatmulConfig(single_core_m=single_core_m, single_core_n=single_core_n, single_core_k=single_core_k,
                        basic_m=basic_m, basic_n=basic_n, basic_k=basic_k, is_per_tensor=is_per_tensor,
                        has_anti_quant_offset=has_anti_quant_offset, is_n_batch=is_n_batch, batch_mode=batch_mode,
                        is_bias_batch=is_bias_batch, intrinsics_check=intrinsics_limit, en_vec_nd2nz=en_vec_nd2_nz,
                        enable_l1_cache_ub=enable_l1_cache_ub, do_mte2_preload=do_mte2_preload,
                        iterate_order=iterate_order, schedule_type=schedule_type, enable_reuse=enable_reuse,
                        enable_ub_reuse=enable_ub_reuse, is_partial_output=is_partial_output,
                        is_a2b2_shared=is_a2_b2_shared, is_enable_channel_split=is_enable_channel_split,
                        enable_kdim_reorder_load=enable_kdim_reorder_load, do_norm=norm, do_multi_data_load=mdl,
                        do_special_mdl=special_mdl, do_ib_share_norm=ib_share)


@overload
def get_normal_config(intrinsics_limit: Optional[bool] = False, batch_loop: Optional[bool] = False,
                      is_vec_nd2_nz: Optional[bool] = False,
                      bmm_mode: Optional[BatchMode] = BatchMode.BATCH_LESS_THAN_L1, is_msg_reuse: Optional[bool] = True,
                      iterate_order: Optional[IterateOrder] = IterateOrder.ORDER_M,
                      schedule_type: Optional[ScheduleType] = ScheduleType.INNER_PRODUCT,
                      en_unit_flag: Optional[bool] = True) -> MatmulConfig:
    ...


@require_jit
@set_matmul_docstring(api_name="get_normal_config")
def get_normal_config(intrinsics_limit: RuntimeBool = False, batch_loop: RuntimeBool = False,
                      is_vec_nd2_nz: RuntimeBool = False,
                      bmm_mode: BatchMode = BatchMode.BATCH_LESS_THAN_L1,
                      is_msg_reuse: RuntimeBool = True, iterate_order: IterateOrder = IterateOrder.ORDER_M,
                      schedule_type: ScheduleType = ScheduleType.INNER_PRODUCT, en_unit_flag: RuntimeBool = True)\
                      -> MatmulConfig:
    if intrinsics_limit is None:
        intrinsics_limit = False
    if batch_loop is None:
        batch_loop = False
    if is_vec_nd2_nz is None:
        is_vec_nd2_nz = False
    if bmm_mode is None:
        bmm_mode = BatchMode.BATCH_LESS_THAN_L1
    if is_msg_reuse is None:
        is_msg_reuse = True
    if iterate_order is None:
        iterate_order = IterateOrder.ORDER_M
    if schedule_type is None:
        schedule_type = ScheduleType.INNER_PRODUCT
    if en_unit_flag is None:
        en_unit_flag = True
    mm_config = MatmulConfig(intrinsics_check=intrinsics_limit, is_n_batch=batch_loop, en_vec_nd2nz=is_vec_nd2_nz,
                             batch_mode=bmm_mode, enable_reuse=is_msg_reuse, iterate_order=iterate_order,
                             schedule_type=schedule_type, en_unit_flag=en_unit_flag)
    return mm_config


@overload
def get_special_basic_config(basic_m: int, basic_n: int, basic_k: int, single_core_m: int, single_core_n: int,
                             single_core_k: int, step_m: int, step_n: int, intrinsics_limit: Optional[bool] = False,
                             batch_loop: Optional[bool] = False,
                             bmm_mode: Optional[BatchMode] = BatchMode.BATCH_LESS_THAN_L1) -> MatmulConfig:
    ...


@require_jit
@set_matmul_docstring(api_name="get_special_basic_config")
def get_special_basic_config(basic_m: RuntimeInt, basic_n: RuntimeInt, basic_k: RuntimeInt,
                             single_core_m: RuntimeInt, single_core_n: RuntimeInt, single_core_k: RuntimeInt,
                             step_m: RuntimeInt, step_n: RuntimeInt, intrinsics_limit: RuntimeBool = False,
                             batch_loop: RuntimeBool = False, bmm_mode: BatchMode = BatchMode.BATCH_LESS_THAN_L1)\
                             -> MatmulConfig:
    if intrinsics_limit is None:
        intrinsics_limit = False
    if batch_loop is None:
        batch_loop = False
    if bmm_mode is None:
        bmm_mode = BatchMode.BATCH_LESS_THAN_L1
    mm_config = MatmulConfig(basic_m=basic_m, basic_n=basic_n, basic_k=basic_k, single_core_m=single_core_m,
                             single_core_n=single_core_n, single_core_k=single_core_k, step_m=step_m, step_n=step_n,
                             intrinsics_check=intrinsics_limit, is_n_batch=batch_loop, batch_mode=bmm_mode)
    return mm_config


@overload
def get_special_mdl_config(intrinsics_limit: Optional[bool] = False, batch_loop: Optional[bool] = False,
                           do_mte2_pre_load: Optional[int] = 0, is_vec_nd2_nz: Optional[bool] = False,
                           is_per_tensor: Optional[bool] = False, has_anti_quant_offset: Optional[bool] = False) \
        -> MatmulConfig:
    ...


@require_jit
@set_matmul_docstring(api_name="get_special_mdl_config")
def get_special_mdl_config(intrinsics_limit: RuntimeBool = False, batch_loop: RuntimeBool = False,
                           do_mte2_pre_load: RuntimeInt = 0, is_vec_nd2_nz: RuntimeBool = False,
                           is_per_tensor: RuntimeBool = False, has_anti_quant_offset: RuntimeBool = False) \
        -> MatmulConfig:
    if intrinsics_limit is None:
        intrinsics_limit = False
    if batch_loop is None:
        batch_loop = False
    if do_mte2_pre_load is None:
        do_mte2_pre_load = 0
    if is_vec_nd2_nz is None:
        is_vec_nd2_nz = False
    if is_per_tensor is None:
        is_per_tensor = False
    if has_anti_quant_offset is None:
        has_anti_quant_offset = False
    mm_config = MatmulConfig(intrinsics_check=intrinsics_limit, is_n_batch=batch_loop, do_mte2_preload=do_mte2_pre_load,
                             en_vec_nd2nz=is_vec_nd2_nz, is_per_tensor=is_per_tensor,
                             has_anti_quant_offset=has_anti_quant_offset)
    return mm_config
