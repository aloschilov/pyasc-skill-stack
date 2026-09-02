# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations
from typing import Any, Iterable, List, Optional, Tuple, Union, overload

import numpy as np

from ..._C import ir
from ...common.compat import isinstance
from .array import Array
from .dtype import DataType, KnownTypes
from .enums import BlockMode, CO2Layout, DataFormat, DeqScale, pad_t, QuantModes, TransposeType
from .ir_value import IRHandle, IRValue, PlainValue, \
                            RuntimeBool, RuntimeInt, RuntimeNumeric, materialize_ir_value as _mat, convert_value
from .utils import require_jit, global_builder
from .properties import property, ONE_BLK_SIZE
from .dtype import KnownTypes as KT


class BinaryRepeatParams(IRValue):

    @overload
    def __init__(self, dst_blk_stride: int = 1, src0_blk_stride: int = 1, src1_blk_stride: int = 1,
                 dst_rep_stride: int = 8, src0_rep_stride: int = 8, src1_rep_stride: int = 8) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, dst_blk_stride: RuntimeInt = 1, src0_blk_stride: RuntimeInt = 1, src1_blk_stride: RuntimeInt = 1,
                 dst_rep_stride: RuntimeInt = 8, src0_rep_stride: RuntimeInt = 8, src1_rep_stride: RuntimeInt = 8,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_BinaryRepeatParamsType(),
            [
                _mat(dst_blk_stride).to_ir(),
                _mat(src0_blk_stride).to_ir(),
                _mat(src1_blk_stride).to_ir(),
                _mat(dst_rep_stride).to_ir(),
                _mat(src0_rep_stride).to_ir(),
                _mat(src1_rep_stride).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_ui8_type()] * 6),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> BinaryRepeatParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class BrcbRepeatParams(IRValue):

    @overload
    def __init__(self, dst_blk_stride: int = 1, dst_rep_stride: int = 8) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, dst_blk_stride: RuntimeInt = 1, dst_rep_stride: RuntimeInt = 8,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_BrcbRepeatParamsType(),
            [
                _mat(dst_blk_stride, KT.uint16).to_ir(),
                _mat(dst_rep_stride, KT.uint16).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_ui16_type(), builder.get_ui16_type()]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> BrcbRepeatParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class CopyRepeatParams(IRValue):

    @overload
    def __init__(self, dst_stride: int = 0, src_stride: int = 0, dst_repeat_size: int = 0,
                 src_repeat_size: int = 0) -> None:
        ...

    @require_jit
    def __init__(self, dst_stride: RuntimeInt = 0, src_stride: RuntimeInt = 0, dst_repeat_size: RuntimeInt = 0,
                 src_repeat_size: RuntimeInt = 0, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_CopyRepeatParamsType(),
            [
                _mat(dst_stride, KnownTypes.uint16).to_ir(),
                _mat(src_stride, KnownTypes.uint16).to_ir(),
                _mat(dst_repeat_size, KnownTypes.uint16).to_ir(),
                _mat(src_repeat_size, KnownTypes.uint16).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_ui16_type()] * 4),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> CopyRepeatParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class DataCopyParams(IRValue):

    @overload
    def __init__(self, block_count: int = 1, block_len: int = 0, src_stride: int = 0, dst_stride: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, block_count: RuntimeInt = 1, block_len: RuntimeInt = 0, src_stride: RuntimeInt = 0,
                 dst_stride: RuntimeInt = 0, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_DataCopyParamsType(),
            [
                _mat(block_count).to_ir(),
                _mat(block_len).to_ir(),
                _mat(src_stride).to_ir(),
                _mat(dst_stride).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_ui16_type()] * 4),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> DataCopyParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class DataCopyEnhancedParams(IRValue):

    @overload
    def __init__(self, block_mode: BlockMode = BlockMode.BLOCK_MODE_NORMAL, deq_scale: DeqScale = DeqScale.DEQ_NONE,
                 deq_value_in: int = 0, sid_store_mode_in: int = 0, is_relu_in: bool = False,
                 pad_mode_in: pad_t = pad_t.PAD_NONE, pad_value_in: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, block_mode: BlockMode = BlockMode.BLOCK_MODE_NORMAL, deq_scale: DeqScale = DeqScale.DEQ_NONE,
                 deq_value_in: RuntimeInt = 0, sid_store_mode_in: RuntimeInt = 0, is_relu_in: RuntimeBool = False,
                 pad_mode_in: pad_t = pad_t.PAD_NONE, pad_value_in: RuntimeInt = 0,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_DataCopyEnhancedParamsType(),
            [
                _mat(block_mode).to_ir(),
                _mat(deq_scale).to_ir(),
                _mat(deq_value_in).to_ir(),
                _mat(sid_store_mode_in).to_ir(),
                _mat(is_relu_in).to_ir(),
                _mat(pad_mode_in).to_ir(),
                _mat(pad_value_in).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_asc_BlockModeType(),
                builder.get_asc_DeqScaleType(),
                builder.get_ui64_type(),
                builder.get_ui8_type(),
                builder.get_i1_type(),
                builder.get_asc_pad_tType(),
                builder.get_ui64_type()
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> DataCopyEnhancedParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class InitConstValueParams(IRValue):

    @overload
    def __init__(self, repeat_times: int = 0, block_num: int = 0, dst_gap: int = 0, init_value=0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        ...

    @require_jit
    def __init__(
        self,
        repeat_times: RuntimeInt = 0,
        block_num: RuntimeInt = 0,
        dst_gap: RuntimeInt = 0,
        init_value=0,
        handle: Optional[IRHandle] = None,
    ) -> None:

        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()

        if isinstance(init_value, IRValue):
            init_ir = init_value.to_ir()
            init_type = init_ir.get_type()

        elif isinstance(init_value, np.generic):
            numpy_type_map = {
                np.float16: KT.float16,
                np.float32: KT.float32,
                np.int16: KT.int16,
                np.uint16: KT.uint16,
                np.int32: KT.int32,
                np.uint32: KT.uint32,
            }
            np_type = type(init_value)
            if np_type not in numpy_type_map:
                raise TypeError(f"Unsupported numpy scalar type: {np_type}")
            pv = convert_value(init_value.item(), numpy_type_map[np_type])

        else:
            pv = _mat(init_value)

        init_ir = pv.to_ir()
        init_type = pv.dtype.to_ir()

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_InitConstValueParamsType(),
            [
                _mat(repeat_times, KnownTypes.uint16).to_ir(),
                _mat(block_num, KnownTypes.uint16).to_ir(),
                _mat(dst_gap, KnownTypes.uint16).to_ir(), init_ir
            ],
            builder.get_type_array_attr(
                [builder.get_ui16_type(),
                 builder.get_ui16_type(),
                 builder.get_ui16_type(), init_type]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "InitConstValueParams":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class ShapeInfo(IRValue):
    """
    ShapeInfo用来存放LocalTensor或GlobalTensor的shape信息。
    """

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, shape: Array, original_shape: Optional[Array] = None,
                 data_format: DataFormat = DataFormat.ND) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(self, shape: Optional[Array] = None, original_shape: Optional[Array] = None,
                 data_format: Optional[DataFormat] = None, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        operands = []
        types = []
        builder = global_builder.get_ir_builder()
        if shape is not None:
            builder.set_emit_as_unsigned(shape.to_ir().get_defining_op())
            shape_len = _mat(len(shape), KnownTypes.int8).to_ir()
            operands += [shape_len, shape.to_ir()]
            ir_memref = ir.get_unranked_memref_type(builder.get_ui32_type())
            types += [builder.get_ui8_type(), ir_memref]
            if original_shape is not None:
                builder.set_emit_as_unsigned(original_shape.to_ir().get_defining_op())
                orig_shape_len = _mat(len(original_shape), KnownTypes.int8).to_ir()
                operands += [orig_shape_len, original_shape.to_ir()]
                types += [builder.get_ui8_type(), ir_memref]
            data_format = DataFormat.ND if data_format is None else data_format
            operands.append(_mat(data_format, KnownTypes.int8).to_ir())
            types.append(builder.get_asc_DataFormatType())
        types_attr = builder.get_type_array_attr(types)
        self.handle = builder.create_asc_ConstructOp(builder.get_asc_ShapeInfoType(), operands, types_attr)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> ShapeInfo:
        return cls(handle=handle)

    @overload
    def shape(self, dim: int) -> int:
        ...

    @require_jit
    def shape(self, dim: RuntimeInt) -> RuntimeInt:
        dim = _mat(dim).to_ir()
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_ShapeInfoShapeOp(builder.get_i32_type(), self.to_ir(), dim)
        return PlainValue(handle)

    @overload
    def original_shape(self, dim: int) -> int:
        ...

    @require_jit
    def original_shape(self, dim: RuntimeInt) -> RuntimeInt:
        dim = _mat(dim).to_ir()
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_ShapeInfoOriginalShapeOp(builder.get_i32_type(), self.to_ir(), dim)
        return PlainValue(handle)

    def to_ir(self) -> IRHandle:
        return self.handle


@overload
def get_shape_size(shape_info: ShapeInfo) -> int:
    ...


@require_jit
def get_shape_size(shape_info: ShapeInfo) -> RuntimeInt:
    """
    获取Shape中所有dim的累乘结果

    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline int GetShapeSize(const ShapeInfo& shapeInfo)

    **参数说明**

    - shape_info：ShapeInfo类型，LocalTensor或GlobalTensor的shape信息。

    **返回值说明**

    输入的shape_info中所有dim的累乘结果。

    **约束说明**

    无。

    **调用示例**

    .. code-block:: python

        size = asc.get_shape_size(shape_info)
    """
    builder = global_builder.get_ir_builder()
    handle = builder.create_asc_GetShapeSizeOp(builder.get_i32_type(), shape_info.to_ir())
    return PlainValue(handle)


class SliceInfo(IRValue):

    @overload
    def __init__(self, start_index: int = 0, end_index: int = None, stride: int = 0, burst_len: int = None,
                 shape_value: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(self, start_index: RuntimeInt = 0, end_index: RuntimeInt = None, stride: RuntimeInt = 0,
                 burst_len: RuntimeInt = None, shape_value: RuntimeInt = 0, handle: Optional[IRHandle] = None) -> None:

        builder = global_builder.get_ir_builder()

        if handle is not None:
            self.handle = handle
            return

        if not end_index:
            end_index = property(prop=ONE_BLK_SIZE, builder=builder).__sub__(1, builder=builder)
        if not burst_len:
            burst_len = property(prop=ONE_BLK_SIZE, builder=builder)

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_SliceInfoType(),
            [
                _mat(start_index).to_ir(),
                _mat(end_index).to_ir(),
                _mat(stride).to_ir(),
                _mat(burst_len).to_ir(),
                _mat(shape_value).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_ui32_type()] * 5),
        )

    @require_jit
    def __getitem__(self, slices: Any) -> Union[SliceInfo, RuntimeNumeric]:
        builder = global_builder.get_ir_builder()

        if isinstance(slices, RuntimeInt):
            handle = builder.create_asc_GetValueSliceInfoOp(self.dtype.to_ir(), self.to_ir(), _mat(slices).to_ir())
            return PlainValue(handle, self.dtype)
        if isinstance(slices, slice):
            if slices.step is not None or slices.stop is not None:
                raise RuntimeError("Slice operation with provided stop and step is not supported for SliceInfo")
            handle = builder.create_asc_SliceInfoSubIndexOp(self.to_ir().get_type(), self.to_ir(),
                                                            _mat(slices.start).to_ir())
            return SliceInfo(handle=handle)
        raise RuntimeError(f"SliceInfo subscript operation is not supported with {slices}")

    @classmethod
    def from_ir(cls, handle: IRHandle) -> SliceInfo:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class TensorShape(Tuple[int, ...]):

    @overload
    def __new__(cls):
        ...

    @overload
    def __new__(cls, size: int, /):
        ...

    @overload
    def __new__(cls, shape: Iterable[int], /):
        ...

    @overload
    def __new__(cls, shape: TensorShape, /):
        ...

    @overload
    def __new__(cls, *dims: int):
        ...

    @overload
    def __new__(cls, empty: None, /):
        ...

    def __new__(cls, *args):
        num_args = len(args)
        if num_args == 0:
            return cls.new_impl(tuple())
        if num_args > 1:
            return cls.new_impl(tuple(cls.as_int(a) for a in args))

        if num_args != 1:
            raise ValueError("num_args must be 1")
        arg = args[0]
        if arg is None:
            return cls.new_impl(tuple())
        if isinstance(arg, TensorShape):
            return arg
        if isinstance(arg, Iterable):
            return cls.new_impl(tuple(cls.as_int(a) for a in arg))
        # single value
        return cls.new_impl((cls.as_int(arg), ))

    @staticmethod
    def as_int(value: Any) -> RuntimeInt:
        try:
            return int(value)
        except Exception as e:
            raise TypeError(f"TensorShape accepts values convertible to int, got {value.__class__.__name__}") from e

    @classmethod
    def new_impl(cls, t: Tuple[int, ...]):
        return super(__class__, cls).__new__(cls, t)


class UnaryRepeatParams(IRValue):

    @overload
    def __init__(self, block_count: int = 1, block_len: int = 0, src_stride: int = 0, dst_stride: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, dst_blk_stride: RuntimeInt = 1, src_blk_stride: RuntimeInt = 1, dst_rep_stride: RuntimeInt = 8,
                 src_rep_stride: RuntimeInt = 8, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_UnaryRepeatParamsType(),
            [
                _mat(dst_blk_stride).to_ir(),
                _mat(src_blk_stride).to_ir(),
                _mat(dst_rep_stride).to_ir(),
                _mat(src_rep_stride).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> UnaryRepeatParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class TransDataTo5HDParams(IRValue):

    @overload
    def __init__(self, dst_high_half: bool = False, src_high_half: bool = False, repeat_times: int = 1,
                 dst_rep_stride: int = 0, src_rep_stride: int = 0) -> None:
        ...

    def __init__(self, dst_high_half: bool = False, src_high_half: bool = False, repeat_times: RuntimeInt = 1,
                 dst_rep_stride: RuntimeInt = 0, src_rep_stride: RuntimeInt = 0,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_TransDataTo5HDParamsType(),
            [
                _mat(dst_high_half, KnownTypes.int1).to_ir(),
                _mat(src_high_half, KnownTypes.int1).to_ir(),
                _mat(repeat_times).to_ir(),
                _mat(dst_rep_stride).to_ir(),
                _mat(src_rep_stride).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_ui8_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type()
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> BinaryRepeatParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class TransposeParamsExt(IRValue):

    @overload
    def __init__(self, n_size: int = 0, c_size: int = 0, h_size: int = 0, w_size: int = 0,
                 transpose_type: TransposeType = TransposeType.TRANSPOSE_ND2ND_B16) -> None:
        ...

    def __init__(self, n_size: RuntimeInt = 0, c_size: RuntimeInt = 0, h_size: RuntimeInt = 0, w_size: RuntimeInt = 0,
                 transpose_type: TransposeType = TransposeType.TRANSPOSE_ND2ND_B16,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_TransposeParamsExtType(),
            [
                _mat(n_size).to_ir(),
                _mat(c_size).to_ir(),
                _mat(h_size).to_ir(),
                _mat(w_size).to_ir(),
                _mat(transpose_type).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_asc_TransposeTypeType()
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> BinaryRepeatParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class DataCopyPadExtParams(IRValue):

    @overload
    def __init__(self, dtype: DataType, is_pad: bool = False, left_padding: int = 0, right_padding: int = 0,
                 padding_value: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(self, dtype: DataType, is_pad: RuntimeBool = False, left_padding: RuntimeInt = 0,
                 right_padding: RuntimeInt = 0, padding_value: RuntimeNumeric = 0,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_DataCopyPadExtParamsType(dtype.to_ir()),
            [
                _mat(is_pad).to_ir(),
                _mat(left_padding).to_ir(),
                _mat(right_padding).to_ir(),
                _mat(padding_value, dtype).to_ir(),
            ],
            builder.get_type_array_attr(
                [builder.get_i1_type(),
                 builder.get_ui16_type(),
                 builder.get_ui16_type(),
                 dtype.to_ir()]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> DataCopyPadExtParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class DataCopyPadParams(IRValue):

    @overload
    def __init__(self, is_pad: bool = False, left_padding: int = 0, right_padding: int = 0,
                 padding_value: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(self, is_pad: RuntimeBool = False, left_padding: RuntimeInt = 0, right_padding: RuntimeInt = 0,
                 padding_value: RuntimeInt = 0, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_DataCopyPadParamsType(), [
                _mat(is_pad).to_ir(),
                _mat(left_padding).to_ir(),
                _mat(right_padding).to_ir(),
                _mat(padding_value).to_ir(),
            ],
            builder.get_type_array_attr(
                [builder.get_i1_type(),
                 builder.get_ui16_type(),
                 builder.get_ui16_type(),
                 builder.get_ui64_type()]))

    @classmethod
    def from_ir(cls, handle: IRHandle) -> DataCopyPadParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class DataCopyExtParams(IRValue):

    @overload
    def __init__(self, block_count: int = 1, block_len: int = 0, src_stride: int = 0, dst_stride: int = 0,
                 rsv: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(self, block_count: RuntimeInt = 1, block_len: RuntimeInt = 0, src_stride: RuntimeInt = 0,
                 dst_stride: RuntimeInt = 0, rsv: RuntimeInt = 0, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_DataCopyExtParamsType(),
            [
                _mat(block_count).to_ir(),
                _mat(block_len).to_ir(),
                _mat(src_stride).to_ir(),
                _mat(dst_stride).to_ir(),
                _mat(rsv).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui16_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type()
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> DataCopyExtParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class GatherMaskParams(IRValue):

    @overload
    def __init__(self, src0_block_stride: int = 1, repeat_times: int = 1, src0_repeat_stride: int = 0,
                 src1_repeat_stride: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        ...

    @require_jit
    def __init__(self, src0_block_stride: RuntimeInt = 1, repeat_times: RuntimeInt = 1,
                 src0_repeat_stride: RuntimeInt = 0, src1_repeat_stride: RuntimeInt = 0,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_GatherMaskParamsType(),
            [
                _mat(src0_block_stride).to_ir(),
                _mat(repeat_times).to_ir(),
                _mat(src0_repeat_stride).to_ir(),
                _mat(src1_repeat_stride).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui8_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui8_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> GatherMaskParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class MmadParams(IRValue):

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, m: int = 0, n: int = 0, k: int = 0, is_bias: bool = False, fm_offset: int = 0,
                 en_ssparse: bool = False, en_winograd_a: bool = False, en_winograd_b: bool = False) -> None:
        ...

    @overload
    def __init__(self, m: int = 0, n: int = 0, k: int = 0, unit_flag: int = 0, cmatrix_source: bool = False,
                 cmatrix_init_val: bool = True) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        ...

    @require_jit
    def __init__(self, m: RuntimeInt = 0, n: RuntimeInt = 0, k: RuntimeInt = 0, is_bias: RuntimeBool = None,
                 fm_offset: RuntimeInt = None, en_ssparse: RuntimeBool = None, en_winograd_a: RuntimeBool = None,
                 en_winograd_b: RuntimeBool = None, unit_flag: RuntimeInt = None, cmatrix_source: RuntimeBool = None,
                 cmatrix_init_val: RuntimeBool = None, handle: IRHandle = None) -> None:

        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()

        if (is_bias is not None and fm_offset is not None and en_ssparse is not None and en_winograd_a is not None
                and en_winograd_b is not None and unit_flag is None and cmatrix_source is None
                and cmatrix_init_val is None):

            args = [
                _mat(m, KT.uint16).to_ir(),
                _mat(n, KT.uint16).to_ir(),
                _mat(k, KT.uint16).to_ir(),
                _mat(is_bias, KT.int1).to_ir(),
                _mat(fm_offset, KT.int32).to_ir(),
                _mat(en_ssparse, KT.int1).to_ir(),
                _mat(en_winograd_a, KT.int1).to_ir(),
                _mat(en_winograd_b, KT.int1).to_ir(),
            ]

            types = builder.get_type_array_attr([
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_i1_type(),
                builder.get_i32_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
            ])

        elif (unit_flag is not None and cmatrix_source is not None and cmatrix_init_val is not None
              and is_bias is None):

            args = [
                _mat(m, KT.uint16).to_ir(),
                _mat(n, KT.uint16).to_ir(),
                _mat(k, KT.uint16).to_ir(),
                _mat(unit_flag, KT.int1).to_ir(),
                _mat(cmatrix_source, KT.int1).to_ir(),
                _mat(cmatrix_init_val, KT.int1).to_ir(),
            ]

            types = builder.get_type_array_attr([
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
            ])

        else:
            args = []
            types = builder.get_type_array_attr([])

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_MmadParamsType(),
            args,
            types,
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "MmadParams":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class LoadImageToLocalParams(IRValue):

    @overload
    def __init__(self, horiz_size: int = 2, vert_size: int = 2, horiz_start_pos: int = 0, vert_start_pos: int = 0,
                 src_horiz_size: int = 2, top_pad_size: int = 0, bot_pad_size: int = 0, left_pad_size: int = 0,
                 right_pad_size: int = 0, sid: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        ...

    @require_jit
    def __init__(self, horiz_size: RuntimeInt = 2, vert_size: RuntimeInt = 2, horiz_start_pos: RuntimeInt = 0,
                 vert_start_pos: RuntimeInt = 0, src_horiz_size: RuntimeInt = 2, top_pad_size: RuntimeInt = 0,
                 bot_pad_size: RuntimeInt = 0, left_pad_size: RuntimeInt = 0, right_pad_size: RuntimeInt = 0,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_LoadImageToLocalParamsType(),
            [
                _mat(horiz_size).to_ir(),
                _mat(vert_size).to_ir(),
                _mat(horiz_start_pos).to_ir(),
                _mat(vert_start_pos).to_ir(),
                _mat(src_horiz_size).to_ir(),
                _mat(top_pad_size).to_ir(),
                _mat(bot_pad_size).to_ir(),
                _mat(left_pad_size).to_ir(),
                _mat(right_pad_size).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_ui16_type()] * 9),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> LoadImageToLocalParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class Nd2NzParams(IRValue):

    @overload
    def __init__(self, nd_num: int, n_value: int, d_value: int, src_nd_matrix_stride: int, src_d_value: int,
                 dst_nz_c0_stride: int, dst_nz_n_stride: int, dst_nz_matrix_stride: int) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, nd_num: RuntimeInt, n_value: RuntimeInt, d_value: RuntimeInt, src_nd_matrix_stride: RuntimeInt,
                 src_d_value: RuntimeInt, dst_nz_c0_stride: RuntimeInt, dst_nz_n_stride: RuntimeInt,
                 dst_nz_matrix_stride: RuntimeInt, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_Nd2NzParamsType(),
            [
                _mat(nd_num).to_ir(),
                _mat(n_value).to_ir(),
                _mat(d_value).to_ir(),
                _mat(src_nd_matrix_stride).to_ir(),
                _mat(src_d_value).to_ir(),
                _mat(dst_nz_c0_stride).to_ir(),
                _mat(dst_nz_n_stride).to_ir(),
                _mat(dst_nz_matrix_stride).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_ui16_type()] * 8),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Nd2NzParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class Nz2NdParamsFull(IRValue):

    @overload
    def __init__(self, nd_num: int, n_value: int, d_value: int, src_nd_matrix_stride: int, src_n_stride: int,
                 dst_d_stride: int, dst_nd_matrix_stride: int) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, nd_num: RuntimeInt, n_value: RuntimeInt, d_value: RuntimeInt, src_nd_matrix_stride: RuntimeInt,
                 src_n_stride: RuntimeInt, dst_d_stride: RuntimeInt, dst_nd_matrix_stride: RuntimeInt,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_Nz2NdParamsFullType(),
            [
                _mat(nd_num).to_ir(),
                _mat(n_value).to_ir(),
                _mat(d_value).to_ir(),
                _mat(src_nd_matrix_stride).to_ir(),
                _mat(src_n_stride).to_ir(),
                _mat(dst_d_stride).to_ir(),
                _mat(dst_nd_matrix_stride).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_ui16_type()] * 7),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Nz2NdParamsFull:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class DataCopyCO12DstParams(IRValue):

    @overload
    def __init__(self, n_size: int, m_size: int, dst_stride: int, src_stride: int,
                 quant_pre: QuantModes = QuantModes.NoQuant, relu_pre: int = 0, channel_split: bool = False,
                 nz2nd_en: bool = False) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, n_size: RuntimeInt, m_size: RuntimeInt, dst_stride: RuntimeInt, src_stride: RuntimeInt,
                 quant_pre: QuantModes = QuantModes.NoQuant, relu_pre: RuntimeInt = 0,
                 channel_split: RuntimeBool = False, nz2nd_en: RuntimeBool = False,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_DataCopyCO12DstParamsType(),
            [
                _mat(n_size).to_ir(),
                _mat(m_size).to_ir(),
                _mat(dst_stride).to_ir(),
                _mat(src_stride).to_ir(),
                _mat(quant_pre).to_ir(),
                _mat(relu_pre).to_ir(),
                _mat(channel_split).to_ir(),
                _mat(nz2nd_en).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui64_type(),
                builder.get_ui64_type(),
                builder.get_ui64_type(),
                builder.get_ui64_type(),
                builder.get_asc_QuantModesType(),
                builder.get_ui8_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> DataCopyCO12DstParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class GatherRepeatParams(IRValue):

    @overload
    def __init__(self, dst_blk_stride: int = 1, dst_rep_stride: int = 8) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        ...

    @require_jit
    def __init__(self, dst_blk_stride: RuntimeInt = 1, dst_rep_stride: RuntimeInt = 8,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_GatherRepeatParamsType(),
            [
                _mat(dst_blk_stride, KnownTypes.uint16).to_ir(),
                _mat(dst_rep_stride, KnownTypes.uint16).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_ui16_type(), builder.get_ui16_type()]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> GatherRepeatParams:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class LoadData2DParams(IRValue):

    @overload
    def __init__(
        self,
        start_index: int = 0,
        repeat_times: int = 1,
        src_stride: int = 0,
        sid: int = 0,
        dst_gap: int = 0,
        if_transpose: bool = False,
        addr_mode: int = 0,
    ) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This constructor should not be called by user"""
        ...

    def __init__(
        self,
        start_index: RuntimeInt = 0,
        repeat_times: RuntimeInt = 1,
        src_stride: RuntimeInt = 0,
        sid: RuntimeInt = 0,
        dst_gap: RuntimeInt = 0,
        if_transpose: RuntimeBool = False,
        addr_mode: RuntimeInt = 0,
        handle: Optional[IRHandle] = None,
    ) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_LoadData2DParamsType(),
            [
                _mat(start_index, KnownTypes.uint16).to_ir(),
                _mat(repeat_times, KnownTypes.uint8).to_ir(),
                _mat(src_stride, KnownTypes.uint16).to_ir(),
                _mat(sid, KnownTypes.uint16).to_ir(),
                _mat(dst_gap, KnownTypes.uint16).to_ir(),
                _mat(if_transpose, KnownTypes.int1).to_ir(),
                _mat(addr_mode, KnownTypes.uint8).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui16_type(),
                builder.get_ui8_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_i1_type(),
                builder.get_ui8_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "LoadData2DParams":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class LoadData2DParamsV2(IRValue):

    @overload
    def __init__(
        self,
        m_start_position: int = 0,
        k_start_position: int = 0,
        m_step: int = 0,
        k_step: int = 0,
        src_stride: int = 0,
        dst_stride: int = 0,
        if_transpose: bool = False,
        sid: int = 0,
    ) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This constructor should not be called by user"""
        ...

    def __init__(
        self,
        m_start_position: RuntimeInt = 0,
        k_start_position: RuntimeInt = 0,
        m_step: RuntimeInt = 0,
        k_step: RuntimeInt = 0,
        src_stride: RuntimeInt = 0,
        dst_stride: RuntimeInt = 0,
        if_transpose: RuntimeBool = False,
        sid: RuntimeInt = 0,
        handle: Optional[IRHandle] = None,
    ) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_LoadData2DParamsV2Type(),
            [
                _mat(m_start_position, KnownTypes.uint32).to_ir(),
                _mat(k_start_position, KnownTypes.uint32).to_ir(),
                _mat(m_step, KnownTypes.uint16).to_ir(),
                _mat(k_step, KnownTypes.uint16).to_ir(),
                _mat(src_stride, KnownTypes.int32).to_ir(),
                _mat(dst_stride, KnownTypes.uint16).to_ir(),
                _mat(if_transpose, KnownTypes.int1).to_ir(),
                _mat(sid, KnownTypes.uint8).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_i32_type(),
                builder.get_ui16_type(),
                builder.get_i1_type(),
                builder.get_ui8_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "LoadData2DParamsV2":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class LoadData3DParamsV1(IRValue):

    @overload
    def __init__(
        self,
        pad_list=(0, 0, 0, 0),
        l1_h: int = 1,
        l1_w: int = 1,
        c1_index: int = 0,
        fetch_filter_w: int = 0,
        fetch_filter_h: int = 0,
        left_top_w: int = 0,
        left_top_h: int = 0,
        stride_w: int = 1,
        stride_h: int = 1,
        filter_w: int = 1,
        filter_h: int = 1,
        dilation_filter_w: int = 1,
        dilation_filter_h: int = 1,
        jump_stride: int = 1,
        repeat_mode: int = 0,
        repeat_time: int = 1,
        c_size: int = 0,
        pad_value: int = 0,
    ) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        ...

    def __init__(
        self,
        pad_list=(0, 0, 0, 0),
        l1_h: RuntimeInt = 1,
        l1_w: RuntimeInt = 1,
        c1_index: RuntimeInt = 0,
        fetch_filter_w: RuntimeInt = 0,
        fetch_filter_h: RuntimeInt = 0,
        left_top_w: RuntimeInt = 0,
        left_top_h: RuntimeInt = 0,
        stride_w: RuntimeInt = 1,
        stride_h: RuntimeInt = 1,
        filter_w: RuntimeInt = 1,
        filter_h: RuntimeInt = 1,
        dilation_filter_w: RuntimeInt = 1,
        dilation_filter_h: RuntimeInt = 1,
        jump_stride: RuntimeInt = 1,
        repeat_mode: RuntimeInt = 0,
        repeat_time: RuntimeInt = 1,
        c_size: RuntimeInt = 0,
        pad_value: RuntimeInt = 0,
        handle: Optional[IRHandle] = None,
    ) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_LoadData3DParamsV1Type(),
            [
                _mat(pad_list[0], KnownTypes.uint8).to_ir(),
                _mat(pad_list[1], KnownTypes.uint8).to_ir(),
                _mat(pad_list[2], KnownTypes.uint8).to_ir(),
                _mat(pad_list[3], KnownTypes.uint8).to_ir(),
                _mat(l1_h, KnownTypes.uint16).to_ir(),
                _mat(l1_w, KnownTypes.uint16).to_ir(),
                _mat(c1_index, KnownTypes.uint16).to_ir(),
                _mat(fetch_filter_w, KnownTypes.uint8).to_ir(),
                _mat(fetch_filter_h, KnownTypes.uint8).to_ir(),
                _mat(left_top_w, KnownTypes.int16).to_ir(),
                _mat(left_top_h, KnownTypes.int16).to_ir(),
                _mat(stride_w, KnownTypes.uint8).to_ir(),
                _mat(stride_h, KnownTypes.uint8).to_ir(),
                _mat(filter_w, KnownTypes.uint8).to_ir(),
                _mat(filter_h, KnownTypes.uint8).to_ir(),
                _mat(dilation_filter_w, KnownTypes.uint8).to_ir(),
                _mat(dilation_filter_h, KnownTypes.uint8).to_ir(),
                _mat(jump_stride, KnownTypes.uint8).to_ir(),
                _mat(repeat_mode, KnownTypes.uint8).to_ir(),
                _mat(repeat_time, KnownTypes.uint8).to_ir(),
                _mat(c_size, KnownTypes.uint8).to_ir(),
                _mat(pad_value, KnownTypes.uint16).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_i16_type(),
                builder.get_i16_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui16_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "LoadData3DParamsV1":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class LoadData3DParamsV2(IRValue):

    @overload
    def __init__(
        self,
        pad_list=(0, 0, 0, 0),
        l1_h: int = 1,
        l1_w: int = 1,
        channel_size: int = 1,
        k_extension: int = 1,
        m_extension: int = 1,
        k_start_pt: int = 0,
        m_start_pt: int = 0,
        stride_w: int = 1,
        stride_h: int = 1,
        filter_w: int = 1,
        filter_h: int = 1,
        dilation_filter_w: int = 1,
        dilation_filter_h: int = 1,
        en_transpose: bool = False,
        pad_value: int = 0,
        filter_size_w: bool = False,
        filter_size_h: bool = False,
        f_matrix_ctrl: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        ...

    def __init__(
        self,
        pad_list=(0, 0, 0, 0),
        l1_h: RuntimeInt = 1,
        l1_w: RuntimeInt = 1,
        channel_size: RuntimeInt = 1,
        k_extension: RuntimeInt = 1,
        m_extension: RuntimeInt = 1,
        k_start_pt: RuntimeInt = 0,
        m_start_pt: RuntimeInt = 0,
        stride_w: RuntimeInt = 1,
        stride_h: RuntimeInt = 1,
        filter_w: RuntimeInt = 1,
        filter_h: RuntimeInt = 1,
        dilation_filter_w: RuntimeInt = 1,
        dilation_filter_h: RuntimeInt = 1,
        en_transpose: RuntimeBool = False,
        pad_value: RuntimeInt = 0,
        filter_size_w: RuntimeBool = False,
        filter_size_h: RuntimeBool = False,
        f_matrix_ctrl: RuntimeBool = False,
        handle: Optional[IRHandle] = None,
    ) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_LoadData3DParamsV2Type(),
            [
                _mat(pad_list[0], KnownTypes.uint8).to_ir(),
                _mat(pad_list[1], KnownTypes.uint8).to_ir(),
                _mat(pad_list[2], KnownTypes.uint8).to_ir(),
                _mat(pad_list[3], KnownTypes.uint8).to_ir(),
                _mat(l1_h, KnownTypes.uint16).to_ir(),
                _mat(l1_w, KnownTypes.uint16).to_ir(),
                _mat(channel_size, KnownTypes.uint8).to_ir(),
                _mat(k_extension, KnownTypes.uint16).to_ir(),
                _mat(m_extension, KnownTypes.uint16).to_ir(),
                _mat(k_start_pt, KnownTypes.uint16).to_ir(),
                _mat(m_start_pt, KnownTypes.uint16).to_ir(),
                _mat(stride_w, KnownTypes.uint8).to_ir(),
                _mat(stride_h, KnownTypes.uint8).to_ir(),
                _mat(filter_w, KnownTypes.uint8).to_ir(),
                _mat(filter_h, KnownTypes.uint8).to_ir(),
                _mat(dilation_filter_w, KnownTypes.uint8).to_ir(),
                _mat(dilation_filter_h, KnownTypes.uint8).to_ir(),
                _mat(en_transpose, KnownTypes.int1).to_ir(),
                _mat(pad_value, KnownTypes.uint16).to_ir(),
                _mat(filter_size_w, KnownTypes.int1).to_ir(),
                _mat(filter_size_h, KnownTypes.int1).to_ir(),
                _mat(f_matrix_ctrl, KnownTypes.int1).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui8_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_ui8_type(),
                builder.get_i1_type(),
                builder.get_ui16_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "LoadData3DParamsV2":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class LoadData3DParamsV2Pro(IRValue):

    @overload
    def __init__(
        self,
        channel_size: int = 0,
        en_transpose: bool = False,
        en_small_k: bool = False,
        filter_size_w: bool = False,
        filter_size_h: bool = False,
        f_matrix_ctrl: bool = False,
        ext_config: int = 0,
        filter_config: int = 0x10101010101,
    ) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This constructor should not be called by user"""
        ...

    def __init__(
        self,
        channel_size: RuntimeInt = 0,
        en_transpose: RuntimeBool = False,
        en_small_k: RuntimeBool = False,
        filter_size_w: RuntimeBool = False,
        filter_size_h: RuntimeBool = False,
        f_matrix_ctrl: RuntimeBool = False,
        ext_config: RuntimeInt = 0,
        filter_config: RuntimeInt = 0x10101010101,
        handle: Optional[IRHandle] = None,
    ) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_LoadData3DParamsV2ProType(),
            [
                _mat(channel_size, KnownTypes.uint16).to_ir(),
                _mat(en_transpose, KnownTypes.int1).to_ir(),
                _mat(en_small_k, KnownTypes.int1).to_ir(),
                _mat(filter_size_w, KnownTypes.int1).to_ir(),
                _mat(filter_size_h, KnownTypes.int1).to_ir(),
                _mat(f_matrix_ctrl, KnownTypes.int1).to_ir(),
                _mat(ext_config, KnownTypes.uint64).to_ir(),
                _mat(filter_config, KnownTypes.uint64).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui16_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_ui64_type(),
                builder.get_ui64_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "LoadData3DParamsV2Pro":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class LoadData2dTransposeParams(IRValue):

    @overload
    def __init__(
        self,
        start_index: int = 0,
        repeat_times: int = 1,
        src_stride: int = 0,
        dst_gap: int = 0,
        dst_frac_gap: int = 0,
        addr_mode: int = 0,
    ) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This constructor should not be called by user"""
        ...

    def __init__(
        self,
        start_index: RuntimeInt = 0,
        repeat_times: RuntimeInt = 1,
        src_stride: RuntimeInt = 0,
        dst_gap: RuntimeInt = 0,
        dst_frac_gap: RuntimeInt = 0,
        addr_mode: RuntimeInt = 0,
        handle: Optional[IRHandle] = None,
    ) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_LoadData2dTransposeParamsType(),
            [
                _mat(start_index, KnownTypes.uint16).to_ir(),
                _mat(repeat_times, KnownTypes.uint8).to_ir(),
                _mat(src_stride, KnownTypes.uint16).to_ir(),
                _mat(dst_gap, KnownTypes.uint16).to_ir(),
                _mat(dst_frac_gap, KnownTypes.uint16).to_ir(),
                _mat(addr_mode, KnownTypes.uint8).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui16_type(),
                builder.get_ui8_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui8_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "LoadData2dTransposeParams":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class LoadDataRepeatParam(IRValue):

    @overload
    def __init__(self, repeat_time: int = 1, repeat_stride: int = 0, repeat_mode: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        ...

    @require_jit
    def __init__(self, repeat_time: RuntimeInt = 1, repeat_stride: RuntimeInt = 0, repeat_mode: RuntimeInt = 0,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_LoadDataRepeatParamType(),
            [
                _mat(repeat_time).to_ir(),
                _mat(repeat_stride).to_ir(),
                _mat(repeat_mode).to_ir(),
            ],
            builder.get_type_array_attr([builder.get_ui8_type(),
                                         builder.get_ui16_type(),
                                         builder.get_ui8_type()]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> 'LoadDataRepeatParam':
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class LoadData2dTransposeParamsV2(IRValue):

    @overload
    def __init__(
        self,
        start_index: int = 0,
        repeat_times: int = 0,
        src_stride: int = 0,
        dst_gap: int = 0,
        dst_frac_gap: int = 0,
        src_frac_gap: int = 0,
        addr_mode: int = 0,
    ) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This constructor should not be called by user"""
        ...

    def __init__(
        self,
        start_index: RuntimeInt = 0,
        repeat_times: RuntimeInt = 0,
        src_stride: RuntimeInt = 0,
        dst_gap: RuntimeInt = 0,
        dst_frac_gap: RuntimeInt = 0,
        src_frac_gap: RuntimeInt = 0,
        addr_mode: RuntimeInt = 0,
        handle: Optional[IRHandle] = None,
    ) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_LoadData2dTransposeParamsV2Type(),
            [
                _mat(start_index, KnownTypes.uint16).to_ir(),
                _mat(repeat_times, KnownTypes.uint8).to_ir(),
                _mat(src_stride, KnownTypes.uint16).to_ir(),
                _mat(dst_gap, KnownTypes.uint16).to_ir(),
                _mat(dst_frac_gap, KnownTypes.uint16).to_ir(),
                _mat(src_frac_gap, KnownTypes.uint16).to_ir(),
                _mat(addr_mode, KnownTypes.uint8).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui16_type(),
                builder.get_ui8_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type(),
                builder.get_ui8_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "LoadData2dTransposeParamsV2":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class MrgSort4Info(IRValue):

    @overload
    def __init__(self, element_lengths: List[int], if_exhausted_suspension: bool = False, valid_bit: int = 15,
                 repeat_times: int = 1) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This constructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, element_lengths: List[int], if_exhausted_suspension: bool = False, valid_bit: int = 15,
                 repeat_times: int = 1, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()

        if_exhausted_suspension_ir = _mat(if_exhausted_suspension, KnownTypes.bool_).to_ir()
        valid_bit_ir = _mat(valid_bit, KnownTypes.uint16).to_ir()
        repeat_times_ir = _mat(repeat_times, KnownTypes.uint16).to_ir()

        from .array import array

        element_lengths_array = array(KnownTypes.uint16, element_lengths)

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_MrgSort4InfoType(),
            [element_lengths_array.to_ir(), if_exhausted_suspension_ir, valid_bit_ir, repeat_times_ir],
            builder.get_type_array_attr([
                element_lengths_array.to_ir().get_type(),
                builder.get_i1_type(),
                builder.get_ui16_type(),
                builder.get_ui16_type()
            ]))

    @classmethod
    def from_ir(cls, handle: IRHandle) -> MrgSort4Info:
        return cls([], False, 15, 1, handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class FixpipeConfig(IRValue):

    @overload
    def __init__(self, layout: CO2Layout) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This constructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, layout: CO2Layout, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(builder.get_asc_FixpipeConfigType(), [_mat(layout).to_ir()],
                                                     builder.get_type_array_attr([builder.get_asc_CO2LayoutType()]),
                                                     isConstexpr=True, isStatic=True)

    @classmethod
    @require_jit
    def cfg_nz(cls) -> FixpipeConfig:
        """constexpr FixpipeConfig CFG_NZ = {CO2Layout::NZ}"""
        return cls(CO2Layout.NZ)

    @classmethod
    @require_jit
    def cfg_row_major(cls) -> FixpipeConfig:
        """constexpr FixpipeConfig CFG_ROW_MAJOR = {CO2Layout::ROW_MAJOR}"""
        return cls(CO2Layout.ROW_MAJOR)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> FixpipeConfig:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class FixpipeParamsV220(IRValue):

    @overload
    def __init__(self, n_size: int, m_size: int, src_stride: int, dst_stride: int,
                 quant_pre: QuantModes = QuantModes.NoQuant, deq_scalar: int = 0, nd_num: int = 1,
                 src_nd_stride: int = 0, dst_nd_stride: int = 0, relu_en: bool = False, unit_flag: int = 0,
                 is_channel_split: bool = False) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This constructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, n_size: RuntimeInt, m_size: RuntimeInt, src_stride: RuntimeInt, dst_stride: RuntimeInt,
                 quant_pre: QuantModes = QuantModes.NoQuant, deq_scalar: RuntimeInt = 0, nd_num: RuntimeInt = 1,
                 src_nd_stride: RuntimeInt = 0, dst_nd_stride: RuntimeInt = 0, relu_en: RuntimeBool = False,
                 unit_flag: RuntimeInt = 0, is_channel_split: RuntimeBool = False,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_FixpipeParamsV220Type(), [
                _mat(n_size, KT.uint16).to_ir(),
                _mat(m_size, KT.uint16).to_ir(),
                _mat(src_stride, KT.uint16).to_ir(),
                _mat(dst_stride, KT.uint32).to_ir(),
                _mat(relu_en).to_ir(),
                _mat(quant_pre).to_ir(),
                _mat(deq_scalar).to_ir(),
                _mat(nd_num, KT.uint16).to_ir(),
                _mat(src_nd_stride, KT.uint16).to_ir(),
                _mat(dst_nd_stride, KT.uint16).to_ir(),
                _mat(unit_flag).to_ir(),
                _mat(is_channel_split).to_ir(),
            ],
            builder.get_type_array_attr([
                builder.get_ui16_type(),  # nSize
                builder.get_ui16_type(),  # mSize
                builder.get_ui16_type(),  # srcStride
                builder.get_ui32_type(),  # dstStride
                builder.get_i1_type(),  # reluEn
                builder.get_asc_QuantModesType(),  # quantPre
                builder.get_ui64_type(),  # deqScalar
                builder.get_ui16_type(),  # ndNum
                builder.get_ui16_type(),  # srcNdStride
                builder.get_ui16_type(),  # dstNdStride
                builder.get_ui8_type(),  # unitFlag
                builder.get_i1_type(),  # isChannelSplit
            ]))

    @classmethod
    def from_ir(cls, handle: IRHandle) -> FixpipeParamsV220:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class VdeqInfo(IRValue):

    @overload
    def __init__(
        self,
        scale: List[float],
        offset: List[int],
        sign_mode: List[bool],
    ) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @require_jit
    def __init__(
        self,
        scale: Optional[List[float]] = None,
        offset: Optional[List[int]] = None,
        sign_mode: Optional[List[bool]] = None,
        handle: Optional[IRHandle] = None,
    ) -> None:
        if handle is not None:
            self.handle = handle
            return

        if scale is None or offset is None or sign_mode is None:
            raise ValueError("VdeqInfo requires scale / offset / sign_mode")
        if len(scale) != 16 or len(offset) != 16 or len(sign_mode) != 16:
            raise ValueError("VdeqInfo expects exactly 16 elements per field")

        builder = global_builder.get_ir_builder()

        from .array import array

        scale_array = array(KnownTypes.float32, scale)
        offset_array = array(KnownTypes.int16, offset)
        sign_mode_array = array(KnownTypes.bit, [1 if x else 0 for x in sign_mode])

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_VdeqInfoType(),
            [
                scale_array.to_ir(),
                offset_array.to_ir(),
                sign_mode_array.to_ir(),
            ],
            builder.get_type_array_attr([
                scale_array.to_ir().get_type(),
                offset_array.to_ir().get_type(),
                sign_mode_array.to_ir().get_type(),
            ]),
        )

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "VdeqInfo":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class CheckLocalMemoryIAParam(IRValue):

    @overload
    def __init__(self, enable_bit: int = 0, start_addr: int = 0, end_addr: int = 0, is_scalar_read: bool = False,
                 is_scalar_write: bool = False, is_vector_read: bool = False, is_vector_write: bool = False,
                 is_mte_read: bool = False, is_mte_write: bool = False, is_enable: bool = False) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This constructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, enable_bit: RuntimeInt = 0, start_addr: RuntimeInt = 0, end_addr: RuntimeInt = 0,
                 is_scalar_read: RuntimeBool = False, is_scalar_write: RuntimeBool = False,
                 is_vector_read: RuntimeBool = False, is_vector_write: RuntimeBool = False,
                 is_mte_read: RuntimeBool = False, is_mte_write: RuntimeBool = False, is_enable: RuntimeBool = False,
                 handle: Optional[IRHandle] = None) -> None:

        if handle is not None:
            self.handle = handle
            return

        builder = global_builder.get_ir_builder()

        enable_bit_ir = _mat(enable_bit, KT.uint8).to_ir()
        start_addr_ir = _mat(start_addr, KT.uint32).to_ir()
        end_addr_ir = _mat(end_addr, KT.uint32).to_ir()
        is_scalar_read_ir = _mat(is_scalar_read, KT.int1).to_ir()
        is_scalar_write_ir = _mat(is_scalar_write, KT.int1).to_ir()
        is_vector_read_ir = _mat(is_vector_read, KT.int1).to_ir()
        is_vector_write_ir = _mat(is_vector_write, KT.int1).to_ir()
        is_mte_read_ir = _mat(is_mte_read, KT.int1).to_ir()
        is_mte_write_ir = _mat(is_mte_write, KT.int1).to_ir()
        is_enable_ir = _mat(is_enable, KT.int1).to_ir()

        self.handle = builder.create_asc_ConstructOp(
            builder.get_asc_CheckLocalMemoryIAParamType(), [
                enable_bit_ir, start_addr_ir, end_addr_ir, is_scalar_read_ir, is_scalar_write_ir, is_vector_read_ir,
                is_vector_write_ir, is_mte_read_ir, is_mte_write_ir, is_enable_ir
            ],
            builder.get_type_array_attr([
                builder.get_ui8_type(),
                builder.get_ui32_type(),
                builder.get_ui32_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
                builder.get_i1_type(),
            ]))

    @classmethod
    def from_ir(cls, handle: IRHandle) -> "CheckLocalMemoryIAParam":
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle
