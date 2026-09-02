# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations
from typing import ClassVar, Optional, overload

from ..._C import ir
from ..core.constexpr import ConstExpr, require_constexpr
from ..core.dtype import DataType, KnownTypes
from ..core.enums import HardEvent, TPosition
from ..core.ir_value import IRHandle, IRValue, PlainValue, RuntimeInt, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.types import TensorShape
from ..core.utils import OverloadDispatcher, require_jit, global_builder
from .utils import set_tpipe_docstring


class TQueBind(IRValue):
    """
    TQueBind绑定源逻辑位置和目的逻辑位置，根据源位置和目的位置，来确定内存分配的位置 、插入对应的同步事件，帮助开发者解决内存分配和管理、同步等问题。
    TQue是TQueBind的简化模式。通常情况下开发者使用TQue进行编程，TQueBind对外提供一些特殊数据通路的内存管理和同步控制，涉及这些通路时可以直接使用TQueBind。
    """

    @overload
    def __init__(self, src: Optional[TPosition] = TPosition.VECIN, dst: Optional[TPosition] = TPosition.VECIN, \
                        depth: int = 0, mask: int = 0) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @set_tpipe_docstring(pipe_name="TQueBind", api_name="construct")
    def __init__(self, src: Optional[TPosition] = TPosition.VECIN, dst: Optional[TPosition] = TPosition.VECIN, \
                    depth: Optional[int] = 0, mask: Optional[int] = 0, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        require_constexpr(src, int, arg_name="src")
        require_constexpr(dst, int, arg_name="dst")
        require_constexpr(depth, int, arg_name="depth")
        src = ConstExpr.unwrap(src)
        dst = ConstExpr.unwrap(dst)
        depth = ConstExpr.unwrap(depth)
        builder = global_builder.get_ir_builder()
        ir_type = builder.get_quebind_type(src, dst, depth)
        self.handle = builder.create_asc_QueBindOp(ir_type)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> TQue:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    @overload
    def alloc_tensor(self, dtype: DataType) -> LocalTensor:
        ...

    @overload
    def alloc_tensor(self, tensor: LocalTensor) -> None:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TQueBind", api_name="alloc_tensor")
    def alloc_tensor(self, *args, **kwargs) -> Optional[LocalTensor]:
        dispatcher = OverloadDispatcher(__name__)

        @dispatcher.register(dtype=DataType)
        def _(dtype: DataType):
            tensor_type = ir.get_local_tensor_type(dtype.to_ir())
            handle = global_builder.get_ir_builder().create_asc_TQueBindAllocTensorOp(tensor_type, self.to_ir())
            return LocalTensor(handle=handle, dtype=dtype, shape=None)

        @dispatcher.register(tensor=LocalTensor)
        def _(tensor: LocalTensor):
            global_builder.get_ir_builder().create_asc_TQueBindAllocTensorInPlaceOp(self.to_ir(), tensor.to_ir())

        return dispatcher(*args, **kwargs)

    @overload
    def deque(self, dtype: DataType) -> LocalTensor:
        ...

    @overload
    def deque(self, tensor: LocalTensor) -> None:
        ...

    @overload
    def deque(self, dtype: DataType, src_user_pos: TPosition, dst_user_pos: TPosition) -> LocalTensor:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TQueBind", api_name="deque")
    def deque(self, *args, **kwargs) -> Optional[LocalTensor]:
        dispatcher = OverloadDispatcher(__name__)
        builder = global_builder.get_ir_builder()

        @dispatcher.register(dtype=DataType)
        def _(dtype: DataType):
            tensor_type = ir.get_local_tensor_type(dtype.to_ir())
            handle = builder.create_asc_TQueBindDequeTensorOp(tensor_type, self.to_ir())
            return LocalTensor(handle=handle, dtype=dtype, shape=None)

        @dispatcher.register(tensor=LocalTensor)
        def _(tensor: LocalTensor):
            builder.create_asc_TQueBindDequeTensorInPlaceOp(self.to_ir(), tensor.to_ir())

        @dispatcher.register(dtype=DataType, src_user_pos=TPosition, dst_user_pos=TPosition)
        def _(dtype: DataType, src_user_pos: TPosition, dst_user_pos: TPosition):
            tensor_type = ir.get_local_tensor_type(dtype.to_ir())
            handle = builder.create_asc_TQueBindDequeTensorPosOp(tensor_type, self.to_ir(), \
                        ir.TPosition.symbolize(src_user_pos), ir.TPosition.symbolize(dst_user_pos))
            return LocalTensor(handle=handle, dtype=dtype, shape=None)

        return dispatcher(*args, **kwargs)

    @require_jit
    @set_tpipe_docstring(pipe_name="TQueBind", api_name="enque")
    def enque(self, *args, **kwargs) -> Optional[LocalTensor]:
        dispatcher = OverloadDispatcher(__name__)
        builder = global_builder.get_ir_builder()

        @dispatcher.register(tensor=LocalTensor)
        def _(tensor: LocalTensor):
            builder.create_asc_TQueBindEnqueTensorOp(self.to_ir(), tensor.to_ir())

        @dispatcher.register(tensor=LocalTensor, src_user_pos=TPosition, dst_user_pos=TPosition)
        def _(tensor: LocalTensor, src_user_pos: TPosition, dst_user_pos: TPosition):
            builder.create_asc_TQueBindEnqueTensorPosOp(self.to_ir(), tensor.to_ir(), \
                    ir.TPosition.symbolize(src_user_pos), ir.TPosition.symbolize(dst_user_pos))

        return dispatcher(*args, **kwargs)

    @require_jit
    @set_tpipe_docstring(pipe_name="TQueBind", api_name="free_all_event")
    def free_all_event(self) -> None:
        builder = global_builder.get_ir_builder()
        builder.create_asc_FreeAllEventOp(self.to_ir())

    @require_jit
    @set_tpipe_docstring(pipe_name="TQueBind", api_name="free_tensor")
    def free_tensor(self, tensor: LocalTensor) -> None:
        global_builder.get_ir_builder().create_asc_TQueBindFreeTensorOp(self.to_ir(), tensor.to_ir())

    @require_jit
    @set_tpipe_docstring(pipe_name="TQueBind", api_name="get_tensor_count_in_que")
    def get_tensor_count_in_que(self) -> RuntimeInt:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TQueBindGetTensorCountInQueOp(builder.get_i32_type(), self.to_ir())
        return PlainValue(handle=handle)

    @require_jit
    @set_tpipe_docstring(pipe_name="TQueBind", api_name="has_idle_buffer")
    def has_idle_buffer(self) -> bool:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TQueBindHasIdleBufferOp(builder.get_i1_type(), self.to_ir())
        return PlainValue(handle=handle)

    @require_jit
    @set_tpipe_docstring(pipe_name="TQueBind", api_name="has_tensor_in_que")
    def has_tensor_in_que(self) -> bool:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TQueBindHasTensorInQueOp(builder.get_i1_type(), self.to_ir())
        return PlainValue(handle=handle)

    @require_jit
    def init_buf_handle(self, buf_pool: TBufPool, index: RuntimeInt, buf_handle: TBufHandle, cur_pool_addr: TBufHandle,
                        len: RuntimeInt) -> None:
        builder = global_builder.get_ir_builder()
        builder.create_asc_TQueBindInitBufHandleOp(self.to_ir(), buf_pool.to_ir(),
                                                   _mat(index, KnownTypes.uint32).to_ir(), buf_handle.to_ir(),
                                                   cur_pool_addr.to_ir(),
                                                   _mat(len, KnownTypes.uint32).to_ir())

    @require_jit
    def init_start_buf_handle(self, start_buf_handle: TBufHandle, num: RuntimeInt, len: RuntimeInt) -> None:
        builder = global_builder.get_ir_builder()
        builder.create_asc_TQueBindInitStartBufHandleOp(self.to_ir(), start_buf_handle.to_ir(),
                                                        _mat(num, KnownTypes.uint8).to_ir(),
                                                        _mat(len, KnownTypes.uint32).to_ir())

    @require_jit
    @set_tpipe_docstring(pipe_name="TQueBind", api_name="vacant_in_que")
    def vacant_in_que(self) -> bool:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TQueBindVacantInQueOp(builder.get_i1_type(), self.to_ir())
        return PlainValue(handle=handle)


class TBuf(TQueBind):
    """
    使用Ascend C编程的过程中，可能会用到一些临时变量。
    这些临时变量占用的内存可以使用TBuf数据结构来管理，存储位置通过模板参数来设置，可以设置为不同的TPosition逻辑位置。
    TBuf占用的存储空间通过TPipe进行管理，您可以通过init_buffer接口为TBuf进行内存初始化操作，之后即可通过get获取指定长度的Tensor参与计算。
    """

    @overload
    def __init__(self, pos: TPosition) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @set_tpipe_docstring(pipe_name="TBuf", api_name="construct")
    def __init__(self, pos: Optional[TPosition] = None, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        super().__init__(pos, pos, 0, 0)
        require_constexpr(pos, int, arg_name="pos")
        pos = ConstExpr.unwrap(pos)
        builder = global_builder.get_ir_builder()
        ir_type = builder.get_buffer_type(pos)
        self.handle = builder.create_asc_TBufOp(ir_type)
        super().__init__(handle=self.handle)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> TBuf:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    @overload
    def get(self, dtype: DataType, shape: Optional[TensorShape] = None) -> LocalTensor:
        ...

    @overload
    def get(self, dtype: DataType, len: int = None, shape: Optional[TensorShape] = None) -> LocalTensor:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TBuf", api_name="get")
    def get(self, dtype: DataType, len: RuntimeInt = None, shape: Optional[TensorShape] = None) -> LocalTensor:
        builder = global_builder.get_ir_builder()
        tensor_type = ir.get_local_tensor_type(dtype.to_ir())

        if len is not None:
            handle = builder.create_asc_TBufGetTensorOp(tensor_type, self.to_ir(), _mat(len, KnownTypes.uint32).to_ir())
        else:
            handle = builder.create_asc_TBufGetTensorOp(tensor_type, self.to_ir())
        return LocalTensor(handle=handle, dtype=dtype, shape=shape)

    @overload
    def get_with_offset(self, size: int, buf_offset: int, dtype: DataType) -> LocalTensor:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TBuf", api_name="get_with_offset")
    def get_with_offset(self, size: RuntimeInt, buf_offset: RuntimeInt, dtype: DataType) -> LocalTensor:
        if buf_offset % 32 != 0:
            raise ValueError("buf_offset must be align to 32B.")

        tensor_type = ir.get_local_tensor_type(dtype.to_ir())
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TBufGetWithOffsetOp(tensor_type, self.to_ir(), \
                     _mat(size, KnownTypes.uint32).to_ir(), _mat(buf_offset, KnownTypes.uint32).to_ir())
        return LocalTensor(handle=handle, dtype=dtype, shape=None)


class TBufHandle(IRValue):

    def __init__(self, handle: IRHandle):
        """This contructor should not be called by user"""
        self.handle = handle
        self.dtype = KnownTypes.uint8

    @classmethod
    def from_ir(cls, handle: IRHandle) -> TBufHandle:
        return TBufHandle(handle)

    def to_ir(self) -> IRHandle:
        return self.handle


class TBufPool(IRValue):
    """
    TPipe可以管理全局内存资源，而TBufPool可以手动管理或复用Unified Buffer/L1 Buffer物理内存，主要用于多个stage计算中Unified Buffer/L1 Buffer物理内存不足的场景。
    """

    @overload
    def __init__(self, pos: Optional[TPosition], buf_id_size: int) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @set_tpipe_docstring(pipe_name="TBufPool", api_name="construct")
    def __init__(self, pos: Optional[TPosition] = None, buf_id_size: RuntimeInt = 4,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        require_constexpr(pos, int, arg_name="pos")
        pos = ConstExpr.unwrap(pos)
        builder = global_builder.get_ir_builder()
        ir_type = builder.get_tbuf_pool_type(pos, buf_id_size)
        self.handle = builder.create_asc_TBufPoolOp(ir_type)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> TBufPool:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    @overload
    def init_buf_pool(self, buf_pool: TBufPool, len: int = 0, share_buf: Optional[TBufPool] = None) -> None:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TBufPool", api_name="init_buf_pool")
    def init_buf_pool(self, buf_pool: TBufPool, len: RuntimeInt = 0, share_buf: Optional[TBufPool] = None) -> None:
        builder = global_builder.get_ir_builder()
        if share_buf is not None:
            builder.create_asc_TBufPoolInitBufPoolOp(builder.get_i1_type(), self.to_ir(), buf_pool.to_ir(),
                                                     _mat(len, KnownTypes.uint32).to_ir(), share_buf.to_ir())
        else:
            builder.create_asc_TBufPoolInitBufPoolOp(builder.get_i1_type(), self.to_ir(), buf_pool.to_ir(),
                                                     _mat(len, KnownTypes.uint32).to_ir())

    @overload
    def init_buffer(self, que: TQue, num: int = 0, len: int = 0) -> None:
        ...

    @overload
    def init_buffer(self, buf: TBuf, len: int = 0) -> None:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TBufPool", api_name="init_buffer")
    def init_buffer(self, *args, **kwargs) -> None:
        dispatcher = OverloadDispatcher(__name__)

        @dispatcher.register(que=TQue, num=RuntimeInt, len=RuntimeInt)
        def _(que: TQue, num: RuntimeInt = 0, len: RuntimeInt = 0):
            global_builder.get_ir_builder().create_asc_TBufPoolInitQueueOp(self.to_ir(), que.to_ir(),
                                                                           _mat(num, KnownTypes.int_).to_ir(),
                                                                           _mat(len, KnownTypes.int_).to_ir())

        @dispatcher.register(buf=TBuf, len=RuntimeInt)
        def _(buf: TBuf, len: RuntimeInt = 0):
            global_builder.get_ir_builder().create_asc_TBufPoolInitBufferOp(self.to_ir(), buf.to_ir(),
                                                                            _mat(len, KnownTypes.int_).to_ir())

        dispatcher(*args, **kwargs)

    @require_jit
    @set_tpipe_docstring(pipe_name="TBufPool", api_name="reset")
    def reset(self) -> None:
        global_builder.get_ir_builder().create_asc_TBufPoolResetOp(self.to_ir())


class TPipe(IRValue):
    """
    TPipe用于统一管理Device端内存等资源，一个Kernel函数必须且只能初始化一个TPipe对象。其主要功能包括：

    - 内存资源管理：通过TPipe的init_buffer接口，可以为TQue和TBuf分配内存，分别用于队列的内存初始化和临时变量内存的初始化。
    - 同步事件管理：通过TPipe的alloc_event_id、release_event_id等接口，可以申请和释放事件ID，用于同步控制。
    """

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @set_tpipe_docstring(pipe_name="TPipe", api_name="construct")
    def __init__(self, handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        self.handle = global_builder.get_ir_builder().create_asc_PipeOp()
        TPipeManager.set(self)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> TPipe:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    @overload
    def alloc_event_id(self, event: HardEvent = HardEvent.V_S) -> int:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TPipe", api_name="alloc_event_id")
    def alloc_event_id(self, event: HardEvent = HardEvent.V_S) -> RuntimeInt:
        return PlainValue(global_builder.get_ir_builder() \
                        .create_asc_TPipeAllocEventIDOp(KnownTypes.int_.to_ir(), self.to_ir(), event))

    @require_jit
    @set_tpipe_docstring(pipe_name="TPipe", api_name="destroy")
    def destroy(self) -> None:
        global_builder.get_ir_builder().create_asc_TPipeTQueBindDestroyOp(self.to_ir())

    @overload
    def fetch_event_id(self, event: HardEvent = HardEvent.V_S) -> int:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TPipe", api_name="fetch_event_id")
    def fetch_event_id(self, event: HardEvent = HardEvent.V_S) -> RuntimeInt:
        return PlainValue(global_builder.get_ir_builder() \
                        .create_asc_TPipeFetchEventIDOp(KnownTypes.int_.to_ir(), self.to_ir(), event))

    @overload
    def get_base_addr(self, logic_pos: Optional[TPosition] = None) -> int:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TPipe", api_name="get_base_addr")
    def get_base_addr(self, logic_pos: Optional[TPosition] = None) -> RuntimeInt:
        require_constexpr(logic_pos, int, arg_name="logic_pos")
        logic_pos = ConstExpr.unwrap(logic_pos)
        builder = global_builder.get_ir_builder()
        return PlainValue(builder.create_asc_TPipeGetBaseAddrOp(builder.get_i32_type(), self.to_ir(), \
                            ir.TPosition.symbolize(logic_pos)))

    @require_jit
    @set_tpipe_docstring(pipe_name="TPipe", api_name="init")
    def init(self) -> None:
        global_builder.get_ir_builder().create_asc_TPipeInitOp(self.to_ir())

    @overload
    def init_buf_pool(self, buf_pool: TBufPool, len: int = 0, share_buf: Optional[TBufPool] = None) -> None:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TPipe", api_name="init_buf_pool")
    def init_buf_pool(self, buf_pool: TBufPool, len: RuntimeInt = 0, share_buf: Optional[TBufPool] = None) -> None:
        builder = global_builder.get_ir_builder()
        if share_buf is not None:
            builder.create_asc_TPipeInitBufPoolOp(builder.get_i1_type(), self.to_ir(), buf_pool.to_ir(),
                                                  _mat(len, KnownTypes.uint32).to_ir(), share_buf.to_ir())
        else:
            builder.create_asc_TPipeInitBufPoolOp(builder.get_i1_type(), self.to_ir(), buf_pool.to_ir(),
                                                  _mat(len, KnownTypes.uint32).to_ir())

    @overload
    def init_buffer(self, que: TQue, num: int = 0, len: int = 0) -> None:
        ...

    @overload
    def init_buffer(self, buf: TBuf, len: int = 0) -> None:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TPipe", api_name="init_buffer")
    def init_buffer(self, *args, **kwargs) -> None:
        dispatcher = OverloadDispatcher(__name__)

        @dispatcher.register(que=TQue, num=RuntimeInt, len=RuntimeInt)
        def _(que: TQue, num: RuntimeInt = 0, len: RuntimeInt = 0):
            global_builder.get_ir_builder().create_asc_TPipeInitQueueOp(self.to_ir(), que.to_ir(),
                                                                        _mat(num, KnownTypes.int_).to_ir(),
                                                                        _mat(len, KnownTypes.int_).to_ir())

        @dispatcher.register(buf=TBuf, len=RuntimeInt)
        def _(buf: TBuf, len: RuntimeInt = 0):
            global_builder.get_ir_builder().create_asc_TPipeInitBufferOp(self.to_ir(), buf.to_ir(),
                                                                         _mat(len, KnownTypes.int_).to_ir())

        dispatcher(*args, **kwargs)

    @overload
    def release_event_id(self, id: int, event: HardEvent = HardEvent.V_S) -> None:
        ...

    @require_jit
    @set_tpipe_docstring(pipe_name="TPipe", api_name="release_event_id")
    def release_event_id(self, id: RuntimeInt, event: HardEvent = HardEvent.V_S) -> None:
        global_builder.get_ir_builder() \
                      .create_asc_TPipeReleaseEventIDOp(self.to_ir(), _mat(id, KnownTypes.int_).to_ir(), event)

    @require_jit
    @set_tpipe_docstring(pipe_name="TPipe", api_name="reset")
    def reset(self) -> None:
        global_builder.get_ir_builder().create_asc_TPipeResetOp(self.to_ir())


class TPipeManager:
    instance: ClassVar[Optional[TPipe]] = None

    @classmethod
    def get(cls) -> TPipe:
        if cls.instance is None:
            raise RuntimeError("TPipe instance is not initialized, use TPipe() to create it")
        return cls.instance

    @classmethod
    def set(cls, pipe: TPipe) -> None:
        if cls.instance is not None:
            raise RuntimeError("TPipe instance is already created, use get_tpipe_ptr() to obtain it")
        cls.instance = pipe
        global_builder.on_teardown(cls.reset)

    @classmethod
    def reset(cls) -> None:
        cls.instance = None


def get_tpipe_ptr() -> TPipe:
    """
    创建TPipe对象时，对象初始化会设置全局唯一的TPipe指针。本接口用于获取该指针，获取该指针后，可进行TPipe相关的操作。

    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline AscendC::TPipe* GetTPipePtr()

    **调用示例**

    .. code-block:: python

        pipe = asc.Tpipe()
        x_gm.set_global_buffer(x, 2048)
        in_queue_x = asc.TQue(asc.TPosition.VECIN, 2)
        get_tpipe_ptr.init_buffer(in_queue_x, 2, 128 * asc.half.sizeof())
    """
    return TPipeManager.get()


class TQue(TQueBind):
    """
    流水任务之间通过队列（Queue）完成任务间通信和同步。TQue是用来执行队列相关操作、管理相关资源的数据结构。TQue继承自TQueBind父类。
    """

    @overload
    def __init__(self, pos: TPosition = TPosition.VECIN, depth: int = 1) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    def __init__(self, pos: Optional[TPosition] = TPosition.VECIN, depth: Optional[int] = None, mask: Optional[int] = 0,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        require_constexpr(pos, int, arg_name="pos")
        require_constexpr(depth, int, arg_name="depth")
        pos = ConstExpr.unwrap(pos)
        depth = ConstExpr.unwrap(depth)
        builder = global_builder.get_ir_builder()
        ir_type = builder.get_queue_type(pos, depth)
        self.handle = builder.create_asc_QueueOp(ir_type)
        super().__init__(handle=self.handle)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> TQue:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle
