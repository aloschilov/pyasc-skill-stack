# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable, Optional, TypeVar

T = TypeVar("T", bound=Callable)


class TQueBindDocstring:

    def __init__(self) -> None:
        ...

    @staticmethod
    def construct_docstring():
        func_introduction = """
        创建TQueBind对象。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline TQueBind()

        """

        param_list = """
        **参数说明**

        无。
        """

        constraint_list = """
        **约束说明**

        无。
        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, ""]

    @staticmethod
    def alloc_tensor_docstring():
        func_introduction = """
        从Que中分配Tensor，Tensor所占大小为init_buffer时设置的每块内存长度。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <typename T>
            __aicore__ inline LocalTensor<T> AllocTensor()

        .. code-block:: c++

            template <typename T>
            __aicore__ inline void AllocTensor(LocalTensor<T>& tensor)

        """

        param_list = """
        **参数说明**

        - dtype：Tensor的数据类型。
        - tensor：inplace接口需要传入LocalTensor作为内存管理的对象。
        """

        return_list = """
        **返回值说明**

        non-inplace接口返回值为LocalTensor对象，inplace接口没有返回值。
        """

        constraint_list = """
        **约束说明**

        - non-inplace接口分配的Tensor内容可能包含随机值。
        - non-inplace接口，需要将TQueBind的depth模板参数设置为非零值；inplace接口，需要将TQueBind的depth模板参数设置为0。
        """

        py_example = """
        **调用示例**

        - non-inplace接口

          .. code-block:: python

              pipe = asc.Tpipe()
              que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 2)
              num = 4
              len = 1024
              pipe.init_buffer(que=que, num=num, len=len)
              tensor = que.alloc_tensor(asc.half)

        - inplace接口

          .. code-block:: python

              pipe = asc.Tpipe()
              que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 2)
              num = 4
              len = 1024
              pipe.init_buffer(que=que, num=num, len=len)
              que.alloc_tensor(asc.half, tensor)

        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def free_tensor_docstring():
        func_introduction = """
        释放Que中的指定Tensor。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <typename T>
            __aicore__ inline void FreeTensor(LocalTensor<T>& tensor)

        """

        param_list = """
        **参数说明**

        - tensor：待释放的Tensor。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            pipe = asc.Tpipe()
            que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 2)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            tensor = que.alloc_tensor(asc.half)
            que.free_tensor(tensor)

        """

        return [func_introduction, cpp_signature, param_list, "", "", py_example]

    @staticmethod
    def enque_docstring():
        func_introduction = """
        将Tensor push到队列。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <typename T>
            __aicore__ inline bool EnQue(const LocalTensor<T>& tensor)

        """

        param_list = """
        **参数说明**

        - tensor：指定的Tensor
        """

        return_list = """
        **返回值说明**

        - True：表示Tensor加入Queue成功
        - False：表示Queue已满，入队失败
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            pipe = asc.Tpipe()
            que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 2)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            tensor = que.alloc_tensor(asc.half)
            que.enque(tensor)

        """

        return [func_introduction, cpp_signature, param_list, return_list, "", py_example]

    @staticmethod
    def deque_docstring():
        func_introduction = """
        将Tensor从队列中取出，用于后续处理。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <typename T>
            __aicore__ inline LocalTensor<T> DeQue()

        .. code-block:: c++

            template <typename T>
            __aicore__ inline void DeQue(LocalTensor<T>& tensor)

        """

        param_list = """
        **参数说明**

        - dtype：Tensor的数据类型。
        - tensor：inplace接口需要通过出参的方式返回Tensor。
        """

        return_list = """
        **返回值说明**

        non-inplace接口的返回值为从队列中取出的LocalTensor；inplace接口没有返回值。
        """

        constraint_list = """
        **约束说明**

        - 对空队列执行deque是一种异常行为，会在CPU调测时报错。
        - non-inplace接口，需要将TQueBind的depth模板参数设置为非零值；inplace接口，需要将TQueBind的depth模板参数设置为0。
        """

        py_example = """
        **调用示例**

        - non-inplace接口

          .. code-block:: python

              pipe = asc.Tpipe()
              que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 4)
              num = 4
              len = 1024
              pipe.init_buffer(que=que, num=num, len=len)
              tensor1 = que.alloc_tensor(asc.half)
              que.enque(tensor1)
              tensor2 = que.deque(asc.half)

        - inplace接口

          .. code-block:: python

              pipe = asc.Tpipe()
              que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 0)
              num = 4
              len = 1024
              pipe.init_buffer(que=que, num=num, len=len)
              tensor1 = que.alloc_tensor(asc.half)
              que.enque(tensor1)
              que.deque(asc.half, tensor1)
              que.free_tensor(tensor1)

        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def vacant_in_que_docstring():
        func_introduction = """
        查询队列是否已满。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline bool VacantInQue()

        """

        param_list = """
        **参数说明**

        无。
        """

        return_list = """
        **返回值说明**

        - True：表示Queue未满，可以继续enque操作
        - False：表示Queue已满，不可以继续入队
        """

        constraint_list = """
        **约束说明**

        该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            # 根据vacant_in_que判断当前que是否已满，设置当前队列深度为4
            pipe = asc.Tpipe()
            que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 4)
            num = 10
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            tensor1 = que.alloc_tensor(asc.half)
            tensor2 = que.alloc_tensor(asc.half)
            tensor3 = que.alloc_tensor(asc.half)
            tensor4 = que.alloc_tensor(asc.half)
            tensor5 = que.alloc_tensor(asc.half)
            que.enque(tensor1)
            que.enque(tensor2)
            que.enque(tensor3)
            que.enque(tensor4)
            ret = que.vacant_in_que()   # 返回False，继续入队操作将报错

        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def has_tensor_in_que_docstring():
        func_introduction = """
        查询Que中目前是否已有入队的Tensor。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline bool HasTensorInQue()

        """

        param_list = """
        **参数说明**

        无。
        """

        return_list = """
        **返回值说明**

        - True：表示Queue中存在已入队的Tensor
        - False：表示Queue为完全空闲
        """

        constraint_list = """
        **约束说明**

        该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            pipe = asc.Tpipe()
            que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 4)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            ret = que.has_tensor_in_que()

        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def get_tensor_count_in_que_docstring():
        func_introduction = """
        查询Que中已入队的Tensor数量。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline int32_t GetTensorCountInQue()

        """

        param_list = """
        **参数说明**

        无。
        """

        return_list = """
        **返回值说明**

        Que中已入队的Tensor数量。
        """

        constraint_list = """
        **约束说明**

        该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            # 通过get_tensor_count_in_que查询que中已入队的Tensor数量，当前通过alloc_tensor接口分配了内存，并加入que中，num为1。
            pipe = asc.Tpipe()
            que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 4)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            tensor1 = que.alloc_tensor(asc.half)
            que.enque(tensor1)
            num = que.get_tensor_count_in_que()

        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def has_idle_buffer_docstring():
        func_introduction = """
        查询Que中是否有空闲的内存块。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline bool HasIdleBuffer()

        """

        param_list = """
        **参数说明**

        无。
        """

        return_list = """
        **返回值说明**

        - True：表示Queue中存在空闲内存
        - False：表示Queue中不存在空闲内存
        """

        constraint_list = """
        **约束说明**

        该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            # 当前Que中已经分配了4块内存
            pipe = asc.Tpipe()
            que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 4)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            ret = que.has_idle_buffer()     # 没有alloc_tensor的操作，返回值为True
            tensor1 = que.alloc_tensor(asc.half)
            ret = que.has_idle_buffer()     # alloc_tensor了一块内存，返回值True
            tensor2 = que.alloc_tensor(asc.half)
            tensor3 = que.alloc_tensor(asc.half)
            tensor4 = que.alloc_tensor(asc.half)
            ret = que.has_idle_buffer()     # alloc_tensor了四块内存，当前无空闲内存，返回值为False，继续alloc_tensor会报错

        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def free_all_event_docstring():
        func_introduction = """
        释放队列中申请的所有同步事件。队列分配的Buffer关联着同步事件的event_id，因为同步事件的数量有限制，
        如果同时使用的队列Buffer数量超过限制，将无法继续申请队列，使用本接口释放队列中的事件后，可以再次申请队列。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline void FreeAllEvent()

        """

        param_list = """
        **参数说明**

        无。
        """

        constraint_list = """
        **约束说明**

        该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            pipe = asc.Tpipe()
            que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 4)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            tensor1 = que.alloc_tensor(asc.half)
            que.enque(tensor1)
            tensor1 = que.deque(asc.half)
            que.free_tensor(tensor1)
            que.free_all_event()

        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]

    @staticmethod
    def init_buf_handle_docstring():
        func_introduction = """
        为TQue、TBuf对象的内存块进行内存分配操作，包括设置内存块的大小，指向的地址等。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <typename T>
            __aicore__ inline void InitBufHandle(T* bufPool, uint32_t index, TBufHandle bufhandle, uint32_t curPoolAddr, uint32_t len)

        """

        param_list = """
        **参数说明**

        - buf_pool：用户自定义的TBufPool对象。
        - index：需要设置的内存块的偏移下标值，第一块为0，第二块为1，...，依次类推。
        - buf_handle：需要设置的内存块指针，类型为TBufHandle(实际为uint8_t*)。
        - cur_pool_addr：需要设置的内存块的地址。
        - len：需要设置的内存块的大小，单位为bytes。
        """

        constraint_list = """
        **约束说明**

        - TQue、TBuf类继承自TQueBind类，所以TQue、TBuf对象也可使用该接口。
        - 目前只提供给自定义TBufPool初始化TQue、TBuf的内存块时使用。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            src_buf1 = TBuf(asc.TPosition.VECIN)
            src_buf1.init_buf_handle(buf_pool, 0, buf_handle, cur_pool_addr, 1024)

        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]

    @staticmethod
    def init_start_buf_handle_docstring():
        func_introduction = """
        设置TQue/TBuf的起始内存块指针、内存块的个数、每一块内存块的大小。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline void InitStartBufHandle(TBufHandle startBufhandle, uint8_t num, uint32_t len)

        """

        param_list = """
        **参数说明**

        - start_buf_handle：TQue/TBuf的起始内存块指针，数据类型为TBufHandle（实际为uint8_t*）。
        - num：分配内存块的个数。
        - len：每一个内存块的大小，单位为Bytes。
        """

        constraint_list = """
        **约束说明**

        - TQue、TBuf类继承自TQueBind类，所以TQue、TBuf对象也可使用该接口。
        - 目前只提供给自定义TBufPool初始化TQue、TBuf的内存块时使用。
        - 当使用TBuf对象调用该接口时，入参num必须为1。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            src_buf1 = TBuf(asc.TPosition.VECIN)
            src_buf1.init_start_buf_handle(buf_handle, 1, 1024)

        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


class TBufDocstring:

    def __init__(self) -> None:
        ...

    @staticmethod
    def construct_docstring():
        func_introduction = """
        创建TBuf对象时，初始化数据成员。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <TPosition pos = TPosition::LCM>
            __aicore__ inline TBuf();

        """

        param_list = """
        **参数说明**

        - pos：TBuf所在的逻辑位置，取值为VECCALC。
        """

        constraint_list = """
        **约束说明**

        无。
        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, ""]

    @staticmethod
    def get_docstring():
        func_introduction = """
        从TBuf上获取指定长度的Tensor，或者获取全部长度的Tensor。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <typename T>
            __aicore__ inline LocalTensor<T> Get()

        .. code-block:: c++

            template <typename T>
            __aicore__ inline LocalTensor<T> Get(uint32_t len)

        """

        param_list = """
        **参数说明**

        - dtype：待获取Tensor的数据类型。
        - len：需要获取的Tensor元素个数。
        """

        return_list = """
        **返回值说明**

        获取到的LocalTensor。
        """

        constraint_list = """
        **约束说明**

        len的数值是Tensor中元素的个数，len*sizeof(T)不能超过TBuf初始化时的长度。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            # 为TBuf初始化分配内存，分配内存长度为1024字节
            pipe = asc.Tpipe()
            calc_buf = asc.TBuf(asc.TPosition.VECCALC)
            byte_len = 1024
            pipe.init_buffer(calc_buf, byte_len)
            # 从calc_buf获取Tensor,Tensor为pipe分配的所有内存大小，为1024字节
            temp_tensor1 = calc_buf.get(asc.int32)
            # 从calc_buf获取Tensor,Tensor为128个int32_t类型元素的内存大小，为512字节
            temp_tensor1 = calc_buf.get(asc.int32, 128)

        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def get_with_offset_docstring():
        func_introduction = """
        以TBuf为基地址，向后偏移指定长度，将偏移后的地址作为起始地址，提取长度为指定值的Tensor。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <typename T>
            __aicore__ inline LocalTensor<T> GetWithOffset(uint32_t size, uint32_t bufOffset)

        """

        param_list = """
        **参数说明**

        - dtype：待获取Tensor的数据类型。
        - size：需要获取的Tensor元素个数。
        - buf_offset：从起始位置的偏移长度，单位是字节，且需32字节对齐。
        """

        return_list = """
        **返回值说明**

        获取到的LocalTensor。
        """

        constraint_list = """
        **约束说明**

        - size的数值是Tensor中元素的个数，size*dtype.sizeof + buf_offset不能超过TBuf初始化时的长度。
        - buf_offset需满足32字节对齐的要求。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            # 为TBuf初始化分配内存，分配内存长度为1024字节
            pipe = asc.Tpipe()
            calc_buf = asc.TBuf(asc.TPosition.VECCALC)
            byte_len = 1024
            pipe.init_buffer(calc_buf, byte_len)
            # 从calc_buf偏移64字节获取Tensor,Tensor为128个int32_t类型元素的内存大小，为512字节
            temp_tensor1 = calc_buf.get_with_offset(asc.int32, 128, 64)

        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


class TBufPoolDocstring:

    def __init__(self) -> None:
        ...

    @staticmethod
    def construct_docstring():
        func_introduction = """
        创建TBufPool对象时，初始化数据成员。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <TPosition pos, uint32_t bufIDSize = defaultBufIDSize>
            __aicore__ inline TBufPool();

        """

        param_list = """
        **参数说明**

        - pos：TBufPool逻辑位置，可以为VECIN、VECOUT、VECCALC、A1、B1、C1。
        - buf_id_size：TBufPool可分配Buffer数量，默认为4，不超过16。
          对于非共享模式的资源分配，在本TBufPool上再次申请TBufPool时，申请的buf_id_size不能超过原TBufPool剩余可用的Buffer数量；
          对于共享模式的资源分配，在本TBufPool上再次申请TBufPool时，申请的buf_id_size不能超过原TBufPool设置的Buffer数量。
        """

        constraint_list = """
        **约束说明**

        无。
        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, ""]

    @staticmethod
    def init_buf_pool_docstring():
        func_introduction = """
        通过Tpipe::InitBufPool接口可划分出整块资源，整块TbufPool资源可以继续通过TBufPool.init_buf_pool接口划分成小块资源。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <class T>
            __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len)

        .. code-block:: c++

            template <class T, class U>
            __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len, U& shareBuf)

        """

        param_list = """
        **参数说明**

        - size：需要获取的Tensor元素个数。
        - buf_offset：从起始位置的偏移长度，单位是字节，且需32字节对齐。
        """

        return_list = """
        **返回值说明**

        获取到的LocalTensor。
        """

        constraint_list = """
        **约束说明**

        - 新划分的资源池与被复用资源池的物理内存需要一致，两者共享起始地址及长度；
        - 输入长度需要小于等于被复用资源池长度；
        - 其他泛用约束参考TBufPool
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            @asc.jit
            def init(src0_gm: asc.GlobalAddress, src1_gm: asc.GlobalAddress, dst_gm: asc.GlobalAddress):
                src0_global.set_global_buffer(src0_gm);
                src1_global.set_global_buffer(src1_gm);
                dst_global.set_global_buffer(dst_gm);
                pipe.init_buf_pool(tbuf_pool0, 131072);
                tbuf_pool0.init_buffer(que=src_que0, num=1, len=65536); // Total src0
                tbuf_pool0.init_buf_pool(tbuf_pool1, 65536);
                tbuf_pool0.init_buf_pool(tbuf_pool2, 65536, tbuf_pool1);

            @asc.jit
            def Process():
                tbuf_pool1.init_buffer(que=src_que1, num=1, len=32768)
                tbuf_pool1.init_buffer(que=dst_que0, num=1, len=32768)
                copy_in()
                compute()
                copy_out()
                tbuf_pool1.reset()
                tbuf_pool2.init_buffer(src_que2, num=1, len=32768)
                tbuf_pool2.init_buffer(dst_que1, num=1, len=32768)
                copy_in1()
                compute1()
                copy_out1()
                tbuf_pool2.reset()
                tbuf_pool0.reset()
                pipe.reset()
        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def init_buffer_docstring():
        func_introduction = """
        调用TBufPool::InitBuffer接口为TQue/TBuf进行内存分配。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <class T> __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)
            template <TPosition pos> __aicore__ inline bool InitBuffer(TBuf<pos>& buf, uint32_t len)

        """

        param_list = """
        **参数说明**

        - pos：Buffer逻辑位置，可以为VECIN、VECOUT、VECCALC、A1、B1、C1。
        - que：需要分配内存的TQue对象。
        - num：分配内存块的个数。
        - len：每个内存块的大小，单位为Bytes，非32Bytes对齐会自动向上补齐至32Bytes对齐。
        - buf：需要分配内存的TBuf对象。
        - len：为TBuf分配的内存大小，单位为Bytes，非32Bytes对齐会自动向上补齐至32Bytes对齐。
        """

        constraint_list = """
        **约束说明**

        声明TBufPool时，可以通过buf_id_size指定可分配Buffer的最大数量，默认上限为4，最大为16。TQue或TBuf的物理内存需要和TBufPool一致。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            @asc.jit
            def init(src0_gm: asc.GlobalAddress, src1_gm: asc.GlobalAddress, dst_gm: asc.GlobalAddress):
                src0_global.set_global_buffer(src0_gm);
                src1_global.set_global_buffer(src1_gm);
                dst_global.set_global_buffer(dst_gm);
                pipe.init_buf_pool(tbuf_pool0, 131072);
                tbuf_pool0.init_buffer(que=src_que0, num=1, len=65536); // Total src0
                tbuf_pool0.init_buf_pool(tbuf_pool1, 65536);
                tbuf_pool0.init_buf_pool(tbuf_pool2, 65536, tbuf_pool1);

            @asc.jit
            def Process():
                tbuf_pool1.init_buffer(que=src_que1, num=1, len=32768)
                tbuf_pool1.init_buffer(que=dst_que0, num=1, len=32768)
                copy_in()
                compute()
                copy_out()
                tbuf_pool1.reset()
                tbuf_pool2.init_buffer(src_que2, num=1, len=32768)
                tbuf_pool2.init_buffer(dst_que1, num=1, len=32768)
                copy_in1()
                compute1()
                copy_out1()
                tbuf_pool2.reset()
                tbuf_pool0.reset()
                pipe.reset()
        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]

    @staticmethod
    def reset_docstring():
        func_introduction = """
        在切换TBufPool资源池时使用，结束当前TbufPool资源池正在处理的相关事件。
        调用后当前资源池及资源池分配的Buffer仍然存在，只是Buffer内容可能会被改写。
        可以切换回该资源池后，重新开始使用该Buffer，无需再次分配。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline void Reset()

        """

        param_list = """
        **参数说明**

        无。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            @asc.jit
            def init(src0_gm: asc.GlobalAddress, src1_gm: asc.GlobalAddress, dst_gm: asc.GlobalAddress):
                src0_global.set_global_buffer(src0_gm);
                src1_global.set_global_buffer(src1_gm);
                dst_global.set_global_buffer(dst_gm);
                pipe.init_buf_pool(tbuf_pool0, 131072);
                tbuf_pool0.init_buffer(que=src_que0, num=1, len=65536); // Total src0
                tbuf_pool0.init_buf_pool(tbuf_pool1, 65536);
                tbuf_pool0.init_buf_pool(tbuf_pool2, 65536, tbuf_pool1);

            @asc.jit
            def Process():
                tbuf_pool1.init_buffer(que=src_que1, num=1, len=32768)
                tbuf_pool1.init_buffer(que=dst_que0, num=1, len=32768)
                copy_in()
                compute()
                copy_out()
                tbuf_pool1.reset()
                tbuf_pool2.init_buffer(src_que2, num=1, len=32768)
                tbuf_pool2.init_buffer(dst_que1, num=1, len=32768)
                copy_in1()
                compute1()
                copy_out1()
                tbuf_pool2.reset()
                tbuf_pool0.reset()
                pipe.reset()
        """

        return [func_introduction, cpp_signature, param_list, "", "", py_example]


class TPipeDocstring:

    def __init__(self) -> None:
        ...

    @staticmethod
    def construct_docstring():
        func_introduction = """
        构造用来管理内存和同步的TPipe对象。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline TPipe()

        """

        param_list = """
        **参数说明**

        无。
        """

        constraint_list = """
        **约束说明**

        - 避免TPipe在对象内创建和初始化，TPipe在对象内创建时，可能会影响编译器对对象内常量的优化，引起scalar性能劣化，具体原理请参考避免TPipe在对象内创建和初始化。
        - TPipe对象同一时刻全局只能存在一份，同时定义多个TPipe对象，会出现卡死等随机行为。如果需要使用多个TPipe时，请先调用destroy接口释放前一个TPipe。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            pipe = asc.Tpipe()

        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]

    @staticmethod
    def init_docstring():
        func_introduction = """
        初始化内存和用于同步流水事件的EventID的初始化。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline void TPipe::Init()

        """

        param_list = """
        **参数说明**

        无。
        """

        constraint_list = """
        **约束说明**

        重复申请释放tpipe，要与destroy接口成对使用，tpipe如果要重复申请需要先destroy释放后再init。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            class KernelAsin:
                ...
            op = KernelAsin
            pipe_in = asc.Tpipe()
            for index in range(1):
                if index != 0:
                    pipe_in.init()
                op.process()
                pipe_in.Destroy()
            pipe_cast = asc.Tpipe()
            op.init(src_gm, dst_gm, src_size, pipe_cast)
            op.Process()
            pipe_cast.destroy()

        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]

    @staticmethod
    def destroy_docstring():
        func_introduction = """
        释放资源。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline void Destroy()

        """

        param_list = """
        **参数说明**

        无。
        """

        constraint_list = """
        **约束说明**

        用于重复申请释放tpipe，创建tpipe对象后，可调用destroy手动释放资源。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            pipe = asc.Tpipe()
            que = asc.TQue(asc.TPosition.VECOUT, 2)
            num = 2
            len = 128
            pipe.init_buffer(que=que, num=num, len=len)
            pipe.destroy()

        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]

    @staticmethod
    def init_buffer_docstring():
        func_introduction = """
        用于为TQue等队列和TBuf分配内存。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <class T>
            __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)

        .. code-block:: c++

            template <TPosition bufPos>
            __aicore__ inline bool InitBuffer(TBuf<bufPos>& buf, uint32_t len)

        """

        param_list = """
        **参数说明**

        - que：需要分配内存的TQue等对象。
        - num：分配内存块的个数。double buffer功能通过该参数开启：num设置为1，表示不开启double buffer；num设置为2，表示开启double buffer。
        - len：每个内存块的大小，单位为字节。当传入的len不满足32字节对齐时，API内部会自动向上补齐至32字节对齐，后续的数据搬运过程会涉及非对齐处理，具体内容请参考非对齐场景。
        - buf：需要分配内存的TBuf对象。
        - len：为TBuf分配的内存大小，单位为字节。当传入的len不满足32字节对齐时，API内部会自动向上补齐至32字节对齐，后续的数据搬运过程会涉及非对齐处理，具体内容请参考非对齐场景。
        """

        constraint_list = """
        **约束说明**

        - init_buffer申请的内存会在TPipe对象销毁时通过析构函数自动释放，无需手动释放。
        - 如果需要重新分配init_buffer申请的内存，可以调用reset，再调用init_buffer接口。
        - 一个kernel中所有使用的Buffer数量之和不能超过64。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            # 为TQue分配内存，分配内存块数为2，每块大小为128字节
            pipe = asc.Tpipe()
            que = asc.TQue(asc.TPosition.VECOUT, 2)
            num = 2
            len = 128
            pipe.init_buffer(que=que, num=num, len=len)
            # 为TBuf分配内存，分配长度为128字节
            pipe = asc.Tpipe()
            buf = asc.TBuf(asc.TPosition.A1)
            len = 128
            pipe.init_buffer(buf=buf, len=len)

        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]

    @staticmethod
    def reset_docstring():
        func_introduction = """
        完成资源的释放与eventId等变量的初始化操作，恢复到TPipe的初始化状态。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            __aicore__ inline void Reset()

        """

        param_list = """
        **参数说明**

        无。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            # 为TQue分配内存，分配内存块数为2，每块大小为128字节
            pipe = asc.Tpipe()
            que = asc.TQue(asc.TPosition.VECOUT, 1)
            num = 1;
            len = 192 * 1024;
            for i in range(2):
                pipe.init_buffer(que=que, num=num, len=len)
                ...     # process
                pipe.reset()

        """

        return [func_introduction, cpp_signature, param_list, "", "", py_example]

    @staticmethod
    def alloc_event_id_docstring():
        func_introduction = """
        用于申请HardEvent（硬件类型同步事件）的TEventID，必须与ReleaseEventID搭配使用，调用该接口后，会占用申请的TEventID，直至调用ReleaseEventID释放。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <HardEvent evt>
            __aicore__ inline TEventID TPipe::AllocEventID()

        """

        param_list = """
        **参数说明**

        - event：HardEvent硬件同步类型。
        """

        return_list = """
        **返回值说明**

        TEventID
        """

        constraint_list = """
        **约束说明**

        TEventID有数量限制，使用结束后应该立刻调用release_event_id释放，防止TEventID耗尽。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            event_id = asc.get_tpipe_ptr().alloc_event_id(asc.HardEvent.V_S)
            asc.set_flag(asc.HardEvent.V_S, event_id)
            ...
            asc.wait_flag(asc.HardEvent.V_S, event_id)
            asc.get_tpipe_ptr().release_event_id(event_id, asc.HardEvent.V_S)

        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def release_event_id_docstring():
        func_introduction = """
        用于释放HardEvent（硬件类型同步事件）的TEventID，通常与AllocEventID搭配使用。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <HardEvent evt>
            __aicore__ inline void ReleaseEventID(TEventID id)

        """

        param_list = """
        **参数说明**

        - event：HardEvent硬件同步类型。
        - id：TEventID类型，调用alloc_event_id申请获得的TEventID。
        """

        constraint_list = """
        **约束说明**

        alloc_event_id、release_event_id需成对出现，release_event_id传入的TEventID需由对应的alloc_event_id申请而来。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            event_id = asc.get_tpipe_ptr().alloc_event_id(asc.HardEvent.V_S)
            asc.set_flag(asc.HardEvent.V_S, event_id)
            ...
            asc.wait_flag(asc.HardEvent.V_S, event_id)
            asc.get_tpipe_ptr().release_event_id(event_id, asc.HardEvent.V_S)

        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]

    @staticmethod
    def fetch_event_id_docstring():
        func_introduction = """
        根据HardEvent（硬件类型的同步事件）获取相应可用的TEventID，此接口不会申请TEventID，仅提供可用的TEventID。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <HardEvent evt>
            __aicore__ inline TEventID TPipe::FetchEventID()
            __aicore__ inline TEventID TPipe::FetchEventID(HardEvent evt)

        """

        param_list = """
        **参数说明**

        - event：HardEvent硬件同步类型。
        """

        return_list = """
        **返回值说明**

        TEventID。
        """

        constraint_list = """
        **约束说明**

        相比于alloc_event_id，fetch_event_id适用于临时使用ID的场景，获取ID后，不会对ID进行占用。在一些复杂的使用场景下，需要开发者自行保证使用正确。
        比如相同流水连续调用set_flag/wait_flag，如果两次传入的ID都是使用fetch_event_id获取的，因为两者ID相同会出现程序卡死等未定义行为，这时推荐用户使用alloc_event_id。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            event_id = asc.get_tpipe_ptr().fetch_event_id(asc.HardEvent.V_S)
            asc.set_flag(asc.HardEvent.V_S, event_id)
            asc.wait_flag(asc.HardEvent.V_S, event_id)

        """

        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def get_base_addr_docstring():
        func_introduction = """
        根据传入的logicPos（逻辑抽象位置），获取该位置的基础地址，只在CPU调试场景下此接口生效。
        通常用于计算Tensor在logicPos的偏移地址即Tensor地址减去GetBaseAddr返回值。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            inline uint8_t* GetBaseAddr(int8_t logicPos)

        """

        param_list = """
        **参数说明**

        - logic_pos：逻辑位置类型。
        """

        return_list = """
        **返回值说明**

        逻辑位置对应的基地址。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            abs_addr = asc.get_tpipe_ptr().get_base_addr(pos);

        """

        return [func_introduction, cpp_signature, param_list, return_list, "", py_example]

    @staticmethod
    def init_buf_pool_docstring():
        func_introduction = """
        初始化TBufPool内存资源池。本接口适用于内存资源有限时，希望手动指定UB/L1内存资源复用的场景。本接口初始化后在整体内存资源中划分出一块子资源池。
        """

        cpp_signature = """
        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <class T>
            __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len)
            template <class T, class U>
            __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len, U& shareBuf)

        """

        param_list = """
        **参数说明**

        - buf_pool：新划分的资源池，类型为TBufPool。
        - len：新划分资源池长度，单位为Byte，非32Bytes对齐会自动补齐至32Bytes对齐。
        - share_buf：被复用资源池，类型为TBufPool，新划分资源池与被复用资源池共享起始地址及长度。
        """

        constraint_list = """
        **约束说明**

        - 新划分的资源池与被复用资源池的硬件属性需要一致，两者共享起始地址及长度；
        - 输入长度需要小于等于被复用资源池长度；
        - 其他泛用约束参考TBufPool。
        """

        py_example = """
        **调用示例**

        .. code-block:: python

            src0_global.set_global_buffer(src0_gm)
            src1_global.set_global_buffer(src1_gm)
            dst_global.set_global_buffer(dst_gm)
            pipe.init_buf_pool(tbuf_pool1, 196608)
            pipe.init_buf_pool(tbuf_pool2, 196608, tbuf_pool1)

        """

        return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


class TQueDocstring:

    def __init__(self) -> None:
        ...

    @staticmethod
    def alloc_tensor_docstring():
        func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = \
            TQueBindDocstring.alloc_tensor_docstring()
        py_example = """
        **调用示例**

        - non-inplace接口

        .. code-block:: python

            pipe = asc.Tpipe()
            que = asc.TQue(asc.TPosition.VECOUT, 2)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            tensor = que.alloc_tensor(asc.half)

        - inplace接口

        .. code-block:: python

            pipe = asc.Tpipe()
            que = asc.TQue(asc.TPosition.VECOUT, 2)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            que.alloc_tensor(asc.half, tensor)

        """
        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def free_tensor_docstring():
        func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = \
            TQueBindDocstring.free_tensor_docstring()
        py_example = """
        **调用示例**

        .. code-block:: python

            pipe = asc.Tpipe()
            que = asc.TQue(asc.TPosition.VECOUT, 2)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            tensor = que.alloc_tensor(asc.half)
            que.free_tensor(tensor)

        """
        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def enque_docstring():
        func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = \
            TQueBindDocstring.enque_docstring()
        py_example = """
        **调用示例**

        .. code-block:: python

            pipe = asc.Tpipe()
            que = asc.TQue(asc.TPosition.VECOUT, 2)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            tensor = que.alloc_tensor(asc.half)
            que.enque(tensor)

        """
        return [func_introduction, cpp_signature, param_list, return_list, py_example]

    @staticmethod
    def deque_docstring():
        func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = \
            TQueBindDocstring.deque_docstring()
        py_example = """
        **调用示例**

        - non-inplace接口

          .. code-block:: python

              pipe = asc.Tpipe()
              que = asc.TQue(asc.TPosition.VECOUT, 4)
              num = 4
              len = 1024
              pipe.init_buffer(que=que, num=num, len=len)
              tensor1 = que.alloc_tensor(asc.half)
              que.enque(tensor1)
              tensor2 = que.deque(asc.half)

        - inplace接口

          .. code-block:: python

              pipe = asc.Tpipe()
              que = asc.TQue(asc.TPosition.VECOUT, 0)
              num = 4
              len = 1024
              pipe.init_buffer(que=que, num=num, len=len)
              tensor1 = que.alloc_tensor(asc.half)
              que.enque(tensor1)
              que.deque(asc.half, tensor1)
              que.free_tensor(tensor1)

        """
        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def vacant_in_que_docstring():
        func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = \
            TQueBindDocstring.vacant_in_que_docstring()
        py_example = """
        **调用示例**

        .. code-block:: python

            # 根据VacantInQue判断当前que是否已满，设置当前队列深度为4
            pipe = asc.Tpipe()
            que = asc.TQue(asc.TPosition.VECOUT, 4)
            num = 10
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            tensor1 = que.alloc_tensor(asc.half)
            tensor2 = que.alloc_tensor(asc.half)
            tensor3 = que.alloc_tensor(asc.half)
            tensor4 = que.alloc_tensor(asc.half)
            tensor5 = que.alloc_tensor(asc.half)
            que.enque(tensor1)
            que.enque(tensor2)
            que.enque(tensor3)
            que.enque(tensor4)
            ret = que.vacant_in_que()   # 返回False，继续入队操作将报错

        """
        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def has_tensor_in_que_docstring():
        func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = \
            TQueBindDocstring.has_tensor_in_que_docstring()
        py_example = """
        **调用示例**

        .. code-block:: python

            pipe = asc.Tpipe()
            que = asc.TQue(asc.TPosition.VECOUT, 4)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            ret = que.has_tensor_in_que()

        """
        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def get_tensor_count_in_que_docstring():
        func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = \
            TQueBindDocstring.get_tensor_count_in_que_docstring()
        py_example = """
        **调用示例**

        .. code-block:: python

            # 通过GetTensorCountInQue查询que中已入队的Tensor数量，当前通过AllocTensor接口分配了内存，并加入que中，num为1。
            pipe = asc.Tpipe()
            que = asc.TQue(asc.TPosition.VECOUT 4)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            tensor1 = que.alloc_tensor(asc.half)
            que.enque(tensor1)
            num = que.get_tensor_count_in_que()

        """
        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]

    @staticmethod
    def has_idle_buffer_docstring():
        func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = \
            TQueBindDocstring.has_idle_buffer_docstring()
        py_example = """
        **调用示例**

        .. code-block:: python

            # 当前Que中已经分配了4块内存
            pipe = asc.Tpipe()
            que = asc.TQueBind(asc.TPosition.VECOUT, 1)
            num = 4
            len = 1024
            pipe.init_buffer(que=que, num=num, len=len)
            ret = que.has_idle_buffer() # 没有alloc_tensor的操作，返回值为True
            tensor1 = que.alloc_tensor(asc.half)
            ret = que.has_idle_buffer() # alloc_tensor了一块内存，返回值True
            tensor2 = que.alloc_tensor(asc.half)
            tensor3 = que.alloc_tensor(asc.half)
            tensor4 = que.alloc_tensor(asc.half)
            ret = que.has_idle_buffer() # alloc_tensor了四块内存，当前无空闲内存，返回值为False，继续alloc_tensor会报错

        """
        return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


DOC_HANDLERS = {
    "TQueBind": {
        "construct": TQueBindDocstring.construct_docstring,
        "alloc_tensor": TQueBindDocstring.alloc_tensor_docstring,
        "free_tensor": TQueBindDocstring.free_tensor_docstring,
        "enque": TQueBindDocstring.enque_docstring,
        "deque": TQueBindDocstring.deque_docstring,
        "vacant_in_que": TQueBindDocstring.vacant_in_que_docstring,
        "has_tensor_in_que": TQueBindDocstring.has_tensor_in_que_docstring,
        "get_tensor_count_in_que": TQueBindDocstring.get_tensor_count_in_que_docstring,
        "has_idle_buffer": TQueBindDocstring.has_idle_buffer_docstring,
        "free_all_event": TQueBindDocstring.free_all_event_docstring,
        "init_buf_handle": TQueBindDocstring.init_buf_handle_docstring,
        "init_start_buf_handle": TQueBindDocstring.init_start_buf_handle_docstring,
    },
    "TBuf": {
        "construct": TBufDocstring.construct_docstring,
        "get": TBufDocstring.get_docstring,
        "get_with_offset": TBufDocstring.get_with_offset_docstring,
    },
    "TBufPool": {
        "construct": TBufPoolDocstring.construct_docstring,
        "init_buf_pool": TBufPoolDocstring.init_buf_pool_docstring,
        "init_buffer": TBufPoolDocstring.init_buffer_docstring,
        "reset": TBufPoolDocstring.reset_docstring,
    },
    "TPipe": {
        "construct": TPipeDocstring.construct_docstring,
        "init": TPipeDocstring.init_docstring,
        "destroy": TPipeDocstring.destroy_docstring,
        "init_buffer": TPipeDocstring.init_buffer_docstring,
        "reset": TPipeDocstring.reset_docstring,
        "alloc_event_id": TPipeDocstring.alloc_event_id_docstring,
        "release_event_id": TPipeDocstring.release_event_id_docstring,
        "fetch_event_id": TPipeDocstring.fetch_event_id_docstring,
        "get_base_addr": TPipeDocstring.get_base_addr_docstring,
        "init_buf_pool": TPipeDocstring.init_buf_pool_docstring,
    },
    "TQue": {
        "alloc_tensor": TQueDocstring.alloc_tensor_docstring,
        "free_tensor": TQueDocstring.free_tensor_docstring,
        "enque": TQueDocstring.enque_docstring,
        "deque": TQueDocstring.deque_docstring,
        "vacant_in_que": TQueDocstring.vacant_in_que_docstring,
        "has_tensor_in_que": TQueDocstring.has_tensor_in_que_docstring,
        "get_tensor_count_in_que": TQueDocstring.get_tensor_count_in_que_docstring,
        "has_idle_buffer": TQueDocstring.has_idle_buffer_docstring,
    },
}


def set_tpipe_docstring(pipe_name: Optional[str] = None, api_name: Optional[str] = None) -> Callable[[T], T]:
    func_introduction = ""
    cpp_signature = ""
    param_list = ""
    return_list = ""
    constraint_list = ""
    py_example = ""

    if DOC_HANDLERS.get(pipe_name) is None:
        raise RuntimeError(f"Invalid pipe name {pipe_name}")
    if DOC_HANDLERS.get(pipe_name, {}).get(api_name) is None:
        raise RuntimeError(f"Unsupported API [{api_name}] for pie type [{pipe_name}]")

    handler = DOC_HANDLERS.get(pipe_name, {}).get(api_name)
    func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = handler()

    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {param_list}
    {return_list}
    {constraint_list}
    {py_example}
    """

    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator
