<!-- Copyright (c) 2025 Huawei Technologies Co., Ltd. -->
<!-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of -->
<!-- CANN Open Software License Agreement Version 2.0 (the "License"). -->
<!-- Please refer to the License for details. You may not use this file except in compliance with the License. -->
<!-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, -->
<!-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. -->
<!-- See LICENSE in the root of the software repository for the full text of the License. -->

# asc.language.fwk

## GetTPipePtr

| [`get_tpipe_ptr`](generated/asc.language.fwk.get_tpipe_ptr.md#asc.language.fwk.get_tpipe_ptr)   | 创建TPipe对象时，对象初始化会设置全局唯一的TPipe指针。本接口用于获取该指针，获取该指针后，可进行TPipe相关的操作。   |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|

## TBuf

### *class* asc.language.fwk.TBuf(pos: TPosition)

### *class* asc.language.fwk.TBuf(handle: Value)

使用Ascend C编程的过程中，可能会用到一些临时变量。
这些临时变量占用的内存可以使用TBuf数据结构来管理，存储位置通过模板参数来设置，可以设置为不同的TPosition逻辑位置。
TBuf占用的存储空间通过TPipe进行管理，您可以通过InitBuffer接口为TBuf进行内存初始化操作，之后即可通过Get获取指定长度的Tensor参与计算。

| [`TBuf.get`](generated/asc.language.fwk.TBuf.get.md#asc.language.fwk.TBuf.get)                                     | 从TBuf上获取指定长度的Tensor，或者获取全部长度的Tensor。              |
|--------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| [`TBuf.get_with_offset`](generated/asc.language.fwk.TBuf.get_with_offset.md#asc.language.fwk.TBuf.get_with_offset) | 以TBuf为基地址，向后偏移指定长度，将偏移后的地址作为起始地址，提取长度为指定值的Tensor。 |

## TBufPool

### *class* asc.language.fwk.TBufPool(pos: TPosition | None, buf_id_size: int)

### *class* asc.language.fwk.TBufPool(handle: Value)

TPipe可以管理全局内存资源，而TBufPool可以手动管理或复用Unified Buffer/L1 Buffer物理内存，主要用于多个stage计算中Unified Buffer/L1 Buffer物理内存不足的场景。

| [`TBufPool.init_buf_pool`](generated/asc.language.fwk.TBufPool.init_buf_pool.md#asc.language.fwk.TBufPool.init_buf_pool)   | 通过Tpipe::InitBufPool接口可划分出整块资源，整块TbufPool资源可以继续通过TBufPool::InitBufPool接口划分成小块资源。                                         |
|----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`TBufPool.init_buffer`](generated/asc.language.fwk.TBufPool.init_buffer.md#asc.language.fwk.TBufPool.init_buffer)         | 调用TBufPool::InitBuffer接口为TQue/TBuf进行内存分配。                                                                                |
| [`TBufPool.reset`](generated/asc.language.fwk.TBufPool.reset.md#asc.language.fwk.TBufPool.reset)                           | 在切换TBufPool资源池时使用，结束当前TbufPool资源池正在处理的相关事件。 调用后当前资源池及资源池分配的Buffer仍然存在，只是Buffer内容可能会被改写。 可以切换回该资源池后，重新开始使用该Buffer，无需再次分配。 |

## TPipe

### *class* asc.language.fwk.TPipe

### *class* asc.language.fwk.TPipe(handle: Value)

TPipe用于统一管理Device端内存等资源，一个Kernel函数必须且只能初始化一个TPipe对象。其主要功能包括：

- 内存资源管理：通过TPipe的InitBuffer接口，可以为TQue和TBuf分配内存，分别用于队列的内存初始化和临时变量内存的初始化。
- 同步事件管理：通过TPipe的AllocEventID、ReleaseEventID等接口，可以申请和释放事件ID，用于同步控制。

| [`TPipe.alloc_event_id`](generated/asc.language.fwk.TPipe.alloc_event_id.md#asc.language.fwk.TPipe.alloc_event_id)       | 用于申请HardEvent（硬件类型同步事件）的TEventID，必须与ReleaseEventID搭配使用，调用该接口后，会占用申请的TEventID，直至调用ReleaseEventID释放。   |
|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| [`TPipe.destroy`](generated/asc.language.fwk.TPipe.destroy.md#asc.language.fwk.TPipe.destroy)                            | 释放资源。                                                                                                |
| [`TPipe.fetch_event_id`](generated/asc.language.fwk.TPipe.fetch_event_id.md#asc.language.fwk.TPipe.fetch_event_id)       | 根据HardEvent（硬件类型的同步事件）获取相应可用的TEventID，此接口不会申请TEventID，仅提供可用的TEventID。                                |
| [`TPipe.init`](generated/asc.language.fwk.TPipe.init.md#asc.language.fwk.TPipe.init)                                     | 初始化内存和用于同步流水事件的EventID的初始化。                                                                          |
| [`TPipe.init_buf_pool`](generated/asc.language.fwk.TPipe.init_buf_pool.md#asc.language.fwk.TPipe.init_buf_pool)          | 初始化TBufPool内存资源池。本接口适用于内存资源有限时，希望手动指定UB/L1内存资源复用的场景。本接口初始化后在整体内存资源中划分出一块子资源池。                        |
| [`TPipe.init_buffer`](generated/asc.language.fwk.TPipe.init_buffer.md#asc.language.fwk.TPipe.init_buffer)                | 用于为TQue等队列和TBuf分配内存。                                                                                 |
| [`TPipe.release_event_id`](generated/asc.language.fwk.TPipe.release_event_id.md#asc.language.fwk.TPipe.release_event_id) | 用于释放HardEvent（硬件类型同步事件）的TEventID，通常与AllocEventID搭配使用。                                                |
| [`TPipe.reset`](generated/asc.language.fwk.TPipe.reset.md#asc.language.fwk.TPipe.reset)                                  | 完成资源的释放与eventId等变量的初始化操作，恢复到TPipe的初始化状态。                                                             |

## TQue

### *class* asc.language.fwk.TQue(pos: TPosition = TPosition.VECIN, depth: int = 1)

### *class* asc.language.fwk.TQue(handle: Value)

流水任务之间通过队列（Queue）完成任务间通信和同步。TQue是用来执行队列相关操作、管理相关资源的数据结构。TQue继承自TQueBind父类。

| [`TQue.alloc_tensor`](generated/asc.language.fwk.TQue.alloc_tensor.md#asc.language.fwk.TQue.alloc_tensor)                                  | 从Que中分配Tensor，Tensor所占大小为InitBuffer时设置的每块内存长度。   |
|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [`TQue.deque`](generated/asc.language.fwk.TQue.deque.md#asc.language.fwk.TQue.deque)                                                       | 将Tensor从队列中取出，用于后续处理。                            |
| [`TQue.enque`](generated/asc.language.fwk.TQue.enque.md#asc.language.fwk.TQue.enque)                                                       | 将Tensor push到队列。                                 |
| [`TQue.free_tensor`](generated/asc.language.fwk.TQue.free_tensor.md#asc.language.fwk.TQue.free_tensor)                                     | 释放Que中的指定Tensor。                                 |
| [`TQue.get_tensor_count_in_que`](generated/asc.language.fwk.TQue.get_tensor_count_in_que.md#asc.language.fwk.TQue.get_tensor_count_in_que) | 查询Que中已入队的Tensor数量。                              |
| [`TQue.has_idle_buffer`](generated/asc.language.fwk.TQue.has_idle_buffer.md#asc.language.fwk.TQue.has_idle_buffer)                         | 查询Que中是否有空闲的内存块。                                 |
| [`TQue.has_tensor_in_que`](generated/asc.language.fwk.TQue.has_tensor_in_que.md#asc.language.fwk.TQue.has_tensor_in_que)                   | 查询Que中目前是否已有入队的Tensor。                           |
| [`TQue.vacant_in_que`](generated/asc.language.fwk.TQue.vacant_in_que.md#asc.language.fwk.TQue.vacant_in_que)                               | 查询队列是否已满。                                        |

## TQueBind

### *class* asc.language.fwk.TQueBind(src: TPosition | None = TPosition.VECIN, dst: TPosition | None = TPosition.VECIN, depth: int = 0, mask: int = 0)

### *class* asc.language.fwk.TQueBind(handle: Value)

TQueBind绑定源逻辑位置和目的逻辑位置，根据源位置和目的位置，来确定内存分配的位置 、插入对应的同步事件，帮助开发者解决内存分配和管理、同步等问题。
Tque是TQueBind的简化模式。通常情况下开发者使用TQue进行编程，TQueBind对外提供一些特殊数据通路的内存管理和同步控制，涉及这些通路时可以直接使用TQueBind。

| [`TQueBind.alloc_tensor`](generated/asc.language.fwk.TQueBind.alloc_tensor.md#asc.language.fwk.TQueBind.alloc_tensor)                                  | 从Que中分配Tensor，Tensor所占大小为InitBuffer时设置的每块内存长度。                                                                   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`TQueBind.deque`](generated/asc.language.fwk.TQueBind.deque.md#asc.language.fwk.TQueBind.deque)                                                       | 将Tensor从队列中取出，用于后续处理。                                                                                            |
| [`TQueBind.enque`](generated/asc.language.fwk.TQueBind.enque.md#asc.language.fwk.TQueBind.enque)                                                       | 将Tensor push到队列。                                                                                                 |
| [`TQueBind.free_all_event`](generated/asc.language.fwk.TQueBind.free_all_event.md#asc.language.fwk.TQueBind.free_all_event)                            | 释放队列中申请的所有同步事件。队列分配的Buffer关联着同步事件的eventID，因为同步事件的数量有限制， 如果同时使用的队列Buffer数量超过限制，将无法继续申请队列，使用本接口释放队列中的事件后，可以再次申请队列。 |
| [`TQueBind.free_tensor`](generated/asc.language.fwk.TQueBind.free_tensor.md#asc.language.fwk.TQueBind.free_tensor)                                     | 释放Que中的指定Tensor。                                                                                                 |
| [`TQueBind.get_tensor_count_in_que`](generated/asc.language.fwk.TQueBind.get_tensor_count_in_que.md#asc.language.fwk.TQueBind.get_tensor_count_in_que) | 查询Que中已入队的Tensor数量。                                                                                              |
| [`TQueBind.has_idle_buffer`](generated/asc.language.fwk.TQueBind.has_idle_buffer.md#asc.language.fwk.TQueBind.has_idle_buffer)                         | 查询Que中是否有空闲的内存块。                                                                                                 |
| [`TQueBind.has_tensor_in_que`](generated/asc.language.fwk.TQueBind.has_tensor_in_que.md#asc.language.fwk.TQueBind.has_tensor_in_que)                   | 查询Que中目前是否已有入队的Tensor。                                                                                           |
| [`TQueBind.vacant_in_que`](generated/asc.language.fwk.TQueBind.vacant_in_que.md#asc.language.fwk.TQueBind.vacant_in_que)                               | 查询队列是否已满。                                                                                                        |
