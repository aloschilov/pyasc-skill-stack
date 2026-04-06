<!-- Copyright (c) 2025 Huawei Technologies Co., Ltd. -->
<!-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of -->
<!-- CANN Open Software License Agreement Version 2.0 (the "License"). -->
<!-- Please refer to the License for details. You may not use this file except in compliance with the License. -->
<!-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, -->
<!-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. -->
<!-- See LICENSE in the root of the software repository for the full text of the License. -->

# asc.language.core

## GlobalTensor

### *class* asc.language.core.GlobalTensor(handle: Value)

GlobalTensor用来存放Global Memory（外部存储）的全局数据。
GlobalTensor public成员函数如下。类型T支持基础数据类型以及TensorTrait类型，但需要遵循使用此GlobalTensor的指令的数据类型支持情况。

| [`GlobalTensor.get_phy_addr`](generated/asc.language.core.GlobalTensor.get_phy_addr.md#asc.language.core.GlobalTensor.get_phy_addr)                | 获取全局数据的地址。                                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`GlobalTensor.get_shape_info`](generated/asc.language.core.GlobalTensor.get_shape_info.md#asc.language.core.GlobalTensor.get_shape_info)          | 获取GlobalTensor的shape信息。注意：Shape信息没有默认值，只有通过SetShapeInfo设置过Shape信息后，才可以调用该接口获取正确的ShapeInfo。 |
| [`GlobalTensor.get_size`](generated/asc.language.core.GlobalTensor.get_size.md#asc.language.core.GlobalTensor.get_size)                            | 获取GlobalTensor的元素个数。                                                                       |
| [`GlobalTensor.get_value`](generated/asc.language.core.GlobalTensor.get_value.md#asc.language.core.GlobalTensor.get_value)                         | 获取GlobalTensor的相应偏移位置的值。                                                                   |
| [`GlobalTensor.set_global_buffer`](generated/asc.language.core.GlobalTensor.set_global_buffer.md#asc.language.core.GlobalTensor.set_global_buffer) | 传入全局数据地址，初始化GlobalTensor。                                                                  |
| [`GlobalTensor.set_l2_cache_hint`](generated/asc.language.core.GlobalTensor.set_l2_cache_hint.md#asc.language.core.GlobalTensor.set_l2_cache_hint) | 设置GlobalTensor是否使能L2 Cache，默认使能L2 Cache。                                                   |
| [`GlobalTensor.set_shape_info`](generated/asc.language.core.GlobalTensor.set_shape_info.md#asc.language.core.GlobalTensor.set_shape_info)          | 设置GlobalTensor的shape信息。                                                                    |
| [`GlobalTensor.set_value`](generated/asc.language.core.GlobalTensor.set_value.md#asc.language.core.GlobalTensor.set_value)                         | 设置GlobalTensor相应偏移位置的值。                                                                    |

## LocalMemAllocator

### *class* asc.language.core.LocalMemAllocator(hard: Hardware = Hardware.UB)

LocalMemAllocator是在使用Ascend C静态Tensor编程方式时用于内存管理的类，用户无需构建TPipe/TQue，
而是直接创建LocalTensor对象并开发算子，从而减少运行时的开销，实现更优的性能。

LocalMemAllocator仅支持在Ascend C静态Tensor编程方式中使用，不可以与TPipe等接口混用。

| [`LocalMemAllocator.alloc`](generated/asc.language.core.LocalMemAllocator.alloc.md#asc.language.core.LocalMemAllocator.alloc)                      | 根据用户指定的逻辑位置、数据类型、数据长度返回对应的 LocalTensor 对象。   |
|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| [`LocalMemAllocator.get_cur_addr`](generated/asc.language.core.LocalMemAllocator.get_cur_addr.md#asc.language.core.LocalMemAllocator.get_cur_addr) | 返回当前物理位置空闲的起始地址。                             |

## LocalTensor

### *class* asc.language.core.LocalTensor

### *class* asc.language.core.LocalTensor(dtype: DataType)

### *class* asc.language.core.LocalTensor(dtype: DataType, addr: int)

### *class* asc.language.core.LocalTensor(dtype: DataType, pos: TPosition | None = TPosition.VECIN, addr: int = 0, tile_size: int = 0)

### *class* asc.language.core.LocalTensor(handle: Value, dtype: DataType, shape: TensorShape)

LocalTensor用于存放AI Core中Local Memory（内部存储）的数据，支持逻辑位置TPosition为VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2。

| [`LocalTensor.get_length`](generated/asc.language.core.LocalTensor.get_length.md#asc.language.core.LocalTensor.get_length)                               | 获取LocalTensor数据长度。                                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [`LocalTensor.get_phy_addr`](generated/asc.language.core.LocalTensor.get_phy_addr.md#asc.language.core.LocalTensor.get_phy_addr)                         | 返回LocalTensor的地址或指定偏移量后的地址。                                                              |
| [`LocalTensor.get_position`](generated/asc.language.core.LocalTensor.get_position.md#asc.language.core.LocalTensor.get_position)                         | 获取LocalTensor所在的TPosition逻辑位置，支持TPosition为VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2。      |
| [`LocalTensor.get_shape_info`](generated/asc.language.core.LocalTensor.get_shape_info.md#asc.language.core.LocalTensor.get_shape_info)                   | 获取LocalTensor的Shape信息。注意：Shape信息没有默认值，只有通过SetShapeInfo设置过Shape信息后，才可以调用该接口获取正确的Shape信息。  |
| [`LocalTensor.get_size`](generated/asc.language.core.LocalTensor.get_size.md#asc.language.core.LocalTensor.get_size)                                     | 获取当前LocalTensor Size大小。                                                                  |
| [`LocalTensor.get_user_tag`](generated/asc.language.core.LocalTensor.get_user_tag.md#asc.language.core.LocalTensor.get_user_tag)                         | 获取指定Tensor块的Tag信息，用户可以根据Tag信息对Tensor进行不同操作。                                              |
| [`LocalTensor.get_value`](generated/asc.language.core.LocalTensor.get_value.md#asc.language.core.LocalTensor.get_value)                                  | 获取LocalTensor指定索引的数值。 该接口仅在LocalTensor的TPosition为VECIN/VECCALC/VECOUT时支持。                |
| [`LocalTensor.reinterpret_cast`](generated/asc.language.core.LocalTensor.reinterpret_cast.md#asc.language.core.LocalTensor.reinterpret_cast)             | 将当前Tensor重解释为用户指定的新类型，转换后的Tensor与原Tensor地址及内容完全相同，Tensor的大小（字节数）保持不变。                    |
| [`LocalTensor.set_addr_with_offset`](generated/asc.language.core.LocalTensor.set_addr_with_offset.md#asc.language.core.LocalTensor.set_addr_with_offset) | 设置带有偏移的Tensor地址。用于快速获取定义一个Tensor，同时指定新Tensor相对于旧Tensor首地址的偏移。偏移的长度为旧Tensor的元素个数。         |
| [`LocalTensor.set_buffer_len`](generated/asc.language.core.LocalTensor.set_buffer_len.md#asc.language.core.LocalTensor.set_buffer_len)                   | 设置Buffer长度。当用户调用operator[]函数创建新LocalTensor时，建议调用该接口设置新LocalTensor长度，便于编译器对内存及同步进行自动优化。   |
| [`LocalTensor.set_shape_info`](generated/asc.language.core.LocalTensor.set_shape_info.md#asc.language.core.LocalTensor.set_shape_info)                   | 设置LocalTensor的Shape信息。                                                                   |
| [`LocalTensor.set_size`](generated/asc.language.core.LocalTensor.set_size.md#asc.language.core.LocalTensor.set_size)                                     | 设置当前LocalTensor Size大小。单位为元素。当用户重用local tensor变量且使用长度发生变化的时候，需要使用此接口重新设置Size。            |
| [`LocalTensor.set_user_tag`](generated/asc.language.core.LocalTensor.set_user_tag.md#asc.language.core.LocalTensor.set_user_tag)                         | 为Tensor添加用户自定义信息，用户可以根据需要设置对应的Tag。后续可通过GetUserTag获取指定Tensor的Tag信息，并根据Tag信息对Tensor进行相应操作。 |
| [`LocalTensor.set_value`](generated/asc.language.core.LocalTensor.set_value.md#asc.language.core.LocalTensor.set_value)                                  | 设置LocalTensor中的某个值。 该接口仅在LocalTensor的TPosition为VECIN/VECCALC/VECOUT时支持。                  |

## ShapeInfo

### *class* asc.language.core.ShapeInfo

### *class* asc.language.core.ShapeInfo(shape: Array, original_shape: Array | None = None, data_format: DataFormat = DataFormat.ND)

### *class* asc.language.core.ShapeInfo(handle: Value)

ShapeInfo用来存放LocalTensor或GlobalTensor的shape信息。

| [`get_shape_size`](generated/asc.language.core.get_shape_size.md#asc.language.core.get_shape_size)   | 获取Shape中所有dim的累乘结果   |
|------------------------------------------------------------------------------------------------------|----------------------|
