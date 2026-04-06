# asc.language.basic.data_copy

### asc.language.basic.data_copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), count: int) → None

### asc.language.basic.data_copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), count: int) → None

### asc.language.basic.data_copy(dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), count: int) → None

### asc.language.basic.data_copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), repeat_params: DataCopyParams) → None

### asc.language.basic.data_copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_params: DataCopyParams) → None

### asc.language.basic.data_copy(dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_params: DataCopyParams) → None

### asc.language.basic.data_copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), intri_params: DataCopyParams, enhanced_params: DataCopyEnhancedParams) → None

### asc.language.basic.data_copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), intri_params: DataCopyParams, enhanced_params: DataCopyEnhancedParams) → None

### asc.language.basic.data_copy(dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), intri_params: DataCopyParams, enhanced_params: DataCopyEnhancedParams) → None

### asc.language.basic.data_copy(dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), slice_list1: list, slice_list2: list, dim_value: int) → None

### asc.language.basic.data_copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), slice_list1: list, slice_list2: list, dim_value: int) → None

### asc.language.basic.data_copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), intri_params: Nd2NzParams) → None

### asc.language.basic.data_copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), intri_params: Nd2NzParams) → None

### asc.language.basic.data_copy(dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), intri_params: Nz2NdParamsFull) → None

### asc.language.basic.data_copy(dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), intri_params: DataCopyCO12DstParams) → None

### asc.language.basic.data_copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), intri_params: DataCopyCO12DstParams) → None

DataCopy系列接口提供全面的数据搬运功能，支持多种数据搬运场景，并可在搬运过程中实现随路格式转换和量化激活等操作。
该接口支持Local Memory与Global Memory之间的数据搬运，以及Local Memory内部的数据搬运。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                const uint32_t count)
```

```c++
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                    const DataCopyParams& repeatParams)
```

```c++
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                const uint32_t count)
```

```c++
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                const DataCopyParams& repeatParams)
```

```c++
template <typename T>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                const uint32_t count)
```

```c++
template <typename T>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                const DataCopyParams& repeatParams)
```

```c++
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                const DataCopyParams& intriParams,
                                const DataCopyEnhancedParams& enhancedParams)
```

```c++
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                const DataCopyParams& intriParams,
                                const DataCopyEnhancedParams& enhancedParams)
```

```c++
template <typename T>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                const DataCopyParams& intriParams,
                                const DataCopyEnhancedParams& enhancedParams)
```

```c++
template <typename T, typename U>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src,
                                const DataCopyParams& intriParams,
                                const DataCopyEnhancedParams& enhancedParams)
```

```c++
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[],
                                const uint32_t dimValue = 1)
```

```c++
template <typename T>
__aicore__ inline void DataCopy(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[],
                                const uint32_t dimValue = 1)
```

**参数说明**

- dst: 目的操作数，类型为LocalTensor或GlobalTensor。
- src：源操作数，类型为LocalTensor或GlobalTensor。
- params：搬运参数，DataCopyParams类型。
- count：参与搬运的元素个数。
- enhanced_params：增强信息参数。
- slice_list1/slice_list2：目的操作数/源操作数切片信息，SliceInfo类型。
- dim_value：操作数维度信息，默认值为1。

**约束说明**

- 如果需要执行多个data_copy指令，且data_copy的目的地址存在重叠，需要通过调用pipe_barrier(ISASI)来插入同步指令，保证多个data_copy指令的串行化，防止出现异常数据。
- 在跨卡通信算子开发场景，data_copy类接口支持跨卡数据搬运，仅支持HCCS物理链路，不支持其他通路；开发者开发过程中，需要关注涉及卡间通信的物理通路，可通过npu-smi info -t topo命令查询HCCS物理链路。

**调用示例**

- 基础数据搬运
  ```python
  pipe = asc.Tpipe()
  in_queue_src = asc.TQue(asc.TPosition.VECIN, 1)
  out_queue_dst = asc.TQue(asc.TPosition.VECOUT, 1)
  src_global = asc.GlobalTensor()
  dst_global = asc.GlobalTensor()
  pipe.init_buffer(que=in_queue_src, num=1, len=512 * asc.half.sizeof())
  pipe.init_buffer(que=out_queue_dst, num=1,len=512 * asc.half.sizeof())
  src_local = in_queue_src.alloc_tensor(asc.half)
  dst_local = out_queue_dst.alloc_tensor(asc.half)
  # 使用传入count参数的搬运接口，完成连续搬运
  asc.data_copy(src_local, src_global, count=512)
  asc.data_copy(dst_local, src_local, count=512)
  asc.data_copy(dst_global, dst_local, count=512)
  # 使用传入DataCopyParams参数的搬运接口，支持连续和非连续搬运
  intri_params = asc.DataCopyParams()
  asc.data_copy(src_local, src_global, params=intri_params)
  asc.data_copy(dst_local, src_local, params=intri_params)
  asc.data_copy(dst_global, dst_local, params=intri_params)
  ```
- 增强数据搬运
  ```python
  pipe = asc.Tpipe()
  in_queue_src = asc.TQue(asc.TPosition.CO1, 1)
  out_queue_dst = asc.TQue(asc.TPosition.CO2, 1)
  ...
  src_local = in_queue_src.alloc_tensor(asc.half)
  dst_local = out_queue_dst.alloc_tensor(asc.half)
  intri_params = asc.DataCopyParams()
  enhanced_params = asc.DataCopyEnhancedParams()
  asc.data_copy(dst_local, src_local, params=intri_params, enhanced_params=enhanced_params)
  ```
