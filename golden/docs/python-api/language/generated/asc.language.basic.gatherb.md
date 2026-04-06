# asc.language.basic.gatherb

### asc.language.basic.gatherb(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), offset: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_times: int, repeat_params: GatherRepeatParams) → None

给定一个输入的张量和一个地址偏移张量，本接口根据偏移地址按照data_block的粒度将输入张量收集到结果张量中。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void Gatherb(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<uint32_t>& offset, const uint8_t repeatTime, const GatherRepeatParams& repeatParams)
```

**参数说明**

- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- offset：每个datablock在源操作数中对应的地址偏移。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
  该偏移量是相对于src0的基地址而言的。每个元素值要大于等于0，单位为字节；且需要保证偏移后的地址满足32字节对齐。
- repeat_time：重复迭代次数，每次迭代完成8个data_block的数据收集，数据范围：repeat_time∈（0,255]。
- dst_rep_stride：目的操作数相邻迭代间的地址步长。以一个repeat_time归约后的长度为单位。每个repeat_time(8个data_block)归约后，得到8个元素，所以输入类型为half类型时，rep_stride单位为16Byte；输入类型为float类型时，rep_stride单位为32Byte。

**约束说明**

无。

**调用示例**

```python
params = asc.GatherRepeatParams(1, 8)
asc.gatherb(y_buf, x_buf, offset_buf, repeat_time=2, repeat_params=params)
```
