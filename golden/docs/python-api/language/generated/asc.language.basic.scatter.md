# asc.language.basic.scatter

### asc.language.basic.scatter(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), dst_offset: [LocalTensor](../core.md#asc.language.core.LocalTensor), dst_base: int, mask: int, repeat_times: int, src_rep_stride: int) → None

### asc.language.basic.scatter(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), dst_offset: [LocalTensor](../core.md#asc.language.core.LocalTensor), dst_base: int, mask: List[int], repeat_times: int, src_rep_stride: int) → None

### asc.language.basic.scatter(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), dst_offset: [LocalTensor](../core.md#asc.language.core.LocalTensor), dst_base: int, count: int) → None

给定一个连续的输入张量和一个目的地址偏移张量，scatter指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。
将源操作数src中的元素按照指定的位置（由dst_offset和dst_base共同作用）分散到目的操作数dst中。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                              const LocalTensor<uint32_t>& dstOffset,
                              const uint32_t dstBaseAddr, const uint32_t count)
```

```c++
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                const LocalTensor<uint32_t>& dstOffset,
                                const uint32_t dstBaseAddr, const uint64_t mask[],
                                const uint8_t repeatTime, const uint8_t srcRepStride)
```

```c++
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                const LocalTensor<uint32_t>& dstOffset,
                                const uint32_t dstBaseAddr, const uint64_t mask,
                                const uint8_t repeatTime, const uint8_t srcRepStride)
```

**参数说明**

- dst：目的操作数。
- src：源操作数，数据类型需与dst保持一致。
- dst_offset：用于存储源操作数的每个元素在dst中对应的地址偏移,以字节为单位。
  偏移基于dst的基地址dst_base计算，以字节为单位，取值应保证按dst数据类型位宽对齐。
- dst_base：dst的起始偏移地址，单位是字节。取值应保证按dst数据类型位宽对齐。
- count：执行处理的数据个数。
- mask：控制每次迭代内参与计算的元素，支持连续模式或逐bit模式。
- repeat_times：指令迭代次数，每次迭代完成8个datablock的数据收集。
- src_rep_stride：相邻迭代间的地址步长，单位是datablock。

**调用示例**

- tensor高维切分计算样例-mask连续模式
  ```python
  asc.scatter(dst, src, dst_offset, dst_base=0, mask=128, repeat_times=1, src_rep_stride=8)
  ```
- tensor高维切分计算样例-mask逐bit模式
  ```python
  mask_bits = [uint64_max, uint64_max]
  asc.scatter(dst, src, dst_offset, dst_base=0, mask=mask_bits, repeat_times=1, src_rep_stride=8)
  ```
- tensor前n个数据计算样例，源操作数为标量
  ```python
  asc.scatter(dst, src, dst_offset, dst_base=0, count=128)
  ```
