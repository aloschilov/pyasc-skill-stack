# asc.language.basic.duplicate

### asc.language.basic.duplicate(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), scalar: int | float, count: int) → None

### asc.language.basic.duplicate(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), scalar: int | float, mask: int, repeat_times: int, dst_block_stride: int, dst_repeat_stride: int, is_set_mask: bool = True) → None

### asc.language.basic.duplicate(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), scalar: int | float, mask: List[int], repeat_times: int, dst_block_stride: int, dst_repeat_stride: int, is_set_mask: bool = True) → None

将一个变量或立即数复制多次并填充到向量中。

**对应的Ascend C函数原型**

```c++
template <typename T>
void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count)

template <typename T, bool isSetMask = true>
void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)

template <typename T, bool isSetMask = true>
void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
```

**参数说明**

- dst：目的操作数。
- scalar：被复制的源操作数，支持输入变量和立即数，数据类型需与dst中元素的数据类型保持一致。
- count：参与计算的元素个数。
- mask：mask用于控制每次迭代内参与计算的元素。
- repeat_time：矢量计算单元，每次读取连续的8个datablock（每个block32Bytes，共256Bytes）数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeat_time表示迭代的次数。
- dst_block_stride：单次迭代内，矢量目的操作数不同datablock间地址步长。
- dst_repeat_stride：相邻迭代间，矢量目的操作数相同datablock地址步长。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

- tensor高维切分计算样例-mask连续模式
  ```python
  mask = 128
  scalar = 18.0
  asc.duplicate(dst_local, scalar, mask=mask, repeat_times=2, dst_block_stride=1, dst_repeat_stride=8)
  ```
- tensor高维切分计算样例-mask逐bit模式
  ```python
  mask = [uint64_max, uint64_max]
  scalar = 18.0
  asc.duplicate(dst_local, scalar, mask=mask, repeat_times=2, dst_block_stride=1, dst_repeat_stride=8)
  ```
- tensor前n个数据计算样例，源操作数为标量
  ```python
  scalar = 18.0
  asc.duplicate(dst_local, scalar, count=src_data_size)
  ```
