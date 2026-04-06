# asc.language.basic.dump_tensor

### asc.language.basic.dump_tensor(tensor: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), desc: int, dump_size: int, shape_info: ShapeInfo | None = None) → None

### asc.language.basic.dump_tensor(tensor: [LocalTensor](../core.md#asc.language.core.LocalTensor), desc: int, dump_size: int, shape_info: ShapeInfo | None = None) → None

基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)
template <typename T>
__aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize)
```

```c++
template <typename T>
__aicore__ inline void DumpTensor(const LocalTensor<T>& tensor, uint32_t desc,
uint32_t dumpSize, const ShapeInfo& shapeInfo)
template <typename T>
__aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc,
uint32_t dumpSize, const ShapeInfo& shapeInfo)
```

**参数说明**

- tensor：需要dump的Tensor。
- desc：用户自定义附加信息（行号或其他自定义数字）。
- dump_size：需要dump的元素个数。
- shape_info：传入Tensor的shape信息，可按照shape信息进行打印。

**约束说明**

- 该功能仅用于NPU上板调试，且仅在如下场景支持：
  - 通过Kernel直调方式调用算子。
  - 通过单算子API调用方式调用算子。
  - 间接调用单算子API(aclnnxxx)接口：Pytorch框架单算子直调的场景。
- 当前仅支持打印存储位置为Unified Buffer/L1 Buffer/L0C Buffer/Global Memory的Tensor信息。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量（包括信息头）不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。

**调用示例**

- 无Tensor shape的打印
  ```python
  asc.dump_tensor(src_local, 5, date_len)
  ```
- 带Tensor shape的打印
  ```python
  shape_info = asc.ShapeInfo()
  asc.dump_tensor(x, 2, 64, shape_info)
  ```
