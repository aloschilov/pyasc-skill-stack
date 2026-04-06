# asc.language.core.LocalTensor.set_addr_with_offset

#### LocalTensor.set_addr_with_offset(src: [LocalTensor](../core.md#asc.language.core.LocalTensor), offset: int) → None

设置带有偏移的Tensor地址。用于快速获取定义一个Tensor，同时指定新Tensor相对于旧Tensor首地址的偏移。偏移的长度为旧Tensor的元素个数。

**对应的Ascend C函数原型**

```c++
template <typename T1>
__aicore__ inline void SetAddrWithOffset(LocalTensor<T1> &src, uint32_t offset)
```

**参数说明**

- src：基础地址的Tensor，将该Tensor的地址作为基础地址，设置偏移后的Tensor地址。
- offset：偏移的长度，单位为元素。

**调用示例**

```python
# 用于快速获取定义一个Tensor，同时指定新Tensor相对于旧Tensor首地址的偏移
# 需要注意，偏移的长度为旧Tensor的元素个数
tmp_buffer = temp_queue.alloc_tensor(asc.float)
tmp_half_buffer.set_addr_with_offset(tmp_buffer, calc_size * 2)
```
