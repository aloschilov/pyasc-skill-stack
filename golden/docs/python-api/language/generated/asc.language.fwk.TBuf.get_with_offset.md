# asc.language.fwk.TBuf.get_with_offset

#### TBuf.get_with_offset(size: int, buf_offset: int, dtype: DataType) → [LocalTensor](../core.md#asc.language.core.LocalTensor)

以TBuf为基地址，向后偏移指定长度，将偏移后的地址作为起始地址，提取长度为指定值的Tensor。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline LocalTensor<T> GetWithOffset(uint32_t size, uint32_t bufOffset)
```

**参数说明**

- dtype：待获取Tensor的数据类型。
- size：需要获取的Tensor元素个数。
- buf_offset：从起始位置的偏移长度，单位是字节，且需32字节对齐。

**返回值说明**

获取到的LocalTensor。

**约束说明**

- size的数值是Tensor中元素的个数，size\*dtype.sizeof + buf_offset不能超过TBuf初始化时的长度。
- buf_offset需满足32字节对齐的要求。

**调用示例**

```python
# 为TBuf初始化分配内存，分配内存长度为1024字节
pipe = asc.Tpipe()
calc_buf = asc.TBuf(asc.TPosition.VECCALC)
byte_len = 1024
pipe.init_buffer(calc_buf, byte_len)
# 从calc_buf偏移64字节获取Tensor,Tensor为128个int32_t类型元素的内存大小，为512字节
temp_tensor1 = calc_buf.get_with_offset(asc.int32, 128, 64)
```
