# asc.language.fwk.TBuf.get

#### TBuf.get(dtype: DataType, shape: TensorShape | None = None) → [LocalTensor](../core.md#asc.language.core.LocalTensor)

#### TBuf.get(dtype: DataType, len: int = None, shape: TensorShape | None = None) → [LocalTensor](../core.md#asc.language.core.LocalTensor)

从TBuf上获取指定长度的Tensor，或者获取全部长度的Tensor。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline LocalTensor<T> Get()
```

```c++
template <typename T>
__aicore__ inline LocalTensor<T> Get(uint32_t len)
```

**参数说明**

- dtype：待获取Tensor的数据类型。
- len：需要获取的Tensor元素个数。

**返回值说明**

获取到的LocalTensor。

**约束说明**

len的数值是Tensor中元素的个数，len\*sizeof(T)不能超过TBuf初始化时的长度。

**调用示例**

```python
# 为TBuf初始化分配内存，分配内存长度为1024字节
pipe = asc.Tpipe()
calc_buf = asc.TBuf(asc.TPosition.VECCALC)
byte_len = 1024
pipe.init_buffer(calc_buf, byte_len)
# 从calc_buf获取Tensor,Tensor为pipe分配的所有内存大小，为1024字节
temp_tensor1 = calc_buf.get(asc.int32)
# 从calc_buf获取Tensor,Tensor为128个int32_t类型元素的内存大小，为512字节
temp_tensor1 = calc_buf.get(asc.int32, 128)
```
