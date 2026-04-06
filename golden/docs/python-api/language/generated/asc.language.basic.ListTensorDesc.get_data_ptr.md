# asc.language.basic.ListTensorDesc.get_data_ptr

#### ListTensorDesc.get_data_ptr(index: int, dtype: DataType) → GlobalAddress

根据index获取储存对应数据的地址。

**对应的Ascend C函数原型**

```c++
template<class T> __aicore__ inline __gm__ T* GetDataPtr(uint32_t index)
```

**参数说明**

- index: 索引值。
- dtype: 输出的指针地址指向的数据的数据类型。

**返回值说明**

储存对应数据的地址。

**调用示例**

```python
x_desc = asc.ListTensorDesc(data=x, length=0xffffffff, shape_size=0xffffffff)
x_ptr = x_desc.get_data_ptr(index=0, dtype=asc.float16)
```
