# asc.language.basic.ListTensorDesc.get_desc

#### ListTensorDesc.get_desc(desc: [TensorDesc](../basic.md#asc.language.basic.TensorDesc), index: int) → None

根据index获得功能说明图中对应的TensorDesc信息。

**对应的Ascend C函数原型**

```c++
template<class T> void GetDesc(TensorDesc<T>& desc, uint32_t index);
```

**参数说明**

- desc: 出参，解析后的Tensor描述信息。
- index: 索引值。

**调用示例**

```python
x_desc = asc.ListTensorDesc(data=x, length=0xffffffff, shape_size=0xffffffff)
y = asc.TensorDesc()
x_desc.get_desc(y, index=0)
```
