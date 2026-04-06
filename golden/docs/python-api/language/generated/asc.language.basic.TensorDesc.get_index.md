# asc.language.basic.TensorDesc.get_index

#### TensorDesc.get_index() → PlainValue | int

获取TensorDesc在ListTensorDesc中对应的索引值。

**对应的Ascend C函数原型**

```c++
uint64_t GetIndex()
```

**返回值说明**

返回TensorDesc在ListTensorDesc中对应的索引值。

**调用示例**

```python
tensor_desc = asc.TensorDesc()
index = tensor_desc.get_index()
```
