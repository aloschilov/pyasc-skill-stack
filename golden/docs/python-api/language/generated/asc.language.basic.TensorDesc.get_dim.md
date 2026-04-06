# asc.language.basic.TensorDesc.get_dim

#### TensorDesc.get_dim() → PlainValue | int

获取Tensor的维度。

**对应的Ascend C函数原型**

```c++
uint64_t GetDim()
```

**返回值说明**

返回Tensor的维度。

**调用示例**

```python
tensor_desc = asc.TensorDesc()
dim = tensor_desc.get_dim()
```
