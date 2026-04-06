# asc.language.basic.TensorDesc.get_shape

#### TensorDesc.get_shape(offset: PlainValue | int) → PlainValue | int

获取对应维度的shape信息。

**对应的Ascend C函数原型**

```c++
uint64_t GetShape(uint32_t offset)
```

**参数说明**

- offset：输入索引值。

**返回值说明**

返回对应维度的shape信息。

**调用示例**

```python
tensor_desc = asc.TensorDesc()
offset = 0
shape = tensor_desc.get_shape
```
