# asc.language.basic.TensorDesc.set_shape_addr

#### TensorDesc.set_shape_addr(shape_ptr: int) → None

配置用于储存shape信息的地址。

**对应的Ascend C函数原型**

```c++
void SetShapeAddr(uint64_t* shapePtr)
```

**参数说明**

- shape_ptr：用于储存shape信息的地址。

**调用示例**

```python
tensor_desc = asc.TensorDesc()
tensor_desc.set_shape_addr(0)
```
