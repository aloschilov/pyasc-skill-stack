# asc.language.core.get_shape_size

### asc.language.core.get_shape_size(shape_info: [ShapeInfo](../core.md#asc.language.core.ShapeInfo)) → int

获取Shape中所有dim的累乘结果

**对应的Ascend C函数原型**

```c++
__aicore__ inline int GetShapeSize(const ShapeInfo& shapeInfo)
```

**参数说明**

- shape_info：ShapeInfo类型，LocalTensor或GlobalTensor的shape信息。

**返回值说明**

输入的shape_info中所有dim的累乘结果。

**约束说明**

无。

**调用示例**

```python
size = asc.get_shape_size(shape_info)
```
