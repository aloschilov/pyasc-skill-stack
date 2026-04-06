# asc.language.core.LocalTensor.set_shape_info

#### LocalTensor.set_shape_info(shape_info: ShapeInfo) → None

设置LocalTensor的Shape信息。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetShapeInfo(const ShapeInfo& shapeInfo)
```

**参数说明**

- shape_info：Shape信息，ShapeInfo结构体类型。

**调用示例**

```python
max_ub = softmax_max_buf.get(asc.float)
shape_array = [16, 1024]
max_ub = set_shape_info(asc.ShapeInfo(2, shape_array, asc.DataFormat.ND))
```
