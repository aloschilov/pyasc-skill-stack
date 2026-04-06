# asc.language.core.LocalTensor.get_shape_info

#### LocalTensor.get_shape_info() → ShapeInfo

获取LocalTensor的Shape信息。注意：Shape信息没有默认值，只有通过SetShapeInfo设置过Shape信息后，才可以调用该接口获取正确的Shape信息。

**对应的Ascend C函数原型**

```c++
__aicore__ inline ShapeInfo GetShapeInfo() const
```

**参数说明**

无。

**返回值说明**

LocalTensor的Shape信息，ShapeInfo结构体类型。

**调用示例**

```python
max_shape_info = max_ub.get_shape_info()
```
