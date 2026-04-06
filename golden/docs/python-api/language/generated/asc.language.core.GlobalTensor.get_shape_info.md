# asc.language.core.GlobalTensor.get_shape_info

#### GlobalTensor.get_shape_info() → ShapeInfo

获取GlobalTensor的shape信息。注意：Shape信息没有默认值，只有通过SetShapeInfo设置过Shape信息后，才可以调用该接口获取正确的ShapeInfo。

**对应的Ascend C函数原型**

```c++
__aicore__ inline ShapeInfo GetShapeInfo() const
```

**参数说明**

无。

**返回值说明**

GlobalTensor的shape信息，ShapeInfo类型。
