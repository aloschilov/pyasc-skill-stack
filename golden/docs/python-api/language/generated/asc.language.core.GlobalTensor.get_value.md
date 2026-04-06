# asc.language.core.GlobalTensor.get_value

#### GlobalTensor.get_value(offset: int) → int

获取GlobalTensor的相应偏移位置的值。

**对应的Ascend C函数原型**

```c++
__aicore__ inline __inout_pipe__(S) PrimType GetValue(const uint64_t offset) const
```

**参数说明**

- offset：偏移offset个元素。

**返回值说明**

返回PrimType类型的立即数。
