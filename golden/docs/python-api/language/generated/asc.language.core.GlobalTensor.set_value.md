# asc.language.core.GlobalTensor.set_value

#### GlobalTensor.set_value(offset: int, value: int | float) → None

设置GlobalTensor相应偏移位置的值。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetValue(const uint64_t offset, PrimType value)
```

**参数说明**

- offset：偏移offset个元素。
- value：设置值。PrimType类型。

**约束说明**

如果get_value的Global Memory地址内容存在被外部改写的可能，需要先调用data_cache_clean_and_invalid，确保Data Cache与Global Memory的Cache一致性，之后再调用此接口。
