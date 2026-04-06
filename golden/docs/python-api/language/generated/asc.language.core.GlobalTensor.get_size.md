# asc.language.core.GlobalTensor.get_size

#### GlobalTensor.get_size() → int

获取GlobalTensor的元素个数。

**对应的Ascend C函数原型**

```c++
__aicore__ inline uint64_t GetSize() const
```

**参数说明**

无。

**返回值说明**

GlobalTensor的元素个数。

**约束说明**

使用仅传入全局数据指针的set_global_buffer接口对GlobalTensor进行初始化，通过本接口获取到的元素个数为0。
