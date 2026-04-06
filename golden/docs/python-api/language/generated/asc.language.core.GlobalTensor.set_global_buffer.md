# asc.language.core.GlobalTensor.set_global_buffer

#### GlobalTensor.set_global_buffer(buffer: GlobalAddress | None = None) → None

#### GlobalTensor.set_global_buffer(buffer: GlobalAddress | None = None, buffer_size: int | None = None) → None

传入全局数据地址，初始化GlobalTensor。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetGlobalBuffer(__gm__ PrimType* buffer, uint64_t bufferSize)
```

```c++
__aicore__ inline void SetGlobalBuffer(__gm__ PrimType* buffer)
```

**参数说明**

- buffer：Host侧传入的全局数据指针。PrimType类型。
- buffer_size：  GlobalTensor所包含的类型为PrimType的数据个数，需自行保证不会超出实际数据的长度。

**调用示例**

```python
data_size = 256
input_global = asc.GlobalTensor()
input_global.set_global_buffer(src_gm, data_size)
input_local = in_queue_x.alloc_tensor(asc.int32)
asc.data_copy(input_local, input_global, data_size)
```
