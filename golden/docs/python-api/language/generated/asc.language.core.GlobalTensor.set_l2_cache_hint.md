# asc.language.core.GlobalTensor.set_l2_cache_hint

#### GlobalTensor.set_l2_cache_hint(mode: CacheMode = CacheMode.CACHE_MODE_NORMAL, rw_mode: CacheRwMode = CacheRwMode.RW) → None

设置GlobalTensor是否使能L2 Cache，默认使能L2 Cache。

**对应的Ascend C函数原型**

```c++
template<CacheRwMode rwMode = CacheRwMode::RW>
__aicore__ inline void SetL2CacheHint(CacheMode mode);
```

**参数说明**

- rw_mode：设置L2 Cache读写模式。
- mode：用户指定的L2 Cache模式。

**约束说明**

该接口功能当前仅支持在自定义算子工程中使用，不支持Kernel直调工程。

**调用示例**

```python
data_size = 256
input_global = asc.GlobalTensor()
input_global.set_global_buffer(src_gm, data_size)
input_global.set_l2_cache_hint(mode=asc.CacheMode.CACHE_MODE_DISABLE)
input_local = in_queue_x.alloc_tensor(asc.int32)
asc.data_copy(input_local, input_global, data_size)
```
