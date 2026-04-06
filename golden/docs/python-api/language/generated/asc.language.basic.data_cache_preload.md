# asc.language.basic.data_cache_preload

### asc.language.basic.data_cache_preload(src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), cache_offset: int) → None

从源地址所在的特定GM地址预加载数据到data cache中。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void DataCachePreload(const GlobalTensor<uint64_t> &src, const T cacheOffset)
```

**参数说明**

- src (asc.GlobalTensor): 源操作数，代表一个 Global Memory 地址空间。
- cache_offset (int): 内存偏移量，表示从 src 的基地址开始，偏移 cache_offset 个字节的位置开始预加载数据。

**返回值**

无。

**约束说明**

频繁调用此接口可能导致保留站拥塞，这种情况下，此指令将被视为NOP指令，阻塞Scalar流水。因此不建议频繁调用该接口。

**调用示例**

```python
weight_gm = asc.GlobalTensor()
weight_gm.set_global_buffer(weight_gm_addr)

asc.data_cache_preload(src=weight_gm, cache_offset=0)
```
