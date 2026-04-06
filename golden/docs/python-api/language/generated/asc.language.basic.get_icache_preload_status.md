# asc.language.basic.get_icache_preload_status

### asc.language.basic.get_icache_preload_status() → int

获取ICACHE的PreLoad的状态。

**对应的Ascend C函数原型**

```c++
__aicore__ inline int64_t GetICachePreloadStatus();
```

**参数说明**

无。

**返回值说明**

int64_t类型，0表示空闲，1表示忙。

**调用示例**

```python
cache_preload_status = asc.get_icache_preload_status()
```
