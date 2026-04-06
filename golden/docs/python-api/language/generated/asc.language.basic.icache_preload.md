# asc.language.basic.icache_preload

### asc.language.basic.icache_preload(pre_fetch_len: int) → None

从指令所在DDR地址预加载指令到ICache中。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void ICachePreLoad(const int64_t preFetchLen);
```

**参数说明**

- pre_fetch_len：预取长度。

**返回值说明**

无。

**调用示例**

```python
pre_fetch_len = 2
asc.icache_preload(pre_fetch_len)
```
