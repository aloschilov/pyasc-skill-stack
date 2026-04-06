# asc.language.basic.get_hccl_context

### asc.language.basic.get_hccl_context(index: int) → GlobalAddress

获取指定Index通信域的context（消息区）地址。

**对应的Ascend C函数原型**

```c++
template <uint32_t index>
__aicore__ inline __gm__ uint8_t* __gm__ GetHcclContext(void)
```

**参数说明**

- index: 模板参数，用来表示要设置的通信域ID，当前只支持2个通信域，index只能为0/1。

**返回值说明**

指定通信域的context（消息区）地址。

**约束说明**

当前最多只支持2个通信域。

**调用示例**

```python
ctx = asc.get_hccl_context(1)
```
