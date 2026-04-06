# asc.language.basic.set_hccl_context

### asc.language.basic.set_hccl_context(index: int, context: GlobalAddress) → None

设置通算融合算子每个通信域对应的context（消息区）地址。

**对应的Ascend C函数原型**

```c++
template <uint32_t index>
__aicore__ inline void SetHcclContext(__gm__ uint8_t* context)
```

**参数说明**

- index: 模板参数，用来表示要设置的通信域ID，当前只支持2个通信域，index只能为0/1。
- context: 对应通信域的context（消息区）地址。

**约束说明**

当前最多只支持2个通信域。

**调用示例**

```python
asc.set_hccl_context(0, x)
```
