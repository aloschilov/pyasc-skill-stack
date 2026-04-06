# asc.language.basic.set_atomic_max

### asc.language.basic.set_atomic_max() → None

原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较：
将待拷贝的内容和GM已有内容进行比较，将最大值写入GM。
可通过设置模板参数来设定不同的数据类型。

**对应的 Ascend C 函数原型**

```c++
template <typename T>
__aicore__ inline void SetAtomicMax();
```

**参数说明**

- dtype：
  原子 max 操作的数据类型，由 Python 前端指定。
  - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。

**约束说明**

- 使用完后，建议通过set_atomic_none关闭原子累加，以免影响后续相关指令功能。
- 对于910B，目前无法对bfloat16_t类型设置inf/nan模式。

**调用示例**

```python
dtype = asc.int32
asc.set_atomic_max(dtype)
```
