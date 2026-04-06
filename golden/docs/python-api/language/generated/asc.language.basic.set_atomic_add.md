# asc.language.basic.set_atomic_add

### asc.language.basic.set_atomic_add() → None

调用该接口后，可对后续的从VECOUT/L0C/L1到GM的数据传输开启原子累加，
通过dtype参数设定不同类型的数据。

**对应的 Ascend C 函数原型**

```c++
template <typename T>
__aicore__ inline void SetAtomicAdd();
```

**参数说明**

- dtype：
  原子加操作的数据类型，由 Python 前端指定。
  - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。

**约束说明**

- 累加操作完成后，建议通过set_atomic_none关闭原子累加，以免影响后续相关指令功能。
- 该指令执行前不会对GM的数据做清零操作，开发者根据实际的算子逻辑判断是否需要清零，如果需要自行进行清零操作。

**调用示例**

```python
dtype = asc.float32
asc.set_atomic_add(dtype)
```
