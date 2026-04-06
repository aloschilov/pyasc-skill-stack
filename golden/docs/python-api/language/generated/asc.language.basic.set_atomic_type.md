# asc.language.basic.set_atomic_type

### asc.language.basic.set_atomic_type() → None

通过设置模板参数来设定原子操作不同的数据类型。

**对应的 Ascend C 函数原型**

```c++
template <typename T>
__aicore__ inline void SetAtomicType();
```

**参数说明**

- dtype：
  原子操作使用的数据类型，由 Python 前端指定。
  - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。

**约束说明**

- 需要和set_atomic_add、set_atomic_max、set_atomic_min配合使用。
- 使用完成后，建议清空原子操作的状态（详见set_atomic_none），以免影响后续相关指令功能。

**调用示例**

```python
dtype = asc.float16
asc.set_atomic_type(dtype)
```
