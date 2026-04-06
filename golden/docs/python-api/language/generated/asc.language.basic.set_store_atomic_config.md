# asc.language.basic.set_store_atomic_config

### asc.language.basic.set_store_atomic_config(atomic_type: AtomicDtype, atomic_op: AtomicOp) → None

设置原子操作使能位与原子操作类型。

**对应的Ascend C函数原型**

```c++
template <AtomicDtype type, AtomicOp op>
__aicore__ inline void SetStoreAtomicConfig()
```

**参数说明**

- type：原子操作使能位，AtomicDtype枚举类的定义如下：
  ```python
  class AtomicDtype(IntEnum):
      ATOMIC_NONE = 0   // 无原子操作
      ATOMIC_F32 = 1    // 使能原子操作，进行原子操作的数据类型为float
      ATOMIC_F16 = 2    // 使能原子操作，进行原子操作的数据类型为half
      ATOMIC_S16 = 3    // 使能原子操作，进行原子操作的数据类型为int16_t
      ATOMIC_S32 = 4    // 使能原子操作，进行原子操作的数据类型为int32_t
      ATOMIC_S8 = 5     // 使能原子操作，进行原子操作的数据类型为int8_t
      ATOMIC_BF16 = 6   // 使能原子操作，进行原子操作的数据类型为bfloat16_t
  ```
- op：原子操作类型，仅当使能原子操作时有效（即“type”为非“ATOMIC_NONE”的场景），当前仅支持求和操作。
  ```python
  class AtomicOp(IntEnum):
      ATOMIC_SUM = 0    // 求和操作
  ```

**约束说明**

无。

**调用示例**

```python
asc.set_store_atomic_config(asc.AtomicDtype.ATOMIC_F16, asc.AtomicOp.ATOMIC_SUM)
```
