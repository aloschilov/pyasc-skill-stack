# asc.language.basic.get_store_atomic_config

### asc.language.basic.get_store_atomic_config() → tuple[int, int]

获取原子操作使能位与原子操作类型的值。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void GetStoreAtomicConfig(uint16_t& atomicType, uint16_t& atomicOp)
```

**参数说明**

无。

**返回值说明**

- atomic_type（第一个返回值）：原子操作使能位。
  - 0：无原子操作
  - 1：使能原子操作，进行原子操作的数据类型为float
  - 2：使能原子操作，进行原子操作的数据类型为half
  - 3：使能原子操作，进行原子操作的数据类型为int16_t
  - 4：使能原子操作，进行原子操作的数据类型为int32_t
  - 5：使能原子操作，进行原子操作的数据类型为int8_t
  - 6：使能原子操作，进行原子操作的数据类型为bfloat16_t
- atomic_op（第二个返回值）：原子操作类型。
  - 0：求和操作

**约束说明**

此接口需要与set_store_atomic_config(ISASI)配合使用，用以获取原子操作使能位与原子操作类型的值。

**调用示例**

```python
asc.set_store_atomic_config(asc.AtomicDtype.ATOMIC_F16, asc.AtomicOp.ATOMIC_SUM)
atomic_type, atomic_op = asc.get_store_atomic_config()
```
