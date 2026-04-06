# asc.language.core.LocalTensor.get_position

#### LocalTensor.get_position() → int

获取LocalTensor所在的TPosition逻辑位置，支持TPosition为VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2。

**对应的Ascend C函数原型**

```c++
__aicore__ inline int32_t GetPosition() const
```

**参数说明**

无。

**返回值说明**

LocalTensor所在的TPosition逻辑位置。

**调用示例**

```python
src_pos = input_local.get_position()
if src_pos == asc.TPosition.VEECCALC:
    # 处理逻辑1
elif src_pos == asc.TPosition.A1:
    # 处理逻辑2
else:
    # 处理逻辑3
```
