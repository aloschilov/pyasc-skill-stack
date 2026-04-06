# asc.language.core.LocalTensor.set_value

#### LocalTensor.set_value(index: int, value: int | float) → None

设置LocalTensor中的某个值。
该接口仅在LocalTensor的TPosition为VECIN/VECCALC/VECOUT时支持。

**对应的Ascend C函数原型**

```c++
template <typename T1> __aicore__ inline __inout_pipe__(S)
void SetValue(const uint32_t index, const T1 value) const
```

**参数说明**

- index：LocalTensor索引，单位为元素。
- value：待设置的数值。

**约束说明**

不要大量使用set_value对LocalTensor进行赋值，会使性能下降。若需要大批量赋值，请根据实际场景选择数据填充基础API接口或数据填充高阶API接口（Pad、Broadcast），以及在需要生成递增数列的场景，选择ArithProgression。

**调用示例**

```python
src_len = 256
num = 100
for i in range(src_len):
    input_local.set_value(i, num)   # 对input_local中第i个位置进行赋值为num
```
