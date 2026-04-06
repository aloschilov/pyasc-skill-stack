# asc.language.basic.set_pad_value

### asc.language.basic.set_pad_value(padding_value: int | float, pos: TPosition | None = TPosition.MAX) → None

设置asc.data_copy_pad需要填充的数值。支持的通路如下：GM->VECIN/GM->VECOUT填充值。

**对应的Ascend C函数原型**

```c++
template <typename T, TPosition pos = TPosition::MAX>
__aicore__ inline void SetPadValue(T paddingValue)
```

**参数说明**

- padding_value: 输入，asc.data_copy_pad接口填充的数值，数据与asc.data_copy_pad接口搬运的数据类型一致。
- pos:
  - 输入，用于指定asc.data_copy_pad接口搬运过程中从GM搬运数据到哪一个目的地址，目的地址通过逻辑位置来表达。
  - 默认值为asc.TPosition.MAX，等效于asc.TPosition.VECIN或asc.TPosition.VECOUT。

**返回值说明**

无。

**约束说明**

无。

**调用示例**

```python
import asc

asc.set_pad_value(37)
asc.set_pad_value(37, asc.TPosition.VECIN)
```
