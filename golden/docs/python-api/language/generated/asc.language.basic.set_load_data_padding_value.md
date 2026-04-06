# asc.language.basic.set_load_data_padding_value

### asc.language.basic.set_load_data_padding_value(pad_value: int) → None

用于调用 load_3d_v1接口/load_3d_v2 接口时设置 Pad 填充的数值。 load_3d_v1/load_3d_v2 的模板参数 isSetPadding 设置为 true 时，用户需要通过本接口设置 Pad 填充的数值，设置为 false 时，本接口设置的填充值不生效。

**对应的 Ascend C 函数原型**

```c++
_template <typename T>
__aicore__ inline void SetLoadDataPaddingValue(const T padValue)
```

**参数说明**

- padValue
  : 输入， Pad 填充值的数值。

**约束说明**
: 无

**调用示例**

```python
import asc
asc.set_load_data_padding_value(10)
asc.set_load_data_padding_value(2.0)
```
