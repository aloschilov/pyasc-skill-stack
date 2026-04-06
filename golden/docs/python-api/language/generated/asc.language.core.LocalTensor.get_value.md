# asc.language.core.LocalTensor.get_value

#### LocalTensor.get_value(index: int) → int

获取LocalTensor指定索引的数值。
该接口仅在LocalTensor的TPosition为VECIN/VECCALC/VECOUT时支持。

**对应的Ascend C函数原型**

> ```c++
> __aicore__ inline __inout_pipe__(S) PrimType GetValue(const uint32_t index) const
> ```

**参数说明**

- index：LocalTensor索引，单位为元素。

**返回值说明**

LocalTensor指定索引的数值，PrimType类型。

**调用示例**

> ```python
> src_len = 256
> num = 100
> for i in range(src_len):
>     element = input_local.get_value(i)  # 获取input_local中第i个位置的数值
> ```
