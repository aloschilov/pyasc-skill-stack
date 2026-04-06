# asc.language.basic.scalar_cast

### asc.language.basic.scalar_cast(value_in: float, dtype: Type[T], round_mode: RoundMode) → T

对标量的数据类型进行转换。

**对应的Ascend C函数原型**

```c++
template <typename T, typename U, RoundMode roundMode>
__aicore__ inline U ScalarCast(T valueIn);
```

**参数说明**

- value_in：被转换数据类型的标量。
- dtypeL：
  目标数据类型，由Python前端指定。
  - 支持：asc.half、asc.float16、asc.int32。
- round_mode：
  精度转换处理模式，类型为RoundMode枚举值。
  - asc.RoundMode.CAST_NONE：在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不取整。
  - asc.RoundMode.CAST_RINT：rint，四舍六入五成双取整。
  - asc.RoundMode.CAST_FLOOR：floor，向负无穷取整。
  - asc.RoundMode.CAST_CEIL：ceil，向正无穷取整。
  - asc.RoundMode.CAST_ROUND：round，四舍五入取整。
  - asc.RoundMode.CAST_ODD：Von Neumann rounding，最近邻奇数舍入。
  - 对应支持关系
    - float -> half(f322f16)：asc.RoundMode.CAST_ODD
    - float -> int32(f322s32)：asc.RoundMode.CAST_ROUND、asc.RoundMode.CAST_CEIL、asc.RoundMode.CAST_FLOOR、asc.RoundMode.CAST_RINT
  - ScalarCast的精度转换规则与Cast保持一致

**返回值说明**

返回值为转换后的标量，类型与dtype一致。

**调用示例**

```python
value_in = 2.5
dtype = asc.int32
round_mode = asc.RoundMode.CAST_ROUND
value_out = asc.scalar_cast(value_in, dtype, round_mode)
```
