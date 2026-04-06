# asc.language.basic.scalar_count_leading_zero

### asc.language.basic.scalar_count_leading_zero(value_in: int) → int

计算一个 uint64_t 类型数字前导 0 的个数
（二进制从最高位到第一个 1 一共有多少个 0 ）。

**对应的 Ascend C 函数原型**

> ```c++
> __aicore__ inline int64_t ScalarCountLeadingZero(uint64_t valueIn);
> ```

**参数说明**

- value_in：输入数据
  - 被统计的二进制数字。

**返回值说明**

- 返回 value_in 的前导 0 的个数。

**调用示例**

> ```python
> import asc
> value_in = 0x0fffffffffffffff
> ans = asc.scalar_count_leading_zero(value_in)
> # ans 输出: 4
> ```
