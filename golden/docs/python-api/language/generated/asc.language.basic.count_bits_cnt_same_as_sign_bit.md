# asc.language.basic.count_bits_cnt_same_as_sign_bit

### asc.language.basic.count_bits_cnt_same_as_sign_bit(value_in: int) → int

计算一个 int64_t 类型数字的二进制中，从最高数值位开始与符号位相同的连续比特位的个数。
当输入是 -1 （比特位全 1 ）或者 0 （比特位全 0 ）时，返回 -1 。

**对应的 Ascend C 函数原型**

> ```c++
> __aicore__ inline int64_t CountBitsCntSameAsSignBit(int64_t valueIn);
> ```

**参数说明**

- value_in：输入数据
  - 数据类型 int64_t 。

**返回值说明**

- 返回从最高数值位开始和符号位相同的连续比特位的个数。

**调用示例**

> ```python
> import asc
> value_in = 0x0f00000000000000
> ans = asc.count_bits_cnt_same_as_sign_bit(value_in)
> # ans 输出: 3
> ```
