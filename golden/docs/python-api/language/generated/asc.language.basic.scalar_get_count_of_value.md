# asc.language.basic.scalar_get_count_of_value

### asc.language.basic.scalar_get_count_of_value(value_in: int, count_value: int) → int

获取一个 uint64_t 类型数字的二进制中 0 或者 1 的个数。

**对应的 Ascend C 函数原型**

> ```c++
> template <int countValue>
> __aicore__ inline int64_t ScalarGetCountOfValue(uint64_t valueIn);
> ```

**参数说明**

- value_in：输入数据
  - 被统计的二进制数字。
- count_value：指定统计 0 还是统计 1 的个数。
  - 只能输入 0 或 1 。

**返回值说明**

- value_in 中 0 或者 1 的个数。

**调用示例**

> ```python
> import asc
> value_in = 0xffff
> count_bits = 1
> one_count = asc.scalar_get_count_of_value(value_in, count_bits)
> # 输出数据 oneCount : 16
> ```
