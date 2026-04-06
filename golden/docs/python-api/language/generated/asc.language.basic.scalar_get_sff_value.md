# asc.language.basic.scalar_get_sff_value

### asc.language.basic.scalar_get_sff_value(value_in: int, count_value: int) → int

获取一个 uint64_t 类型数字的二进制表示中，从最低有效位（LSB）开始第一个 0 或 1 出现的位置。
如果未找到指定值，则返回 -1。

**对应的 Ascend C 函数原型**

```c++
template <int countValue>
__aicore__ inline int64_t ScalarGetSFFValue(uint64_t valueIn);
```

**参数说明**

- value_in：
  输入数据，类型为 uint64_t。
  - 表示待查找的无符号整数。
- count_value：
  指定要查找的值，类型为 int。
  - 取值为 0 或 1。
  - 0 表示查找从最低有效位开始的第一个 0 出现的位置；
  - 1 表示查找从最低有效位开始的第一个 1 出现的位置。

**返回值说明**

- 返回 int64 类型的数：
  表示 value_in 的二进制表示中，第一个匹配值（0 或 1）出现的位置。
  - 如果未找到，则返回 -1。

**调用示例**

```python
value_in = 28
count_value = 1
one_count = asc.scalar_get_sff_value(value_in, count_value)
```
