# asc.language.basic.get_sort_len

### asc.language.basic.get_sort_len(elem_count: int) → int

获取排序结构中的排序长度。

**对应的 Ascend C 函数原型**

```c++
namespace AscendC {
template <typename T>
__aicore__ inline uint32_t GetSortLen(const uint32_t elemCount);
} // namespace AscendC
```

**参数说明**

elem_count: 参与排序的元素个数。

**返回值说明**

int: 排序长度。

**约束说明**

elem_count 必须为 32 位无符号整数。

**调用示例**

```python
length = asc.get_sort_len(100)
```
