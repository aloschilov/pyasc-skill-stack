# asc.language.basic.get_sort_offset

### asc.language.basic.get_sort_offset(elem_offset: int) → int

获取排序结构中的排序偏移量。

**对应的 Ascend C 函数原型**

```c++
namespace AscendC {
template <typename T>
__aicore__ inline uint32_t GetSortOffset(const uint32_t elemOffset);
} // namespace AscendC
```

**参数说明**

elem_offset: 元素的偏移量。必须为32位无符号整数。

**返回值说明**

int: 排序偏移量

**约束说明**

elem_offset 必须为 32 位无符号整数。

**调用示例**

```python
offset = asc.get_sort_offset(10)
```
