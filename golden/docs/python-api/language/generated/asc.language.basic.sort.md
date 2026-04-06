# asc.language.basic.sort

### asc.language.basic.sort(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), concat: [LocalTensor](../core.md#asc.language.core.LocalTensor), index: [LocalTensor](../core.md#asc.language.core.LocalTensor), tmp: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_time: int) → None

排序函数，按照数值大小进行降序排序。

**对应的Ascend C函数原型**

```c++
template <typename T, bool isFullSort>
__aicore__ inline void Sort(const LocalTensor<T> &dst, const LocalTensor<T> &concat,
                           const LocalTensor<uint32_t> &index, LocalTensor<T> &tmp,
                           const int32_t repeatTime)
```

**参数说明**

- dst (asc.LocalTensor): 目的操作数，shape为[2n]。
- concat (asc.LocalTensor): 源操作数，shape为[n]，数据类型与目的操作数保持一致。
- index (asc.LocalTensor): 源操作数，shape为[n]。固定为uint32_t数据类型。
- tmp (asc.LocalTensor): 临时空间。接口内部复杂计算时用于存储中间变量，由开发者提供。数据类型与源操作数保持一致。
- repeat_time (int): 重复迭代次数，int32_t类型。
- is_full_sort (bool, 可选): 模板参数，是否开启全排序模式。

**约束说明**

- 当存在score[i]与score[j]相同时，如果i>j，则score[j]将首先被选出来，排在前面，即index的顺序与输入顺序一致。
- 非全排序模式下，每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

```python
# 处理128个half类型数据
element_count = 128
sort_repeat_times = element_count // 32
extract_repeat_times = element_count // 32
asc.sort(dst_local, concat_local, index_local, tmp_local, sort_repeat_times, is_full_sort=True)
```
