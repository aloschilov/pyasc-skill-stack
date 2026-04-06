# asc.language.basic.mrg_sort

### asc.language.basic.mrg_sort(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), sort_list: MrgSortSrcList, element_count_list: List[int], sorted_num: List[int], valid_bit: int, repeat_time: int, is_exhausted_suspension: bool = False) → None

### asc.language.basic.mrg_sort(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), sort_list: MrgSortSrcList, params: MrgSort4Info) → None

将已经排好序的多个队列合并成一条队列，结果按照指定顺序排序。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void MrgSort(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src,
                             const uint16_t elementCountList[], uint32_t sortedNum[],
                             uint16_t validBit, uint16_t repeatTime,
                             bool isExhaustedSuspension = false)
```

```c++
template <typename T>
__aicore__ inline void MrgSort(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src,
                             const MrgSort4Info& params)
```

**参数说明**

- dst (asc.LocalTensor): 目的操作数，存储经过排序后的结果。支持的TPosition为VECIN/VECCALC/VECOUT。
- src (asc.MrgSortSrcList): 源操作数，多个已经排好序的队列。具体定义如下：

```python
class MrgSortSrcList:
    src1: LocalTensor  # 第一个已经排好序的Region Proposals队列
    src2: LocalTensor  # 第二个已经排好序的Region Proposals队列
    src3: LocalTensor  # 第三个已经排好序的Region Proposals队列
    src4: LocalTensor  # 第四个已经排好序的Region Proposals队列
```

- element_count_list: 各个源队列的长度（元素数目），类型为uint16_t数组。
- sorted_num: 输出参数，存储各个队列排序后的元素数目，类型为uint32_t数组。
- valid_bit: 有效队列个数位掩码。
- repeat_time: 迭代次数。
- is_exhausted_suspension: 某条队列耗尽后，指令是否需要停止，类型为bool，默认false。
- params (asc.MrgSort4Info)：排序所需参数。
  - element_lengths: 四个源Region Proposals队列的长度（Region Proposal数目），类型为长度为4的uint16_t数组，每个元素取值范围[0, 4095]。
  - is_exhausted_suspension: 某条队列耗尽后，指令是否需要停止，类型为bool，默认false。
  - valid_bit：有效队列个数。
  - repeat_times：迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。取值范围[1,255]。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 不支持源操作数与目的操作数之间存在地址重叠。

**调用示例**

```python
src_list = asc.MrgSortSrcList(queue1, queue2, queue3, queue4)
element_counts = [16, 16, 16, 16]
sorted_nums = [0, 0, 0, 0]
valid_bit = 15  # 所有4个队列都有效
repeat_time = 1
asc.mrg_sort(dst, src_list, element_counts, sorted_nums, valid_bit, repeat_time)

src_list = asc.MrgSortSrcList(queue1, queue2, queue3, queue4)
element_lengths = [16, 16, 16, 16]
params = asc.MrgSort4Info(element_lengths, False, 15, 1)
asc.mrg_sort(dst, src_list, params)
```
