# asc.language.basic.mrg_sort4

### asc.language.basic.mrg_sort4(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: MrgSortSrcList, params: MrgSort4Info) → None

将已经排好序的最多4条Region Proposals队列，排列并合并成1条队列，结果按照score域由大到小排序。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void MrgSort4(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src, const MrgSort4Info& params)
```

**参数说明**

- dst (asc.LocalTensor): 目的操作数，存储经过排序后的Region Proposals。支持的TPosition为VECIN/VECCALC/VECOUT。
- src (asc.MrgSortSrcList): 源操作数，多个已经排好序的队列。具体定义如下：

```python
class MrgSortSrcList:
    src1: LocalTensor
    src2: LocalTensor
    src3: LocalTensor
    src4: LocalTensor
```

- params (asc.MrgSort4Info)
  : 排序所需参数。
    - element_lengths: 四个源Region Proposals队列的长度（Region Proposal数目），类型为长度为4的uint16_t数组，每个元素取值范围[0, 4095]。
    - is_exhausted_suspension: 某条队列耗尽后，指令是否需要停止，类型为bool，默认false。
    - valid_bit：有效队列个数。
    - repeat_times：迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。取值范围[1,255]。

**约束说明**

- 当存在proposal[i]与proposal[j]的score值相同时，如果i>j，则proposal[j]将首先被选出来，排在前面。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 不支持源操作数与目的操作数之间存在地址重叠。

**调用示例**

```python
# vconcat_work_local为已经创建并且完成排序的4个Region Proposals，每个Region Proposal数目是16个
src_list = asc.MrgSortSrcList(vconcat_work_local[0], vconcat_work_local[1], vconcat_work_local[2], vconcat_work_local[3])
element_lengths = [16, 16, 16, 16]
src_info = asc.MrgSort4Info(element_lengths, False, 15, 1)
asc.mrg_sort4(dst_local, src_list, src_info)
```
