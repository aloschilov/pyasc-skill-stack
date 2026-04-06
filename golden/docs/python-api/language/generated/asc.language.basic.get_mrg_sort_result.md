# asc.language.basic.get_mrg_sort_result

### asc.language.basic.get_mrg_sort_result() → tuple[int, int, int, int]

获取mrg_sort已经处理过的队列里的Region Proposal个数，并依次存储在四个出参中。

本接口和mrg_sort相关指令的配合关系如下：

- 配合mrg_sort_4指令使用，获取mrg_sort_4指令处理过的队列里的Region Proposal个数。使用时，需要将mrg_sort_4中的mrg_sort_4_info.if_exhausted_suspension参数配置为TTrue，该配置模式下某条队列耗尽后，mrg_sort_4指令即停止。
  以上说明适用于如下型号：
  - Atlas 推理系列产品AI Core
- 配合mrg_sort指令使用，获取mrg_sort指令处理过的队列里的Region Proposal个数。使用时，需要将mrg_sort中的mrg_sort_4_info.if_exhausted_suspension参数配置为True，该配置模式下某条队列耗尽后，mrg_sort指令即停止。
  以上说明适用于如下型号：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品
  - Atlas 200I/500 A2 推理产品

**对应的Ascend C函数原型**

```c++
__aicore__ inline void GetMrgSortResult(uint16_t &mrgSortList1, uint16_t &mrgSortList2, uint16_t &mrgSortList3, uint16_t &mrgSortList4)
```

**参数说明**

无。

**返回值说明**

- mrg_sort_list1（第一个返回值）：类型为uint16_t，表示mrg_sort第一个队列里已经处理过的Region Proposal个数。
- mrg_sort_list2（第二个返回值）：类型为uint16_t，表示mrg_sort第二个队列里已经处理过的Region Proposal个数。
- mrg_sort_list3（第三个返回值）：类型为uint16_t，表示mrg_sort第三个队列里已经处理过的Region Proposal个数。
- mrg_sort_list4（第四个返回值）：类型为uint16_t，表示mrg_sort第四个队列里已经处理过的Region Proposal个数。

**约束说明**

无。

**调用示例**

```python
src1 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
src2 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=512, tile_size=512)
src3 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=1024, tile_size=512)
src4 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=1536, tile_size=512)
dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=2048)
element_count_list = [128, 128, 128, 128]
sorted_num = [0, 0, 0, 0]
asc.mrg_sort(dst, sort_list, element_count_list, sorted_num, valid_bit=15, repeat_time=1)
asc.mrg_sort(dst, sort_list, element_count_list, sorted_num, valid_bit=15,
            repeat_time=1, is_exhausted_suspension=True)
mrg_sort4_info = asc.MrgSort4Info(element_count_list, if_exhausted_suspension=False,
                                  valid_bit=7, repeat_times=1)
asc.mrg_sort(dst, sort_list, mrg_sort4_info)

mrg1, mrg2, mrg3, mrg4 = asc.get_mrg_sort_result()
```
