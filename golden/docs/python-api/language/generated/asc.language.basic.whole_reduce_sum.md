# asc.language.basic.whole_reduce_sum

### asc.language.basic.whole_reduce_sum(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: int, repeat_time: int, dst_rep_stride: int, src_blk_stride: int, src_rep_stride: int, is_set_mask: bool = True) → None

### asc.language.basic.whole_reduce_sum(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: List[int], repeat_time: int, dst_rep_stride: int, src_blk_stride: int, src_rep_stride: int, is_set_mask: bool = True) → None

每个迭代内所有数据求和。归约指令的总体介绍请参考如何使用归约指令。

**对应的Ascend C函数原型**

```c++
// mask 逐比特模式
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const uint64_t mask[],
                                      const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);
```

```c++
// mask 连续模式
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask,
                                      const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);
```

**参数说明**

- dst：
  输出，目的操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
  - LocalTensor 的起始地址需保证 2 字节对齐（half 数据类型）或 4 字节对齐（float 数据类型）。
  - 数据类型根据产品支持情况：half / float。
- src：
  输入，源操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
  - LocalTensor 起始地址需 32 字节对齐。
  - 源操作数的数据类型需要与目的操作数保持一致。
- mask：
  控制每次迭代内参与计算的元素。
  - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。
    - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
    - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
    - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
    - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算
  - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
    - 操作数 16 位：mask ∈ [1, 128]
    - 操作数 32 位：mask ∈ [1, 64]
    - 操作数 64 位：mask ∈ [1, 32]
- repeat_time：迭代次数，取值范围 [0, 255]。具体描述请参考 如何使用Tensor 高维切分计算API。
- dst_rep_stride：
  目的操作数相邻迭代间地址步长，以一个 repeat 归约后的长度为单位。
  - 单位为 dst 数据类型所占字节长度。比如当dst为half时，单位为2Bytes。
  - 注意：Atlas 训练系列产品不支持配置 0。
- src_blk_stride：单次迭代内datablock的地址步长。详细说明请参考data_block_stride。
- src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的DataBlock数目。详细说明请参考repeat_stride。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。
- 对于whole_reduce_sum，其内部的相加方式采用二叉树方式，两两相加
  假设源操作数为128个half类型的数据[data0,data1,data2…data127]，一个repeat可以计算完，计算过程如下。
  1. data0和data1相加得到data00，data2和data3相加得到data01…data124和data125相加得到data62，data126和data127相加得到data63；
  2. data00和data01相加得到data000，data02和data03相加得到data001…data62和data63相加得到data031；
  3. 以此类推，得到目的操作数为1个half类型的数据[data]。
  4. 需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。例如源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，第二步计算-30000+100=-29900，第四步计算65504-29900=35604。

**调用示例**

- mask 连续模式
  ```python
  x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
  z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
  asc.whole_reduce_sum(z_local, x_local, mask=128, repeat_time=4,
                       dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
  ```
- mask 逐bit模式
  ```python
  uint64_max = 2**64 - 1
  mask = [uint64_max, uint64_max]
  asc.whole_reduce_sum(z_local, x_local, mask=mask, repeat_time=4,
                       dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
  ```
