# asc.language.basic.pair_reduce_sum

### asc.language.basic.pair_reduce_sum(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_time: int, mask: int, dst_rep_stride: int, src_blk_stride: int, src_rep_stride: int, is_set_mask: bool = True) → None

### asc.language.basic.pair_reduce_sum(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_time: int, mask: List[int], dst_rep_stride: int, src_blk_stride: int, src_rep_stride: int, is_set_mask: bool = True) → None

PairReduceSum：相邻两个（奇偶）元素求和。例如，对于序列 (a1, a2, a3, a4, a5, a6, …)，
相邻两个数据求和为 (a1+a2, a3+a4, a5+a6, …)。

**对应的Ascend C函数原型**

```c++
// mask 逐bit模式
template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const uint64_t mask[],
                                    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);
```

```c++
// mask 连续模式
template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask,
                                    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);
```

**参数说明**

- dst：输出操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。LocalTensor 的起始地址需要 32 字节对齐。
- src：输入操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。LocalTensor 的起始地址需要 32 字节对齐。
- repeat_time：迭代次数，取值范围 [0, 255]。关于该参数的具体描述请参考如何使用 Tensor 高维切分计算 API。
- mask：控制每次迭代内参与计算的元素。

  **逐比特模式**
  数组形式，按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。
  - 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
  - 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
  - 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]

  **连续模式**
  整数形式，表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
  - 16 位：mask ∈ [1, 128]
  - 32 位：mask ∈ [1, 64]
  - 64 位：mask ∈ [1, 32]
- dst_rep_stride：
  目的操作数相邻迭代间的地址步长，以一个 repeat 归约后的长度为单位。
  - PairReduce 完成后，一个 repeat 的长度减半。
  - 注意：Atlas 训练系列产品不支持配置 0。
- src_blk_stride：单次迭代内数据 block 的地址步长，详细说明请参考 data_block_stride。
- src_rep_stride：源操作数相邻迭代间的地址步长，即每次迭代跳过的 data block 数目。详细说明请参考 repeatStride。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 如果两两相加的两个元素mask位未配置（即当前两个元素不参与运算）。
  - 对于 Atlas 200I/500 A2 推理产品 ，对应的目的操作数中的值会置为0。
  - 对于其他产品型号，对应的目的操作数中的值不会变化。
- 比如float场景下对64个数使用当前指令，mask配置为62，表示最后两个元素不参与运算。
  - 对于 Atlas 200I/500 A2 推理产品 ，目的操作数中最后一个值会返回0。
  - 对于其他产品型号，目的操作数中最后一个值不会变化。

**调用示例**

- mask 连续模式
  ```python
  x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
  z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
  asc.pair_reduce_sum(z_local, x_local, repeat_time=2, mask=128,
                      dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
  ```
- mask 逐bit模式
  ```python
  x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
  z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
  uint64_max = 2**64 - 1
  mask = [uint64_max, uint64_max]
  asc.pair_reduce_sum(z_local, x_local, repeat_time=2, mask=mask,
                      dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
  ```
