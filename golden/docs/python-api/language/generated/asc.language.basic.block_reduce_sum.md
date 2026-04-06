# asc.language.basic.block_reduce_sum

### asc.language.basic.block_reduce_sum(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat: int, mask: int, dst_rep_stride: int, src_blk_stride: int, src_rep_stride: int) → None

### asc.language.basic.block_reduce_sum(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat: int, mask: List[int], dst_rep_stride: int, src_blk_stride: int, src_rep_stride: int) → None

对每个datablock内所有元素求和。源操作数相加采用二叉树方式，两两相加。归约指令的总体介绍请参考如何使用归约计算API。
以128个half类型的数据求和为例，每个datablock可以计算16个half类型数据，分成8个datablock进行计算；每个datablock内，通过二叉树的方式，两两相加。
需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。
例如，源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，然后计算-30000+100=-29900，最后计算65504-29900=35604。

**对应的Ascend C函数原型**

- mask逐比特模式
  ```c++
  template <typename T, bool isSetMask = true>
  __aicore__ inline void BlockReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
  ```
- mask连续模式
  ```c++
  template <typename T, bool isSetMask = true>
  __aicore__ inline void BlockReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,const int32_t repeatTime,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
  ```

**参数说明**

- is_set_mask: 是否在接口内部设置mask。
  - True，表示在接口内部设置mask。
  - False，表示在接口外部设置mask，开发者需要使用set_vector_mask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。
- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。
- src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- repeat_time：迭代次数。取值范围为[0, 255]。
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
- dst_rep_stride：目的操作数相邻迭代间的地址步长。以一个repeat_time归约后的长度为单位。每个repeat_time(8个datablock)归约后，得到8个元素，所以输入类型为half类型时，rep_stride单位为16Byte；输入类型为float类型时，rep_stride单位为32Byte。
- src_blk_stride：单次迭代内datablock的地址步长。
- src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 为了节省地址空间，您可以定义一个Tensor，供源操作数与目的操作数同时使用（即地址重叠），需要注意计算后的目的操作数数据不能覆盖未参与计算的源操作数，需要谨慎使用。

**调用示例**

- mask连续模式
  ```python
  asc.block_reduce_sum(z_local, x_local, repeat=1, mask=128, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
  ```
- mask逐bit模式
  ```python
  uint64_max = 2**64 - 1
  mask = [uint64_max, uint64_max]
  asc.block_reduce_sum(z_local, x_local, repeat=1, mask=mask, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
  ```
