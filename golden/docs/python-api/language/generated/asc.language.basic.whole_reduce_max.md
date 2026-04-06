# asc.language.basic.whole_reduce_max

### asc.language.basic.whole_reduce_max(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: List[int], repeat_time: int, dst_rep_stride: int, src_blk_stride: int, src_rep_stride: int, order: ReduceOrder | None = ReduceOrder.ORDER_VALUE_INDEX, is_set_mask: bool = True) → None

### asc.language.basic.whole_reduce_max(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: int, repeat_time: int, dst_rep_stride: int, src_blk_stride: int, src_rep_stride: int, order: ReduceOrder | None = ReduceOrder.ORDER_VALUE_INDEX, is_set_mask: bool = True) → None

每个repeat内所有数据求最大值以及其索引index，返回的索引值为每个repeat内部索引。
归约指令的总体介绍请参考如何使用归约指令。

**对应的Ascend C函数原型**

```c++
// mask 逐bit模式
template <typename T, bool isSetMask = true>
 __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask[], const int32_t repeatTime,
                                        const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
                                        ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
```

```c++
// mask 连续模式
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t mask, const int32_t repeatTime,
                                        const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
                                        ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
```

**参数说明**

- dst：
  输出，目的操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
  - LocalTensor 的起始地址需 4 字节对齐（half 数据类型）或 8 字节对齐（float 数据类型）。
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
  目的操作数相邻迭代间地址步长。
  - 以一个 repeat 归约后的长度为单位。
  - 返回索引和最值时，单位为 dst 数据类型所占字节长度的两倍，比如当dst为half时，单位为4Bytes。
  - 仅返回最值时，单位为 dst 数据类型所占字节长度。
  - 仅返回索引时，单位为 uint32_t 类型所占字节长度。
  - 注意：Atlas 训练系列产品不支持配置 0。
- src_blk_stride：单次迭代内数据 block 的地址步长。详细说明请参考 data_block_stride。
- src_rep_stride：源操作数相邻迭代间地址步长，即每次迭代跳过的 data_block 数目。详细说明请参考 repeat_stride。
- order
  可选参数，指定 dst 中 index 与 value 的相对位置以及返回结果行为，类型为 ReduceOrder。
  - 默认值为 asc.ReduceOrder.ORDER_VALUE_INDEX。
  - asc.ReduceOrder.ORDER_VALUE_INDEX：value 位于低半部，返回顺序 [value, index]。
  - asc.ReduceOrder.ORDER_INDEX_VALUE：index 位于低半部，返回顺序 [index, value]。
  - asc.ReduceOrder.ORDER_ONLY_VALUE：只返回最值，顺序 [value]。
  - asc.ReduceOrder.ORDER_ONLY_INDEX：只返回索引，顺序 [index]。
  - 910B，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。
  - 910C，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。
- dst结果存储顺序由order决定，默认为最值、最值索引。返回结果中索引index数据按照dst的数据类型进行存储，比如dst使用half类型时，index按照half类型进行存储，读取时需要使用reinterpret_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret_cast<uint16_t\*>，若输入是float，需要使用reinterpret_cast<uint32_t\*>。比如完整样例中，前两个计算结果为[9.980e-01 5.364e-06]，5.364e-06需要使用reinterpret_cast方法转换得到索引值90。
- 针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考选择低延迟指令，优化归约操作性能，具体样例请参考ReduceCustom。

**调用示例**

- mask 连续模式，默认 ORDER_VALUE_INDEX
  ```python
  x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
  z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
  asc.whole_reduce_max(z_local, x_local, mask=128, repeat_time=4,
                       dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
  ```
- mask 连续模式，ORDER_INDEX_VALUE
  ```python
  x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
  z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
  asc.whole_reduce_max(z_local, x_local, mask=128, repeat_time=4,
                       dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8,
                       order=asc.ReduceOrder.ORDER_INDEX_VALUE)
  ```
- mask 逐bit模式
  ```python
  uint64_max = 2**64 - 1
  mask = [uint64_max, uint64_max]
  asc.whole_reduce_max(z_local, x_local, mask=mask, repeat_time=4,
                       dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
  ```
- mask 逐bit模式，ORDER_INDEX_VALUE
  ```python
  uint64_max = 2**64 - 1
  mask = [uint64_max, uint64_max]
  asc.whole_reduce_max(z_local, x_local, mask=mask, repeat_time=4,
                       dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8,
                       order=asc.ReduceOrder.ORDER_INDEX_VALUE)
  ```
