# asc.language.basic.repeat_reduce_sum

### asc.language.basic.repeat_reduce_sum(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_time: int, mask: int, dst_blk_stride: int, src_blk_stride: int, dst_rep_stride: int, src_rep_stride: int, is_set_mask: bool = True) → None

对每个 repeat 内的所有数据进行求和。
与 whole_reduce_sum 接口相比，不支持 mask 逐比特模式。
建议使用功能更全面的 whole_reduce_sum 接口。

**对应的Ascend C函数原型**

```c++
template <typename T, bool isSetMask = true>
__aicore__ inline void RepeatReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask,
                                        const int32_t dstBlkStride, const int32_t srcBlkStride, const int32_t dstRepStride, const int32_t srcRepStride);
```

**参数说明**

- dst：
  输出操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
  - LocalTensor 起始地址需保证 2 字节对齐（half 数据类型）或 4 字节对齐（float 数据类型）。
  - 数据类型根据产品支持情况：half / float。
- src：
  输入操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
  - LocalTensor 起始地址需 32 字节对齐。
  - 数据类型需与 dst 保持一致。
- repeat_time：
  重复迭代次数，取值范围 [0, 255]。
  - 矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代才能完成所有数据的读取与计算。repeatTime表示迭代的次数。
  - 具体描述请参考 Tensor 高维切分计算 API。
- mask：
  控制每次迭代内连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
  - 操作数为 16 位：mask ∈ [1, 128]
  - 操作数为 32 位：mask ∈ [1, 64]
- dst_blk_stride：此参数无效，可配置任意值。
- src_blk_stride：单次迭代内数据 data_block 的地址步长。详细说明请参考 data_block_stride。
- dst_rep_stride：
  目的操作数相邻迭代间的地址步长，以一个 repeat 归约后的长度为单位。
  - 单位为 dst 数据类型所占字节长度。比如当dst为half时，单位为2Bytes。
  - 注意：Atlas 训练系列产品不支持配置 0。
- src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的data_block数目。详细说明请参考 repeat_stride。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。
- 对于RepeatReduceSum，其内部的相加方式采用二叉树方式，两两相加
  假设源操作数为128个half类型的数据[data0,data1,data2…data127]，一个repeat可以计算完，计算过程如下。
  1. data0和data1相加得到data00，data2和data3相加得到data01…data124和data125相加得到data62，data126和data127相加得到data63；
  2. data00和data01相加得到data000，data02和data03相加得到data001…data62和data63相加得到data031；
  3. 以此类推，得到目的操作数为1个half类型的数据[data]。
  4. 需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。例如源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，第二步计算-30000+100=-29900，第四步计算65504-29900=35604。

**调用示例**

```python
x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
asc.repeat_reduce_sum(z_local, x_local, repeat_time=4, mask=128,
                      dst_blk_stride=0, src_blk_stride=1, dst_rep_stride=8, src_rep_stride=8)
```
