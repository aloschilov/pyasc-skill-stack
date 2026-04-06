# asc.language.basic.gather_mask

### asc.language.basic.gather_mask(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1_pattern: [LocalTensor](../core.md#asc.language.core.LocalTensor), reduce_mode: bool, mask: int, params: GatherMaskParams, gather_mask_mode=GatherMaskMode.DEFAULT) → int

### asc.language.basic.gather_mask(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1_pattern: int, reduce_mode: bool, mask: int, params: GatherMaskParams, gather_mask_mode=GatherMaskMode.DEFAULT) → int

以内置固定模式对应的二进制或者用户自定义输入的Tensor数值对应的二进制为gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。

**对应的Ascend C函数原型**

```c++
template <typename T, typename U, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                  const LocalTensor<U>& src1Pattern, const bool reduceMode,
                                  const uint32_t mask, const GatherMaskParams& gatherMaskParams,
                                  uint64_t& rsvdCnt)
```

```c++
template <typename T, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                  const uint8_t src1Pattern, const bool reduceMode,
                                  const uint32_t mask, const GatherMaskParams& gatherMaskParams,
                                  uint64_t& rsvdCnt)
```

**参数说明**

- dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型需要与目的操作数保持一致。
- src1_pattern: gather mask（数据收集的掩码），分为内置固定模式和用户自定义模式两种：
  - 内置固定模式：src1_pattern数据类型为uint8_t，取值范围为[1,7]，所有repeat迭代使用相同的gather mask。不支持配置src1_repeat_stride。

    1：01010101…0101 # 每个repeat取偶数索引元素
    2：10101010…1010 # 每个repeat取奇数索引元素
    3：00010001…0001 # 每个repeat内每四个元素取第一个元素
    4：00100010…0010 # 每个repeat内每四个元素取第二个元素
    5：01000100…0100 # 每个repeat内每四个元素取第三个元素
    6：10001000…1000 # 每个repeat内每四个元素取第四个元素
    7：11111111…1111 # 每个repeat内取全部元素
  - 用户自定义模式：src1_pattern数据类型为LocalTensor，迭代间间隔由src1_repeat_stride决定，迭代内src1_pattern连续消耗。
- reduce_mode: 用于选择mask参数模式，数据类型为bool，支持如下取值：
  - False：Normal模式。该模式下，每次repeat操作256Bytes数据，总的数据计算量为repeat_times \* 256Bytes。mask参数无效，建议设置为0。按需配置repeat_times、src0_block_stride、src0_repeat_stride参数。支持src1_pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1_repeat_stride。
  - True：Counter模式。根据mask等参数含义的不同，该模式有以下两种配置方式：

    配置方式一：每次repeat操作mask个元素，总的数据计算量为repeat_times \* mask个元素。mask值配置为每一次repeat计算的元素个数。按需配置repeat_times、src0_block_stride、src0_repeat_stride参数。支持src1_pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1_repeat_stride。
    配置方式二：总的数据计算量为mask个元素。mask配置为总的数据计算量。repeat_times值不生效，指令的迭代次数由源操作数和mask共同决定。按需配置src0_block_stride、src0_repeat_stride参数。支持src1_pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1_repeat_stride。
- mask: 用于控制每次迭代内参与计算的元素。根据reduce_mode，分为两种模式：
  - Normal模式：mask无效，建议设置为0。
  - Counter模式：取值范围[1, 232 – 1]。不同的版本型号Counter模式下，mask参数表示含义不同。具体配置规则参考上文reduce_mode参数描述。
- params: 控制操作数地址步长的数据结构，GatherMaskParams类型。具体参数包括：
  - src0_block_stride: 用于设置src0同一迭代不同DataBlock间的地址步长。
  - repeat_times: 迭代次数。
  - src0_repeat_stride: 用于设置src0相邻迭代间的地址步长。
  - src1_repeat_stride: 用于设置src1相邻迭代间的地址步长。
- mode: 模板参数，用于指定gather_mask的模式，当前仅支持默认模式GatherMaskMode.DEFAULT，为后续功能做预留。

**返回值说明**

该条指令筛选后保留下来的元素计数，对应dst_local中有效元素个数，数据类型为uint64_t。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。
- 若调用该接口前为Counter模式，在调用该接口后需要显式设置回Counter模式（接口内部执行结束后会设置为Normal模式）。

**调用示例**

```python
src0_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
dst_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
pattern_value = 2
reduce_mode = False
gather_mask_mode = asc.GatherMaskMode.DEFAULT
mask = 0
params = asc.GatherMaskParams(src0_block_stride=1, repeat_times=1, src0_repeat_stride=0, src1_repeat_stride=0)
rsvd_cnt = asc.gather_mask(dst_local, src0_local, pattern_value, reduce_mode, mask, params, gather_mask_mode)
```
