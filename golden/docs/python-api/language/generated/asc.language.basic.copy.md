# asc.language.basic.copy

### asc.language.basic.copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: int, repeat_time: int, repeat_params: CopyRepeatParams) → None

### asc.language.basic.copy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: List[int], repeat_time: int, repeat_params: CopyRepeatParams) → None

在 Vector Core 的不同内部存储单元（VECIN, VECCALC, VECOUT）之间进行数据搬运。

这是一个矢量指令，支持通过掩码（mask）进行灵活的数据块选择，并通过重复参数（repeat parameters）
实现高效的间隔操作和高维数据处理。

**对应的Ascend C函数原型**

该接口支持两种掩码（mask）模式，以进行高维切分计算。

1. **mask 为逐 bit 模式**
   ```c++
   template <typename T, bool isSetMask = true>
   __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                               const uint64_t mask[], const uint8_t repeatTime,
                               const CopyRepeatParams& repeatParams)
   ```
2. **mask 为连续模式**
   ```c++
   template <typename T, bool isSetMask = true>
   __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                               const uint64_t mask, const uint8_t repeatTime,
                               const CopyRepeatParams& repeatParams)
   ```

**参数说明**

- dst (asc.LocalTensor): 目标操作数。
  - 必须是 LocalTensor。
  - 支持的 TPosition 为 asc.Position.VECIN, asc.Position.VECCALC, asc.Position.VECOUT。
  - 起始地址需要 32 字节对齐。
- src (asc.LocalTensor): 源操作数。
  - 必须是 LocalTensor，且数据类型与 dst 保持一致。
  - 支持的 TPosition 为 asc.Position.VECIN, asc.Position.VECCALC, asc.Position.VECOUT。
  - 起始地址需要 32 字节对齐。
- mask (Union[int, List[int]]): 掩码，用于控制在单次迭代中哪些元素参与搬运。
  - **连续模式** (当 mask 为 int): 表示从起始位置开始，连续搬运多少个元素。
    - 当数据类型为 16-bit (如 fp16) 时，取值范围是 [1, 128]。
    - 当数据类型为 32-bit (如 fp32) 时，取值范围是 [1, 64]。
  - **逐 bit 模式** (当 mask 为 List[int]): 掩码数组中的每个 bit 对应一个元素，bit 为 1 表示搬运，0 表示跳过。
    - 当数据类型为 16-bit 时，mask 是一个长度为 2 的列表，例如 mask=[mask0, mask1]。
    - 当数据类型为 32-bit 时，mask 是一个长度为 1 的列表，例如 mask=[mask0]。
- repeat_time (int): 重复迭代次数。矢量计算单元每次处理一个数据块（256字节），此参数指定了处理整个 Tensor 需要重复迭代的次数。
- repeat_params (asc.CopyRepeatParams): 控制地址步长的数据结构，用于处理高维或非连续数据。
  - dstStride, srcStride: 设置同一次迭代内，不同数据块（DataBlock）之间的地址步长。
  - dstRepeatSize, srcRepeatSize: 设置相邻两次迭代之间的地址步长。
- is_set_mask (bool, 可选): 模板参数，默认为 True。
  - True: 在接口内部设置 mask 值。
  - False: 在接口外部通过 asc.set_vector_mask 接口设置 mask，此时 mask 参数必须为占位符 asc.MASK_PLACEHOLDER。

**约束说明**

- 源操作数和目的操作数的起始地址需要保证32字节对齐。
- Copy和矢量计算API一样，支持和掩码操作API配合使用。但Counter模式配合高维切分计算API时，和通用的Counter模式有一定差异。具体差异如下：
  - 通用的Counter模式：Mask代表整个矢量计算参与计算的元素个数，迭代次数不生效。
  - Counter模式配合Copy高维切分计算API，Mask代表每次Repeat中处理的元素个数，迭代次数生效。

**调用示例**

```python
TILE_LENGTH = 1024
# 1. 定义源和目标 LocalTensor
src_tensor = asc.LocalTensor(asc.fp16, asc.Position.VECIN, size=TILE_LENGTH)
dst_tensor = asc.LocalTensor(asc.fp16, asc.Position.VECOUT, size=TILE_LENGTH)

...

# 2. 定义地址步长参数
# 示例：实现一个交错拷贝，源地址每次迭代跳 256 字节，目标地址连续
params = asc.CopyRepeatParams(
    dstStride=1,       # 迭代内，目标 datablock 连续
    srcStride=2,       # 迭代内，源 datablock 间隔为 1 个 datablock
    dstRepeatSize=8,   # 迭代间，目标地址步长为 8 个元素
    srcRepeatSize=16   # 迭代间，源地址步长为 16 个元素
)

# 3. 使用连续模式调用 Copy
# 每次迭代处理 128 个元素（一个 256 字节的 block），重复 4 次
asc.copy(dst_tensor, src_tensor, mask=128, repeat_time=4, repeat_params=params)
```
