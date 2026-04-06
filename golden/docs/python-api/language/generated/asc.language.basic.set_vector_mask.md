# asc.language.basic.set_vector_mask

### asc.language.basic.set_vector_mask(length: int, dtype: DataType, mode: MaskMode) → None

### asc.language.basic.set_vector_mask(mask_high: int, mask_low: int, dtype: DataType, mode: MaskMode) → None

用于在矢量计算时设置mask。使用前需要先调用 set_mask_count/set_mask_norm 设置 mask 模式。
在不同模式下，mask的含义不同：

- **Normal 模式**

  mask参数用来控制单次迭代内参与计算的元素个数。此时又可以划分为如下两种模式：
  - **连续模式（len）**：表示单次迭代中前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
    - 操作数为16位时：mask ∈ [1, 128]
    - 操作数为32位时：mask ∈ [1, 64]
    - 操作数为64位时：mask ∈ [1, 32]
  - **逐比特模式（mask_high / mask_low）**：按位控制参与计算的元素，bit位的值为1表示参与计算，0表示不参与。

    分为mask_high（高位mask）和mask_low（低位mask）。参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
    - 操作数为16位时：mask_low、mask_high ∈ [0, 2⁶⁴-1]，并且不同时为 0
    - 操作数为32位时：mask_high = 0，mask_low ∈ (0, 2⁶⁴-1]
    - 操作数为64位时：mask_high = 0，mask_low ∈ (0, 2³²-1]
- **Counter 模式**

  mask参数表示整个矢量计算参与计算的元素个数。

**对应的Ascend C函数原型**

```c++
template <typename T, MaskMode mode = MaskMode::NORMAL>
__aicore__ static inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow);
```

```c++
template <typename T, MaskMode mode = MaskMode::NORMAL>
__aicore__ static inline void SetVectorMask(int32_t len);
```

**参数说明**

- mask_high
  - Normal模式：对应Normal模式下的逐比特模式，可以按位控制哪些元素参与计算。传入高位mask值。
  - Counter模式：需要置0，本入参不生效。
- mask_low
  - Normal模式：对应Normal模式下的逐比特模式，可以按位控制哪些元素参与计算。传入低位mask值。
  - Counter模式：整个矢量计算过程中，参与计算的元素个数。
- len
  - Normal模式：对应Normal模式下的mask连续模式，表示单次迭代内表示前面连续的多少个元素参与计算。
  - Counter模式：整个矢量计算过程中，参与计算的元素个数。
- dtype：矢量计算操作数的数据类型，由 Python 前端显式指定，用于推导 C++ 模板参数 T。
- mode：
  mask 模式，类型为 MaskMode 枚举值
  - asc.MaskMode.NORMAL：Normal 模式，支持连续模式与逐比特模式。
  - asc.MaskMode.COUNTER：Counter 模式，mask 参数表示整个矢量计算参与的总元素个数。

**约束说明**

该接口仅在矢量计算API的isSetMask模板参数为false时生效，使用完成后需要使用ResetMask将mask恢复为默认值。

**调用示例**

- Counter 模式：整个计算中参与 128 个元素
  ```python
  len = 128
  asc.set_mask_count()
  asc.set_vector_mask(len, dtype=asc.float16, mode=asc.MaskMode.COUNTER)
  asc.reset_mask()
  ```
- Normal 模式（逐bit模式）：使用 bitmask 控制参与计算的元素
  ```python
  mask_high = 2**64 - 1
  mask_low = 2**64 - 1
  asc.set_mask_norm()
  asc.set_vector_mask(mask_high, mask_low, dtype=asc.float16, mode=asc.MaskMode.NORMAL)
  asc.reset_mask()
  ```
- Normal 模式（连续模式）：前 64 个元素参与每次迭代计算
  ```python
  len = 64
  asc.set_mask_norm()
  asc.set_vector_mask(len, dtype=asc.float32, mode=asc.MaskMode.NORMAL)
  asc.reset_mask()
  ```
