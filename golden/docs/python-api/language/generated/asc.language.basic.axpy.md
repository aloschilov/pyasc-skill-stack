# asc.language.basic.axpy

### asc.language.basic.axpy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), scalar: int | float, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.axpy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), scalar: int | float, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.axpy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), scalar: int | float, count: int) → None

源操作数src中每个元素与标量求积后和目的操作数dst中的对应元素相加，计算公式如下：
dst[i] = src[i] \* scalar + dst[i]

**对应的Ascend C函数原型**

- tensor前n个数据计算
  ```c++
  template <typename T, typename U>
  __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, const int32_t& count)
  ```
- tensor高维切分计算
  - mask逐bit模式
    ```c++
    template <typename T, typename U, bool isSetMask = true>
    __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue,
                            uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
    ```
  - mask连续模式
    ```c++
    template <typename T, typename U, bool isSetMask = true>
    __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue,
                            uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
    ```

**参数说明**

- is_set_mask：是否在接口内部设置mask。
  - True，表示在接口内部设置mask。
  - False，表示在接口外部设置mask，开发者需要使用set_vector_mask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。
- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- scalar：源操作数，scalar标量。scalar的数据类型需要和src保持一致。
- count：参与计算的元素个数。
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
- repeat_time：重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，
  必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeat_time表示迭代的次数。
- repeat_params：控制操作数地址步长的参数。UnaryRepeatParams类型，包含操作数相邻迭代间相同data_block的地址步长，操作数同一迭代内不同data_block的地址步长等参数。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。

使用tensor高维切分计算接口时，src和scalar的数据类型为half、dst的数据类型为float的情况下，
一个迭代处理的源操作数元素个数需要和目的操作数保持一致，所以每次迭代选取前4个data_block参与计算。
设置repeat_stride参数和mask参数以及地址重叠时，需要考虑该限制。

**调用示例**

- tensor高维切分计算样例-mask连续模式
  ```python
  params = asc.UnaryRepeatParams(1, 1, 8, 8)
  asc.axpy(dst, src, 2.0, mask=128, repeat_time=4, repeat_params=params)
  ```
- tensor高维切分计算样例-mask逐bit模式
  ```python
  uint64_max = 2**64 - 1
  mask = [uint64_max, uint64_max]
  params = asc.UnaryRepeatParams(1, 1, 8, 8)
  asc.axpy(dst, src, 2.0, mask=mask, repeat_time=4, repeat_params=params)
  ```
- tensor前n个数据计算样例
  ```python
  asc.axpy(dst, src, 2.0, count=512)
  ```
