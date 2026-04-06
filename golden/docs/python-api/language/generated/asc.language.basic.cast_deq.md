# asc.language.basic.cast_deq

### asc.language.basic.cast_deq(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), count: int, is_vec_deq: bool = True, half_block: bool = True) → None

### asc.language.basic.cast_deq(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: int, repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True, is_vec_deq: bool = True, half_block: bool = True) → None

### asc.language.basic.cast_deq(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True, is_vec_deq: bool = True, half_block: bool = True) → None

对输入做量化并进行精度转换。不同的数据类型，转换公式不同。

- 在输入类型为int16_t的情况下，对int16_t类型的输入做量化并进行精度转换，得到int8_t/uint8_t类型的数据。
  使用该接口前需要调用set_deq_scale设置scale、offset、sign_mode等量化参数。

  通过模板参数is_vec_deq控制是否选择向量量化模式。
  - 当is_vec_deq=false时，根据set_deq_scale设置的scale、offset、sign_mode，对输入做量化并进行精度转换。计算公式如下：
    dst[i] = (src[i] \* scale) + offset
  - 当is_vec_deq=true时，根据set_deq_scale设置的一段128B的UB上的16组量化参数scale[0]-scale[15]、offset[0]-offset[15]、sign_mode[0]-sign_mode[15]，
    以循环的方式对输入做量化并进行精度转换。计算公式如下：dst[i] = (src[i] \* scale[j]) + offset[j], 0<=j<=15
- 在输入类型为int32_t的情况下，对int32_t类型的输入做量化并进行精度转换，得到half类型的数据。使用该接口前需要调用set_deq_scale设置scale参数。
  dst[i] = src[i] \* scale

**对应的Ascend C函数原型**

- tensor前n个数据计算
  ```c++
  template <typename T, typename U, bool isVecDeq = true, bool halfBlock = true>
  __aicore__ inline void CastDeq(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint32_t count)
  ```
- tensor高维切分计算
  - mask逐bit模式
    ```c++
    template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
    __aicore__ inline void CastDeq(const LocalTensor<T>& dst, const LocalTensor<U>& src,
                            const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
    ```
  - mask连续模式
    ```c++
    template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
    __aicore__ inline void CastDeq(const LocalTensor<T>& dst, const LocalTensor<U>& src,
                            const int32_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
    ```

**参数说明**

- is_set_mask：是否在接口内部设置mask。
  - True，表示在接口内部设置mask。
  - False，表示在接口外部设置mask，开发者需要使用set_vector_mask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。
- is_vec_deq：控制是否选择向量量化模式。和set_deq_scale接口配合使用，当set_deq_scale接口传入Tensor时，is_vec_deq必须为true。
- half_block：对int16_t类型的输入做量化并进行精度转换得到int8_t/uint8_t类型的数据时，half_block参数用于指示输出元素存放在上半还是下半Block。half_block=True时，结果存放在下半Block；half_block=False时，结果存放在上半Block。
- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
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
- repeat_time：重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，
  为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeat_time表示迭代的次数。
- repeat_params：控制操作数地址步长的参数。UnaryRepeatParams类型，包含操作数相邻迭代间相同data_block的地址步长，操作数同一迭代内不同data_block的地址步长等参数。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。

**调用示例**

- tensor高维切分计算样例-mask连续模式
  ```python
  params = asc.UnaryRepeatParams(1, 1, 8, 8)
  asc.cast_deq(dst, src, mask=128, repeat_time=4, repeat_params=params)
  ```
- tensor高维切分计算样例-mask逐bit模式
  ```python
  uint64_max = 2**64 - 1
  mask = [uint64_max, uint64_max]
  params = asc.UnaryRepeatParams(1, 1, 8, 8)
  asc.cast_deq(dst, src, mask=mask, repeat_time=4, repeat_params=params)
  ```
- tensor前n个数据计算样例
  ```python
  asc.cast_deq(dst, src, count=512)
  ```
