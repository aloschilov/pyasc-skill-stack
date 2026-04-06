# asc.language.basic.reduce_max

### asc.language.basic.reduce_max(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), shared_tmp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: int, repeat_time: int, src_rep_stride: int, cal_index: bool = False) → None

### asc.language.basic.reduce_max(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), shared_tmp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: List[int], repeat_time: int, src_rep_stride: int, cal_index: bool = False) → None

### asc.language.basic.reduce_max(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), shared_tmp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor), count: int, cal_index: bool = False) → None

在所有的输入数据中找出最小值及最小值对应的索引位置。

**对应的Ascend C函数原型**

- tensor前n个数据计算
  ```c++
  template <typename T>
  __aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0)
  ```
- tensor高维切分计算
  - mask逐比特模式
    ```c++
    template <typename T>
    __aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                      const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[],
                                      const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
    ```
  - mask连续模式
    ```c++
    template <typename T>
    __aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                      const LocalTensor<T>& sharedTmpBuffer, const int32_t mask,
                                      const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
    ```

**参数说明**

- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要保证4字节对齐（针对half数据类型），8字节对齐（针对float数据类型）。
- src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。源操作数的数据类型需要与目的操作数保持一致。
- shared_tmp_buffer：API执行期间，部分硬件型号需要一块空间用于存储中间结果，空间大小需要满足最小所需空间的要求。
  类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型需要与目的操作数保持一致。
- count：参与计算的元素个数。
  参数取值范围和操作数的数据类型有关，数据类型不同，能够处理的元素个数最大值不同，最大处理的数据量不能超过UB大小限制。
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
- repeat_time：迭代次数。与通用参数说明中不同的是，支持更大的取值范围，保证不超过int32_t最大值的范围即可。
- src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。
- cal_index：指定是否获取最小值的索引，bool类型，默认值为false，取值：
  - True：同时获取最小值和最小值索引。
  - False：不获取索引，只获取最小值。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。需要使用shared_tmp_buffer的情况下，支持dst与shared_tmp_buffer地址重叠（通常情况下dst比shared_tmp_buffer所需的空间要小），此时shared_tmp_buffer必须满足最小所需空间要求，否则不支持地址重叠。
- dst结果存储顺序为最大值，最大值索引，若不需要索引，只会存储最大值。返回结果中索引index数据是按照dst的数据类型进行存储的，比如dst使用half类型时，index按照half类型进行存储，如果按照half格式进行读取，index的值是不对的，因此index的读取需要使用reinterpret_cast方法转换到整数类型。
- 返回最大值索引时，如果存在多个最大值，返回第一个最大值的索引。
- 当输入类型是half的时候，只支持获取最大不超过65535（uint16_t能表示的最大值）的索引值。

**调用示例**

- tensor高维切分计算样例-mask连续模式
  ```python
  asc.reduce_max(dst, src, shared_tmp_buffer=shared_tmp, mask=128, repeat_time=128, src_rep_stride=65, cal_index=True)
  ```
- tensor高维切分计算样例-mask逐bit模式
  ```python
  uint64_max = 2**64 - 1
  mask = [uint64_max, uint64_max]
  asc.reduce_max(dst, src, shared_tmp_buffer=shared_tmp, mask=mask, repeat_time=65, src_rep_stride=8, cal_index=True)
  ```
- tensor前n个数据计算样例
  ```python
  asc.reduce_max(dst, src, shared_tmp_buffer=shared_tmp, count=2048, cal_index=True)
  ```
