# asc.language.basic.load_data_with_transpose

### asc.language.basic.load_data_with_transpose(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: LoadData2dTransposeParams) → None

### asc.language.basic.load_data_with_transpose(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: LoadData2dTransposeParamsV2) → None

该接口实现带转置的2D格式数据从A1/B1到A2/B2的加载。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dst,
                                             const LocalTensor<T>& src,
                                             const LoadData2dTransposeParams& loadDataParams)
```

```c++
template <typename T>
__aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dst,
                                             const LocalTensor<T>& src,
                                             const LoadData2dTransposeParamsV2& loadDataParams)
```

**参数说明**

- dst：目的操作数，类型为 LocalTensor。
  - 用于接收转置后的二维数据。
  - 存储位置需属于 VECIN / VECCALC / VECOUT 中的一种。
  - 起始地址需满足 32 字节对齐要求。
- src：源操作数，类型为 LocalTensor。
  - 作为 2D 输入块的提供者。
  - 仅支持 Local → Local（A1/B1 → A2/B2），不支持 GlobalTensor。
  - 数据类型必须与 dst 一致。
- params：二维转置加载参数，类型为 LoadData2dTransposeParams 或 LoadData2dTransposeParamsV2。
  - LoadData2dTransposeParams 结构体
    - start_index：方块矩阵ID，搬运起始位置为源操作数中第几个方块矩阵（0 为源操作数中第1个方块矩阵）。取值范围：start_index∈[0, 65535] 。默认为0。
    - repeat_times：迭代次数，取值范围：repeat_times∈[0, 255]。默认为0。
    - src_stride：相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔。这里的单位实际上是拼接后的方块矩阵的大小。取值范围：src_stride∈[0, 65535]。默认为0。
    - dst_gap：相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔，单位：512B。取值范围：dst_gap∈[0, 65535]。默认为0。
    - dst_frac_gap：每个迭代内目的操作数转置前一个分形结束地址与后一个分形起始地址的间隔，单位：512B，仅在数据类型为float/int32_t/uint32_t/uint8_t/int8_t/int4b_t时有效。取值范围：dst_frac_gap∈[0, 65535]。默认为0。
    - addr_mode：预留参数
  - LoadData2dTransposeParamsV2 结构体
    - start_index：方块矩阵ID，搬运起始位置为源操作数中第几个方块矩阵（0 为源操作数中第1个方块矩阵）。取值范围：start_index∈[0, 65535] 。默认为0。
    - repeat_times：迭代次数，取值范围：repeat_times∈[0, 255]。默认为0。
    - src_stride：源操作数步长，取值范围：src_stride∈[0, 65535]。默认为0。
    - dst_gap：目的操作数间隔，取值范围：dst_gap∈[0, 65535]。默认为0。
    - dst_frac_gap：分形间隔，取值范围：dst_frac_gap∈[0, 65535]。默认为0。
    - src_frac_gap：源分形间隔，取值范围：src_frac_gap∈[0, 65535]。默认为0。
    - addr_mode：地址模式，取值范围：addr_mode∈[0, 255]。默认为0。

**约束说明**

- repeatTimes 为 0 时表示不执行搬运操作。
- 开发者需要保证目的操作数转置后的分形没有重叠。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- repeat_times 为 0 时表示不执行搬运操作。
- 开发者需要保证目的操作数转置后的分形没有重叠。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

- 调用示例（V1版本）
  ```python
  @asc.jit
  def kernel_load_data_with_transpose(x: asc.GlobalAddress) -> None:
      x_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECIN,
                                addr=0, tile_size=512)

      y_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECOUT,
                                addr=0, tile_size=512)

      params = asc.LoadData2dTransposeParams(0, 4, 0, 0, 0, 0)

      asc.load_data_with_transpose(y_local, x_local, params)
  ```
- 调用示例（V2版本）
  ```python
  @asc.jit
  def kernel_load_data_with_transpose_v2(x: asc.GlobalAddress) -> None:
      x_local = asc.LocalTensor(dtype=asc.float16,
                              pos=asc.TPosition.VECIN,
                              addr=0, tile_size=512)

      y_local = asc.LocalTensor(dtype=asc.float16,
                              pos=asc.TPosition.VECOUT,
                              addr=0, tile_size=512)

      params_v2 = asc.LoadData2dTransposeParamsV2(0, 4, 0, 0, 0, 0, 0)

      asc.load_data_with_transpose(y_local, x_local, params_v2)
  ```
