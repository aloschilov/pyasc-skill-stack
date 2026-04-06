# asc.language.basic.load_data

### asc.language.basic.load_data(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: LoadData2DParams) → None

### asc.language.basic.load_data(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), params: LoadData2DParams) → None

### asc.language.basic.load_data(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: LoadData2DParamsV2) → None

### asc.language.basic.load_data(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), params: LoadData2DParamsV2) → None

### asc.language.basic.load_data(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: LoadData3DParamsV1) → None

### asc.language.basic.load_data(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: LoadData3DParamsV2) → None

### asc.language.basic.load_data(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: LoadData3DParamsV2Pro) → None

源操作数/目的操作数的数据类型为uint8_t/int8_t时，分形矩阵大小在A1/A2上为16\*32， 在B1/B2上为32\*16。
源操作数/目的操作数的数据类型为uint16_t/int16_t/half/bfloat16_t时，分形矩阵在A1/B1/A2/B2上的大小为16\*16。
源操作数/目的操作数的数据类型为uint32_t/int32_t/float时，分形矩阵大小在A1/A2上为16\*8， 在B1/B2上为8\*16。
支持如下数据通路：
GM->A1; GM->B1; GM->A2; GM->B2;
A1->A2; B1->B2。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                const LocalTensor<T>& src,
                                const LoadData2DParams& loadDataParams)
```

```c++
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                const GlobalTensor<T>& src,
                                const LoadData2DParams& loadDataParams)
```

```c++
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                const LocalTensor<T>& src,
                                const LoadData2DParamsV2& loadDataParams)
```

```c++
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                const GlobalTensor<T>& src,
                                const LoadData2DParamsV2& loadDataParams)
```

```c++
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                const LocalTensor<T>& src,
                                const LoadData3DParamsV2Pro& loadDataParams)
```

```c++
template <typename T,
        const IsResetLoad3dConfig &defaultConfig = IS_RESER_LOAD3D_DEFAULT_CONFIG,
        typename U = PrimT<T>,
        typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                const LocalTensor<T>& src,
                                const LoadData3DParamsV1<U>& loadDataParams)
```

```c++
template <typename T,
        const IsResetLoad3dConfig &defaultConfig = IS_RESER_LOAD3D_DEFAULT_CONFIG,
        typename U = PrimT<T>,
        typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                const LocalTensor<T>& src,
                                const LoadData3DParamsV2<U>& loadDataParams)
```

**参数说明**

- dst：目的操作数，类型为 LocalTensor。
  - 作为二维数据加载的目标 Tensor。
  - 支持的 TPosition 为 VECIN/VECCALC/VECOUT。
  - 起始地址需要 32 字节对齐。
- src：源操作数，类型为 LocalTensor 或 GlobalTensor。
  - 当为 LocalTensor 时，表示在芯片内部不同本地存储单元之间按 2D 方式搬运。
  - 当为 GlobalTensor 时，表示从 Global Memory 按 2D 方式加载数据到 LocalTensor。
  - 元素数据类型需与 dst 保持一致。
- params：类型为下面结构体
  - LoadData2DParams 结构体
    - start_index：分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：start_index∈[0, 65535] 。单位：512B。默认为0。
    - repeat_times：迭代次数，每个迭代可以处理512B数据。取值范围：repeat_times∈[1, 255]。
    - src_stride：相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。
    - sid：预留参数，配置为0即可。
    - dst_gap：相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dst_gap∈[0, 65535]。默认为0。
    - if_transpose：是否启用转置功能，对每个分形矩阵进行转置，默认为false。
    - addr_mode：预留参数，配置为0即可。
  - LoadData2DParamsV2 结构体
    - m_start_position：M维起始位置，取值范围：m_start_position∈[0, 65535]。默认为0。
    - k_start_position：K维起始位置，取值范围：k_start_position∈[0, 65535]。默认为0。
    - m_step：M维步长，取值范围：m_step∈[0, 65535]。默认为0。
    - k_step：K维步长，取值范围：k_step∈[0, 65535]。默认为0。
    - src_stride：源操作数步长，取值范围：src_stride∈[-2147483648, 2147483647]。默认为0。
    - dst_stride：目的操作数步长，取值范围：dst_stride∈[0, 65535]。默认为0。
    - if_transpose：是否启用转置功能，默认为false。
    - sid：流ID，取值范围：sid∈[0, 255]。默认为0。
  - LoadData3DParamsV2Pro 结构体
    - channel_size：通道大小，取值范围：channel_size∈[0, 65535]。默认为0。
    - en_transpose：是否启用转置功能，默认为false。
    - en_small_k：是否启用小K优化，默认为false。
    - filter_size_w：是否启用滤波器宽度优化，默认为false。
    - filter_size_h：是否启用滤波器高度优化，默认为false。
    - f_matrix_ctrl：是否启用矩阵控制，默认为false。
    - ext_config：扩展配置，取值范围：ext_config∈[0, 18446744073709551615]。默认为0。
    - filter_config：滤波器配置，取值范围：filter_config∈[0, 18446744073709551615]。默认为0x10101010101。
  - LoadData3DParamsV1 结构体
    - pad_list：padding列表，顺序为[padding_left, padding_right, padding_top, padding_bottom]，每个元素取值范围：[0, 255]。
    - l1_h：源操作数height，取值范围：[1, 32767]。
    - l1_w：源操作数width，取值范围：[1, 32767]。
    - c1_index：卷积窗口在源Tensor C1维度的起点，取值范围：[0, 4095]。
    - fetch_filter_w：卷积窗口在filter W维度的起始位置，取值范围：[0, 254]。
    - fetch_filter_h：卷积窗口在filter H维度的起始位置，取值范围：[0, 254]。
    - left_top_w：卷积窗口在源Tensor W维度的起点，取值范围：[-255, 32767]。
    - left_top_h：卷积窗口在源Tensor H维度的起点，取值范围：[-255, 32767]。
    - stride_w：卷积核在W维的滑动步长，取值范围：[1, 63]。
    - stride_h：卷积核在H维的滑动步长，取值范围：[1, 63]。
    - filter_w：卷积核width，取值范围：[1, 255]。
    - filter_h：卷积核height，取值范围：[1, 255]。
    - dilation_filter_w：卷积核W维膨胀系数，取值范围：[1, 255]。
    - dilation_filter_h：卷积核H维膨胀系数，取值范围：[1, 255]。
    - jump_stride：迭代之间目的操作数地址递增步长，取值范围：[1, 127]。
    - repeat_mode：迭代模式，取值范围：[0, 1]，默认为0。
    - repeat_time：迭代次数，取值范围：[1, 255]。
    - c_size：通道展开优化控制参数，取值范围：[0, 1]，默认为0。
    - pad_value：padding填充值，需与src数据类型一致，默认为0。
  - LoadData3DParamsV2 结构体
    - pad_list：padding列表，顺序为[padding_left, padding_right, padding_top, padding_bottom]，每个元素取值范围：[0, 255]。
    - l1_h：源操作数height，取值范围：[1, 32767]。
    - l1_w：源操作数width，取值范围：[1, 32767]。
    - channel_size：通道大小，不同数据类型与平台存在对齐约束。
    - k_extension：K维扩展长度，取值范围：[1, 65535]。
    - m_extension：M维扩展长度，取值范围：[1, 65535]。
    - k_start_pt：K维起始位置，取值范围：[0, 65535]。
    - m_start_pt：M维起始位置，取值范围：[0, 65535]。
    - stride_w：卷积核在W维的滑动步长，取值范围：[1, 63]。
    - stride_h：卷积核在H维的滑动步长，取值范围：[1, 63]。
    - filter_w：卷积核width，取值范围：[1, 255]。
    - filter_h：卷积核height，取值范围：[1, 255]。
    - dilation_filter_w：卷积核W维膨胀系数，取值范围：[1, 255]。
    - dilation_filter_h：卷积核H维膨胀系数，取值范围：[1, 255]。
    - en_transpose：是否启用转置功能，取值为bool，默认为false。
    - pad_value：padding填充值，需与src数据类型一致，默认为0。
    - filter_size_w：是否在filterW基础上增加256元素，默认为false。
    - filter_size_h：是否在filterH基础上增加256元素，默认为false。
    - f_matrix_ctrl：FeatureMap矩阵控制开关，默认为false。

**约束说明**

- dst 与 src 的数据需要满足起始地址对齐要求，具体可查看文档。
- 不使用或者不想改变的配置，建议保持默认值，有助于性能提升。

**调用示例**

- Local Memory 内部 2D 搬运（Local -> Local）
  ```python
  @asc.jit
  def kernel_load_data_l2l(x: asc.GlobalAddress) -> None:
      x_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECIN,
                                addr=0, tile_size=512)
      y_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECOUT,
                                addr=0, tile_size=512)

      params = asc.LoadData2DParams(0, 4, 0, 0, 0, 0, 0)

      asc.load_data(y_local, x_local, params)
  ```
- Global Memory 到 Local Memory 的 2D 搬运（Global -> Local）
  ```python
  @asc.jit
  def kernel_load_data_g2l(x: asc.GlobalAddress) -> None:
      x_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECIN,
                                addr=0, tile_size=512)
      y_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECOUT,
                                addr=0, tile_size=512)

      x_gm = asc.GlobalTensor()
      x_gm.set_global_buffer(x)

      params = asc.LoadData2DParams(0, 4, 0, 0, 0, 0, 0)

      asc.load_data(y_local, x_local, params)
      asc.load_data(x_local, x_gm, params)
  ```
- Local Memory 内部 2D 搬运（V2版本，Local -> Local）
  ```python
  @asc.jit
  def kernel_load_data_l2l_v2(x: asc.GlobalAddress) -> None:
      x_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECIN,
                                addr=0, tile_size=512)
      y_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECOUT,
                                addr=0, tile_size=512)

      params_v2 = asc.LoadData2DParamsV2(0, 0, 16, 16, 0, 0, False, 0)

      asc.load_data(y_local, x_local, params_v2)
  ```
- Global Memory 到 Local Memory 的 2D 搬运（V2版本，Global -> Local）
  ```python
  @asc.jit
  def kernel_load_data_g2l_v2(x: asc.GlobalAddress) -> None:
      x_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECIN,
                                addr=0, tile_size=512)
      y_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECOUT,
                                addr=0, tile_size=512)

      x_gm = asc.GlobalTensor()
      x_gm.set_global_buffer(x)

      params_v2 = asc.LoadData2DParamsV2(0, 0, 16, 16, 0, 0, False, 0)

      asc.load_data(y_local, x_local, params_v2)
      asc.load_data(x_local, x_gm, params_v2)
  ```
- Local Memory 内部 3D 搬运（V2Pro版本，Local -> Local）
  ```python
  @asc.jit
  def kernel_load_data_3d_v2pro(x: asc.GlobalAddress) -> None:
      x_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECIN,
                                addr=0, tile_size=512)
      y_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECOUT,
                                addr=0, tile_size=512)

      params_3d_v2_pro = asc.LoadData3DParamsV2Pro(16, False, False, False, False, False, 0, 0x10101010101)

      asc.load_data(y_local, x_local, params_3d_v2_pro)
  ```
- Local Memory 内部 3D 搬运（LoadData3DParamsV1）
  ```python
      def test_load_data_v1(mock_launcher_run):

          @asc.jit
          def kernel_load_data_v1(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(
                  dtype=asc.float16,
                  pos=asc.TPosition.VECIN,
                  addr=0,
                  tile_size=512,
              )
              y_local = asc.LocalTensor(
                  dtype=asc.float16,
                  pos=asc.TPosition.VECOUT,
                  addr=0,
                  tile_size=512,
              )
              x_gm = asc.GlobalTensor()
              x_gm.set_global_buffer(x)

              params_3d_v1 = asc.LoadData3DParamsV1(
                  [0, 0, 0, 0],
                  16, 16,
                  0,
                  0, 0,
                  0, 0,
                  1, 1,
                  3, 3,
                  1, 1,
                  1,
                  0, 1,
                  0,
                  0,
              )

              asc.load_data(y_local, x_local, params_3d_v1)

          x = MockTensor(asc.float16)
          kernel_load_data_v1[1](x)
          assert mock_launcher_run.call_count == 1
  ```
- Local Memory 内部 3D 搬运（LoadData3DParamsV2）
  ```python
      def test_load_data_v2(mock_launcher_run):

          @asc.jit
          def kernel_load_data_v2(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(
                  dtype=asc.float16,
                  pos=asc.TPosition.VECIN,
                  addr=0,
                  tile_size=512,
              )
              y_local = asc.LocalTensor(
                  dtype=asc.float16,
                  pos=asc.TPosition.VECOUT,
                  addr=0,
                  tile_size=512,
              )
              x_gm = asc.GlobalTensor()
              x_gm.set_global_buffer(x)

              params_3d_v2 = asc.LoadData3DParamsV2(
                  [0, 0, 0, 0],
                  16, 16,
                  16,
                  16, 16,
                  0, 0,
                  1, 1,
                  3, 3,
                  1, 1,
                  False,
                  0,
                  False, False,
                  False,
              )

              asc.load_data(y_local, x_local, params_3d_v2)

          x = MockTensor(asc.float16)
          kernel_load_data_v2[1](x)
          assert mock_launcher_run.call_count == 1
  ```
