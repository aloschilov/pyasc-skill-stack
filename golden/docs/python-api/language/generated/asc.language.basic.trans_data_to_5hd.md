# asc.language.basic.trans_data_to_5hd

### asc.language.basic.trans_data_to_5hd(dst_list: TensorList, src_list: TensorList, params: TransDataTo5HDParams) → None

### asc.language.basic.trans_data_to_5hd(dst_list: AddrList, src_list: AddrList, params: TransDataTo5HDParams) → None

### asc.language.basic.trans_data_to_5hd(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: TransDataTo5HDParams) → None

数据格式转换，一般用于将NCHW格式转换成NC1HWC0格式，也可用于二维矩阵数据块的转置。
相比于transpose接口，本接口单次repeat内可处理512Byte的数据（16个datablock），
支持不同shape的矩阵转置，还可以支持多次repeat操作。

**对应的Ascend C函数原型**

```c++
// 使用LocalTensor数组版本
template <typename T>
__aicore__ inline void TransDataTo5HD(const LocalTensor<T> (&dstList)[NCHW_CONV_ADDR_LIST_SIZE],
                                    const LocalTensor<T> (&srcList)[NCHW_CONV_ADDR_LIST_SIZE],
                                    const TransDataTo5HDParams& nchwconvParams)

// 使用地址值数组版本（性能更优）
template<typename T>
__aicore__ inline void TransDataTo5HD(uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE],
                                    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE],
                                    const TransDataTo5HDParams& nchwconvParams)

// 使用连续存储地址值版本
template <typename T>
__aicore__ inline void TransDataTo5HD(const LocalTensor<uint64_t>& dst,
                                    const LocalTensor<uint64_t>& src,
                                    const TransDataTo5HDParams& nchwconvParams)
```

**参数说明**

- dst_or_list：目的操作数地址序列，类型为LocalTensor数组、地址值数组或连续存储地址值的LocalTensor
- src_or_list：源操作数地址序列，类型与dst_or_list对应，数据类型需要与目的操作数保持一致
- params：控制参数结构体，包含读取写入位置控制、迭代次数、地址步长等参数
  - dst_high_half：指定数据存储到datablock的高半部还是低半部（仅支持int8_t/uint8_t）
  - src_high_half：指定数据从datablock的高半部还是低半部读取（仅支持int8_t/uint8_t）
  - repeat_times：重复迭代次数，取值范围[0,255]
  - dst_rep_stride：相邻迭代间目的操作数相同datablock地址步长
  - src_rep_stride：相邻迭代间源操作数相同datablock地址步长

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 普通转置接口支持src和dst复用。
- 增强转置接口，transposeType为TRANSPOSE_ND2ND_B16时支持src和dst复用，transposeType为TRANSPOSE_NCHW2NHWC、TRANSPOSE_NHWC2NCHW时不支持src和dst复用。

**调用示例**

此接口通过不同的方式构造源和目的操作数序列，以实现灵活的数据重组。

- dst_list, src_list：定义了源数据块和目标数据块。它们可以是包含 LocalTensor 物理地址的 list/tuple，
  也可以是包含 LocalTensor 视图对象的 list/tuple，或者是将地址值连续存储的 LocalTensor<uint64_t>。
  ```python
  params = asc.TransDataTo5HDParams(
      dst_high_half=False,
      src_high_half=False,
      repeat_times=4,
      dst_rep_stride=8,
      src_rep_stride=8
  )

  asc.trans_data_to_5hd(dst_list, src_list, params)
  ```
