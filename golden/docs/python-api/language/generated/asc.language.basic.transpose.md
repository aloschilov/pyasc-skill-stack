# asc.language.basic.transpose

### asc.language.basic.transpose(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

### asc.language.basic.transpose(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), shared_tmp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: TransposeParamsExt) → None

用于实现16 \* 16的二维矩阵数据块转置或者[N,C,H,W]与[N,H,W,C]数据格式互相转换。

**对应的Ascend C函数原型**

```c++
// 普通转置，支持16 * 16的二维矩阵数据块进行转置
template <typename T>
__aicore__ inline void Transpose(const LocalTensor<T>& dst, const LocalTensor<T>& src)

// 增强转置，支持16 * 16的二维矩阵数据块转置，支持[N,C,H,W]与[N,H,W,C]互相转换
template <typename T>
__aicore__ inline void Transpose(const LocalTensor<T>& dst, const LocalTensor<T> &src,
                                const LocalTensor<uint8_t> &sharedTmpBuffer,
                                const TransposeParamsExt &transposeParams)
```

**参数说明**

- dst: 目的操作数，类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT，起始地址需要32字节对齐
- src: 源操作数，类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT，起始地址需要32字节对齐，数据类型需要与dst保持一致
- shared_tmp_buffer: 共享的临时Buffer，大小根据transposeType确定
- params: 控制Transpose的数据结构，包含输入的shape信息和transposeType参数

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 普通转置接口支持src和dst复用。
- 增强转置接口，transposeType为TRANSPOSE_ND2ND_B16时支持src和dst复用，transposeType为TRANSPOSE_NCHW2NHWC、TRANSPOSE_NHWC2NCHW时不支持src和dst复用。

**调用示例**

- 基础转置样例
  ```python
  pipe = asc.TPipe()
  in_queue_x = asc.TQue(asc.TPosition.VECIN, buffer_num)
  out_queue_z = asc.TQue(asc.TPosition.VECOUT, buffer_num)
  ...
  x_local = in_queue_x.alloc_tensor(asc.float16)
  z_local = out_queue_z.alloc_tensor(asc.float16)
  asc.transpose(z_local, x_local)
  ```
- 增强转置样例
  ```python
  pipe = asc.TPipe()
  in_queue_x = asc.TQue(asc.TPosition.VECIN, buffer_num)
  out_queue_z = asc.TQue(asc.TPosition.VECOUT, buffer_num)
  in_queue_tmp = asc.TQue(asc.TPosition.VECIN, buffer_num)
  ...
  x_local = in_queue_x.alloc_tensor(asc.float16)
  z_local = out_queue_z.alloc_tensor(asc.float16)
  tmp_buffer = in_queue_tmp.alloc_tensor(asc.uint8)

  params = asc.TransposeParamsExt(
      n_size=1,
      c_size=16,
      h_size=4,
      w_size=4,
      transpose_type=asc.TransposeType.TRANSPOSE_NCHW2NHWC
  )

  asc.transpose(z_local, x_local, tmp_buffer, params)
  ```
