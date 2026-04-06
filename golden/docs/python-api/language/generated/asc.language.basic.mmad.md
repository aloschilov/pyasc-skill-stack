# asc.language.basic.mmad

### asc.language.basic.mmad(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), fm: [LocalTensor](../core.md#asc.language.core.LocalTensor), filter: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: MmadParams) → None

### asc.language.basic.mmad(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), fm: [LocalTensor](../core.md#asc.language.core.LocalTensor), filter: [LocalTensor](../core.md#asc.language.core.LocalTensor), bias: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: MmadParams) → None

完成矩阵乘加（C += A \* B）操作。矩阵ABC分别为A2/B2/CO1中的数据。
ABC矩阵的数据排布格式分别为ZZ，ZN，NZ。

**对应的 Ascend C 函数原型**

```c++
template <typename T, typename U, typename S>
__aicore__ inline void Mmad(const LocalTensor<T>& dst,
                            const LocalTensor<U>& fm,
                            const LocalTensor<S>& filter,
                            const MmadParams& mmadParams)
```

```c++
template <typename T, typename U, typename S, typename V>
__aicore__ inline void Mmad(const LocalTensor<T>& dst,
                            const LocalTensor<U>& fm,
                            const LocalTensor<S>& filter,
                            const LocalTensor<V>& bias,
                            const MmadParams& mmadParams)
```

**参数说明**

- dst：结果输出 Tensor，类型为 LocalTensor。
  - 用于存放矩阵乘累加的结果。
  - **必须位于 CO1 存储位置（TPosition.CO1）**。
  - 元素数据类型需与累加结果类型匹配。
- fm：左矩阵（A 矩阵）输入，类型为 LocalTensor。
  - 表示矩阵乘法中的左操作数。
  - **必须位于 A2 存储位置（TPosition.A2）**。
  - 需要按照满足 Mmad 格式要求的 A2 布局存储。
- filter：右矩阵（B 矩阵）输入，类型为 LocalTensor。
  - 表示矩阵乘法中的右操作数。
  - **必须位于 B2 存储位置（TPosition.B2）**。
  - 需要按照符合指令格式的 B2 分块布局排布。
- bias（可选）：偏置项，类型为 LocalTensor。
  - 用于执行 dst += fm × filter + bias 的计算。
  - 当提供 bias 时，将使用带偏置版本的指令。
- params：MmadParams 类型的矩阵乘参数。
  - m：左矩阵Height，取值范围：m∈[0, 4095] 。默认值为0。
  - n：右矩阵Width，取值范围：n∈[0, 4095] 。默认值为0。
  - k：左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095] 。默认值为0。
  - cmatrixInitVal：配置C矩阵初始值是否为0。默认值true。
  - cmatrixSource：配置C矩阵初始值是否来源于C2（存放Bias的硬件缓存区）。默认值为false。
  - isBias：该参数废弃，新开发内容不要使用该参数。
  - fmOffset：预留参数。
  - enSsparse：预留参数。
  - enWinogradA：预留参数。
  - enWinogradB：预留参数。
  - unitFlag：预留参数。
  - kDirectionAlign：预留参数。

**约束说明**

- dst只支持位于CO1，fm只支持位于A2，filter只支持位于B2。
- 当M、K、N中的任意一个值为0时，该指令不会被执行。
- 当M = 1时，会默认开启GEMV（General Matrix-Vector Multiplication）功能。在这种情况下，Mmad API从L0A Buffer读取数据时，会以ND格式进行读取，而不会将其视为ZZ格式。所以此时左矩阵需要直接按照ND格式进行排布。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

- 基本 Mmad（无 bias）
  ```python
  @asc.jit
  def kernel_mmad_basic():
      dst = asc.LocalTensor(dtype=asc.float16,
                            pos=asc.TPosition.CO1,
                            addr=0, tile_size=1024)

      fm = asc.LocalTensor(dtype=asc.float16,
                           pos=asc.TPosition.A2,
                           addr=0, tile_size=1024)

      filter = asc.LocalTensor(dtype=asc.float16,
                               pos=asc.TPosition.B2,
                               addr=0, tile_size=1024)

      params = asc.MmadParams(4, 4, 4)

      asc.mmad(dst, fm, filter, params)
  ```
- Mmad 带 bias
  ```python
  @asc.jit
  def kernel_mmad_bias():
      dst = asc.LocalTensor(dtype=asc.float16,
                            pos=asc.TPosition.CO1,
                            addr=0, tile_size=1024)

      fm = asc.LocalTensor(dtype=asc.float16,
                           pos=asc.TPosition.A2,
                           addr=0, tile_size=1024)

      filter = asc.LocalTensor(dtype=asc.float16,
                               pos=asc.TPosition.B2,
                               addr=0, tile_size=1024)

      bias = asc.LocalTensor(dtype=asc.float16,
                             pos=asc.TPosition.VECIN,
                             addr=0, tile_size=1024)

      params = asc.MmadParams(4, 4, 4)

      asc.mmad(dst, fm, filter, bias, params)
  ```
