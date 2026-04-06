# asc.language.adv.xor

### asc.language.adv.xor(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), count: int | None = None, temp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor) | None = None, is_reuse_source: bool = False) → None

按元素执行Xor运算。

**对应的Ascend C函数原型**

```c++
template <typename T, bool isReuseSource = false>
__aicore__ inline void Xor(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)

template <typename T, bool isReuseSource = false>
 __aicore__ inline void Xor(const LocalTensor<T>& dstTensor, const LocalTensor<T> &src0Tensor,
    const LocalTensor<T> &src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)

template <typename T, bool isReuseSource = false>
__aicore__ inline void Xor(const LocalTensor<T> &dstTensor, const LocalTensor<T> &src0Tensor,
    const LocalTensor<T> &src1Tensor, const uint32_t calCount)

template <typename T, bool isReuseSource = false>
__aicore__ inline void Xor(const LocalTensor<T> &dstTensor, const LocalTensor<T> &src0Tensor,
    const LocalTensor<T> &src1Tensor)
```

**参数说明**

- is_reuse_source：是否允许修改源操作数，默认值为false。
- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src0：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。
- src1：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。
- temp_buffer：临时内存空间。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- count：参与计算的元素个数。

**约束说明**

- 不支持源操作数与目的操作数地址重叠。
- 当前仅支持ND格式的输入，不支持其他格式。
- count需要保证小于或等于src0Tensor和src1Tensor和dstTensor存储的元素范围。
- 对于不带count参数的接口，需要保证src0Tensor和src1Tensor的shape大小相等。
- 不支持temp_buffer与源操作数和目的操作数地址重叠。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

..code-block:: python

> asc.adv.xor(z_local, x_local, y_local)
