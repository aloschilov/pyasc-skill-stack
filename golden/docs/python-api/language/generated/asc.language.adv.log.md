# asc.language.adv.log

### asc.language.adv.log(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), count: int | None = None, temp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor) | None = None, is_reuse_source: bool = False) → None

按元素以e为底做对数运算。

**对应的Ascend C函数原型**

```c++
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                    const uint32_t calCount)
```

```c++
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
```

**参数说明**

- is_reuse_source：是否允许修改源操作数。
- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。
- temp_buffer：临时缓存。
- count：参与计算的元素个数。

**约束说明**

- 不支持源操作数与目的操作数地址重叠。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

```python
asc.adv.log(dst, src)
```
