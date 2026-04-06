# asc.language.adv.digamma

### asc.language.adv.digamma(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), count: int | None = None, temp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor) | None = None, is_reuse_source: bool = False) → None

按元素计算x的gamma函数的对数导数。

**对应的Ascend C函数原型**

```c++
template <typename T, bool isReuseSource = false>
__aicore__ inline void Digamma(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
```

```c++
template <typename T, bool isReuseSource = false>
__aicore__ inline void Digamma(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                    const LocalTensor<uint8_t>& sharedTmpBuffer)
```

```c++
template <typename T, bool isReuseSource = false>
__aicore__ inline void Digamma(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                    const uint32_t calCount)
```

```c++
template <typename T, bool isReuseSource = false>
__aicore__ inline void Digamma(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
```

**参数说明**

- is_reuse_source：是否允许修改源操作数。
- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。
- temp_buffer：临时缓存。
- count：参与计算的元素个数。

**约束说明**

- 不支持源操作数与目的操作数地址重叠。
- 不支持temp_buffer与源操作数和目的操作数地址重叠。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

```python
pipe = asc.Tpipe()
tmp_que = asc.TQue(asc.TPosition.VECCALC, 1)
pipe.init_buffer(que=tmp_que, num=1, len=buffer_size)   # buffer_size 通过Host侧tiling参数获取
shared_tmp_buffer = tmp_que.alloc_tensor(asc.uint8)
# 输入tensor长度为1024，算子输入的数据类型为half，实际计算个数为512
asc.adv.Digamma(dst, src, count=512, temp_buffer=shared_tmp_buffer)
```
