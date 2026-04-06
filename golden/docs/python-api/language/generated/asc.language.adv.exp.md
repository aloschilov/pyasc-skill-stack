# asc.language.adv.exp

### asc.language.adv.exp(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), count: int, taylor_expand_level: int, temp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor) | None = None, is_reuse_source: bool = False) → None

按元素取自然指数，用户可以选择是否使用泰勒展开公式进行计算。

**对应的Ascend C函数原型**

```c++
template <typename T, uint8_t taylorExpandLevel, bool isReuseSource = false>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                            const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)

template <typename T, uint8_t taylorExpandLevel, bool isReuseSource = false>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                            const uint32_t calCount)
```

**参数说明**

- taylor_expand_level：泰勒展开项数，项数为0表示不使用泰勒公式进行计算。项数太少时，精度会有一定误差。项数越多，精度相对而言更高，但是性能会更差。
- is_reuse_source：是否允许修改源操作数，默认值为false。该参数仅在输入的数据类型为float时生效。
- dst：目的操作数。
- src：源操作数。
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
asc.adv.exp(dst, src, count=512, taylor_expand_level=0, temp_buffer=shared_tmp_buffer)
```
