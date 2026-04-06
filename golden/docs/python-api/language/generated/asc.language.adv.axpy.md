# asc.language.adv.axpy

### asc.language.adv.axpy(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), scalar: float | int, count: int | None = None, temp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor) | None = None, is_reuse_source: bool = False) → None

源操作数(srcTensor)中每个元素与标量求积后和目的操作数(dstTensor)中的对应元素相加。
该接口功能同基础API Axpy，区别在于此接口指令是通过Muls和Add组合计算，从而提供更优的精度。

**对应的Ascend C函数原型**

```c++
template <typename T, typename U, bool isReuseSource = false>
__aicore__ inline void Axpy(const LocalTensor<T>& dstTensor, const LocalTensor<U>& srcTensor,
    const U scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
```

**参数说明**

- is_reuse_source：是否允许修改源操作数，默认值为false。
- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- scalar：scalar标量。支持的数据类型为：half/float。scalar操作数的类型需要和srcTensor保持一致。
- temp_buffer：临时缓存。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- count：参与计算的元素个数。

**约束说明**

- 不支持源操作数与目的操作数地址重叠。
- 不支持temp_buffer与源操作数和目的操作数地址重叠。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 该接口支持的精度组合如下：
  - half精度组合：src_local数据类型=half；scalar数据类型=half；dst_local数据类型=half；PAR=128
  - float精度组合：src_local数据类型=float；scalar数据类型=float；dst_local数据类型=float；PAR=64
  - mix精度组合：src_local数据类型=half；scalar数据类型=half；dst_local数据类型=float；PAR=64

**调用示例**

```python
pipe = asc.Tpipe()
tmp_que = asc.TQue(asc.TPosition.VECCALC, 1)
pipe.init_buffer(que=tmp_que, num=1, len=buffer_size)   # buffer_size 通过Host侧tiling参数获取
shared_tmp_buffer = tmp_que.alloc_tensor(asc.uint8)
# 输入tensor长度为1024，算子输入的数据类型为half，实际计算个数为512
asc.adv.axpy(dst, src, 3.0, count=512, temp_buffer=shared_tmp_buffer)
```
