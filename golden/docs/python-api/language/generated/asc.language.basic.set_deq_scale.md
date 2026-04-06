# asc.language.basic.set_deq_scale

### asc.language.basic.set_deq_scale(scale: float) → None

### asc.language.basic.set_deq_scale(scale: float, offset: int, sign_mode: bool) → None

### asc.language.basic.set_deq_scale(vdeq: LocalTensor, vdeq_info: VdeqInfo) -> None:

设置DEQSCALE寄存器的值。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetDeqScale(half scale)

__aicore__ inline void SetDeqScale(float scale, int16_t offset, bool signMode)

template <typename T>
__aicore__ inline void SetDeqScale(const LocalTensor<T>& vdeq, const VdeqInfo& vdeqInfo)
```

**参数说明**

- scale(half)：scale量化参数，half类型。
- scale(float)：scale量化参数，float类型。
- offset：offset量化参数，int16_t类型，只有前9位有效。
- sign_mode：bool类型，表示量化结果是否带符号。
- vdeq: 输入量化tensor，大小为128Byte。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- vdeq_info: 存储量化tensor信息的数据结构，结构体内包含量化tensor中的16组量化参数。

**调用示例**

```python
# Cast
scale = 1.0
asc.set_deq_scale(scale)
asc.cast(cast_dst_local, cast_dsrc_local, asc.RoundMode.CAST_NONE, src_size)
```
```python
# CastDeq
scale = 1.0
offset = 0
sign_mode = True
asc.set_deq_scale(scale, offset, sign_mode)
asc.cast_deq(dst_local, src_local, count=src_size, is_vec_deq=False, half_block=False)
```
```python
# CastVdeq
vdeq_local = asc.LocalTensor(dtype=asc.uint64, pos=asc.TPosition.VECIN, addr=0, tile_size=16)
vdeq_scale = [1.0] * 16
vdeq_offset = [0] * 16
vdeq_sign_mode = [False] * 16
vdeq_info = asc.VdeqInfo(vdeq_scale, vdeq_offset, vdeq_sign_mode)
asc.set_deq_scale(vdeq_local, vdeq_info)
asc.cast_deq(dst_local, src_local, count=src_size, is_vec_deq=True, half_block=False)
```
