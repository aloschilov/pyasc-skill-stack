# asc.language.basic.get_cmp_mask

### asc.language.basic.get_cmp_mask(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

用于获取compare（结果存入寄存器）指令的比较结果。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void GetCmpMask(const LocalTensor<T>& dst)
```

**参数说明**

- dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。

**约束说明**

- dst的空间大小不能少于128字节。

**调用示例**

```python
dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
asc.get_cmp_mask(dst)
```
