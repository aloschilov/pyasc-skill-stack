# asc.language.basic.set_cmp_mask

### asc.language.basic.set_cmp_mask(src: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

为select不传入mask参数的接口设置比较寄存器。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void SetCmpMask(const LocalTensor<T>& src)
```

**参数说明**

- src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。

**调用示例**

```python
src = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
asc.set_cmp_mask(src)
```
