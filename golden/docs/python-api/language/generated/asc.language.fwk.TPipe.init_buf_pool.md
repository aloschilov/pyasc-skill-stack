# asc.language.fwk.TPipe.init_buf_pool

#### TPipe.init_buf_pool(buf_pool: [TBufPool](../fwk.md#asc.language.fwk.TBufPool), len: int = 0, share_buf: [TBufPool](../fwk.md#asc.language.fwk.TBufPool) = None) → None

初始化TBufPool内存资源池。本接口适用于内存资源有限时，希望手动指定UB/L1内存资源复用的场景。本接口初始化后在整体内存资源中划分出一块子资源池。

**对应的Ascend C函数原型**

```c++
template <class T>
__aicore__ inline bool InitBufPool(T& bufPool, uint32_t len)
template <class T, class U>
__aicore__ inline bool InitBufPool(T& bufPool, uint32_t len, U& shareBuf)
```

**参数说明**

- buf_pool：新划分的资源池，类型为TBufPool。
- len：新划分资源池长度，单位为Byte，非32Bytes对齐会自动补齐至32Bytes对齐。
- share_buf：被复用资源池，类型为TBufPool，新划分资源池与被复用资源池共享起始地址及长度。

**约束说明**

- 新划分的资源池与被复用资源池的硬件属性需要一致，两者共享起始地址及长度；
- 输入长度需要小于等于被复用资源池长度；
- 其他泛用约束参考TBufPool。

**调用示例**

```python
src0_global.set_global_buffer(src0_gm)
src1_global.set_global_buffer(src1_gm)
dst_global.set_global_buffer(dst_gm)
pipe.init_buf_pool(tbuf_pool1, 196608)
pipe.init_buf_pool(tbuf_pool2, 196608, tbuf_pool1)
```
