# asc.language.fwk.TBufPool.init_buf_pool

#### TBufPool.init_buf_pool(buf_pool: [TBufPool](../fwk.md#asc.language.fwk.TBufPool), len: int = 0, share_buf: [TBufPool](../fwk.md#asc.language.fwk.TBufPool) = None) → None

通过Tpipe::InitBufPool接口可划分出整块资源，整块TbufPool资源可以继续通过TBufPool.init_buf_pool接口划分成小块资源。

**对应的Ascend C函数原型**

```c++
template <class T>
__aicore__ inline bool InitBufPool(T& bufPool, uint32_t len)
```

```c++
template <class T, class U>
__aicore__ inline bool InitBufPool(T& bufPool, uint32_t len, U& shareBuf)
```

**参数说明**

- size：需要获取的Tensor元素个数。
- buf_offset：从起始位置的偏移长度，单位是字节，且需32字节对齐。

**返回值说明**

获取到的LocalTensor。

**约束说明**

- 新划分的资源池与被复用资源池的物理内存需要一致，两者共享起始地址及长度；
- 输入长度需要小于等于被复用资源池长度；
- 其他泛用约束参考TBufPool

**调用示例**

```python
@asc.jit
def init(src0_gm: asc.GlobalAddress, src1_gm: asc.GlobalAddress, dst_gm: asc.GlobalAddress):
    src0_global.set_global_buffer(src0_gm);
    src1_global.set_global_buffer(src1_gm);
    dst_global.set_global_buffer(dst_gm);
    pipe.init_buf_pool(tbuf_pool0, 131072);
    tbuf_pool0.init_buffer(que=src_que0, num=1, len=65536); // Total src0
    tbuf_pool0.init_buf_pool(tbuf_pool1, 65536);
    tbuf_pool0.init_buf_pool(tbuf_pool2, 65536, tbuf_pool1);

@asc.jit
def Process():
    tbuf_pool1.init_buffer(que=src_que1, num=1, len=32768)
    tbuf_pool1.init_buffer(que=dst_que0, num=1, len=32768)
    copy_in()
    compute()
    copy_out()
    tbuf_pool1.reset()
    tbuf_pool2.init_buffer(src_que2, num=1, len=32768)
    tbuf_pool2.init_buffer(dst_que1, num=1, len=32768)
    copy_in1()
    compute1()
    copy_out1()
    tbuf_pool2.reset()
    tbuf_pool0.reset()
    pipe.reset()
```
