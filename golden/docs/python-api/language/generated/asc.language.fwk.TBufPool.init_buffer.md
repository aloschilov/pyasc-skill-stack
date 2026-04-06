# asc.language.fwk.TBufPool.init_buffer

#### TBufPool.init_buffer(que: [TQue](../fwk.md#asc.language.fwk.TQue), num: int = 0, len: int = 0) → None

#### TBufPool.init_buffer(buf: [TBuf](../fwk.md#asc.language.fwk.TBuf), num: int = 0) → None

调用TBufPool::InitBuffer接口为TQue/TBuf进行内存分配。

**对应的Ascend C函数原型**

```c++
template <class T> __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)
template <TPosition pos> __aicore__ inline bool InitBuffer(TBuf<pos>& buf, uint32_t len)
```

**参数说明**

- pos：Buffer逻辑位置，可以为VECIN、VECOUT、VECCALC、A1、B1、C1。
- que：需要分配内存的TQue对象。
- num：分配内存块的个数。
- len：每个内存块的大小，单位为Bytes，非32Bytes对齐会自动向上补齐至32Bytes对齐。
- buf：需要分配内存的TBuf对象。
- len：为TBuf分配的内存大小，单位为Bytes，非32Bytes对齐会自动向上补齐至32Bytes对齐。

**约束说明**

声明TBufPool时，可以通过buf_id_size指定可分配Buffer的最大数量，默认上限为4，最大为16。TQue或TBuf的物理内存需要和TBufPool一致。

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
