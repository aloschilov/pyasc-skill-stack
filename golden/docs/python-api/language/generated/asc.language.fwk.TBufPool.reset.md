# asc.language.fwk.TBufPool.reset

#### TBufPool.reset() → None

在切换TBufPool资源池时使用，结束当前TbufPool资源池正在处理的相关事件。
调用后当前资源池及资源池分配的Buffer仍然存在，只是Buffer内容可能会被改写。
可以切换回该资源池后，重新开始使用该Buffer，无需再次分配。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void Reset()
```

**参数说明**

无。

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
