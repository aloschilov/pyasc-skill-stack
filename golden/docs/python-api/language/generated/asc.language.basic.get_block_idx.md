# asc.language.basic.get_block_idx

### asc.language.basic.get_block_idx() → int

获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。

**对应的Ascend C函数原型**

```c++
__aicore__ inline int64_t GetBlockIdx()
```

**参数说明**

无。

**约束说明**

GetBlockIdx为一个系统内置函数，返回当前核的index。

**调用示例**

```python
src0_global.set_global_buffer(src0_gm + asc.get_block_idx() * single_core_offset)
src1_global.set_global_buffer(src1_gm + asc.get_block_idx() * single_core_offset)
dst_global.set_global_buffer(dst_gm + asc.get_block_idx() * single_core_offset)
pipe.init_buffer(que=in_queue_src0, num=1, len=256*asc.float.sizeof())
pipe.init_buffer(que=in_queue_src1, num=1, len=256*asc.float.sizeof())
pipe.init_buffer(que=sel_mask, num=1, len=256)
pipe.init_buffer(que=out_queue_dst, num=1, len=256*asc.float.sizeof())
```
