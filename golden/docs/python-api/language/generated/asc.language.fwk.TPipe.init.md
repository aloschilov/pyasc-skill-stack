# asc.language.fwk.TPipe.init

#### TPipe.init() → None

初始化内存和用于同步流水事件的EventID的初始化。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void TPipe::Init()
```

**参数说明**

无。

**约束说明**

重复申请释放tpipe，要与destroy接口成对使用，tpipe如果要重复申请需要先destroy释放后再init。

**调用示例**

```python
class KernelAsin:
    ...
op = KernelAsin
pipe_in = asc.Tpipe()
for index in range(1):
    if index != 0:
        pipe_in.init()
    op.process()
    pipe_in.Destroy()
pipe_cast = asc.Tpipe()
op.init(src_gm, dst_gm, src_size, pipe_cast)
op.Process()
pipe_cast.destroy()
```
