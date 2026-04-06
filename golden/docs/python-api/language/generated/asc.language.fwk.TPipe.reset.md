# asc.language.fwk.TPipe.reset

#### TPipe.reset() → None

完成资源的释放与eventId等变量的初始化操作，恢复到TPipe的初始化状态。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void Reset()
```

**参数说明**

无。

**调用示例**

```python
# 为TQue分配内存，分配内存块数为2，每块大小为128字节
pipe = asc.Tpipe()
que = asc.TQue(asc.TPosition.VECOUT, 1)
num = 1;
len = 192 * 1024;
for i in range(2):
    pipe.init_buffer(que=que, num=num, len=len)
    ...     # process
    pipe.reset()
```
