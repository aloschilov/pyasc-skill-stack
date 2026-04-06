# asc.language.fwk.TQue.has_idle_buffer

#### TQue.has_idle_buffer() → bool

查询Que中是否有空闲的内存块。

**对应的Ascend C函数原型**

```c++
__aicore__ inline bool HasIdleBuffer()
```

**参数说明**

无。

**返回值说明**

- True：表示Queue中存在空闲内存
- False：表示Queue中不存在空闲内存

**约束说明**

该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。

**调用示例**

```python
# 当前Que中已经分配了4块内存
pipe = asc.Tpipe()
que = asc.TQue(asc.TPosition.VECOUT, 1)
num = 4
len = 1024
pipe.init_buffer(que=que, num=num, len=len)
ret = que.has_idle_buffer() # 没有alloc_tensor的操作，返回值为True
tensor1 = que.alloc_tensor(asc.half)
ret = que.has_idle_buffer() # alloc_tensor了一块内存，返回值True
tensor2 = que.alloc_tensor(asc.half)
tensor3 = que.alloc_tensor(asc.half)
tensor4 = que.alloc_tensor(asc.half)
ret = que.has_idle_buffer() # alloc_tensor了四块内存，当前无空闲内存，返回值为False，继续alloc_tensor会报错
```
