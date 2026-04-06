# asc.language.fwk.TQueBind.free_all_event

#### TQueBind.free_all_event() → None

释放队列中申请的所有同步事件。队列分配的Buffer关联着同步事件的event_id，因为同步事件的数量有限制，
如果同时使用的队列Buffer数量超过限制，将无法继续申请队列，使用本接口释放队列中的事件后，可以再次申请队列。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void FreeAllEvent()
```

**参数说明**

无。

**约束说明**

该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。

**调用示例**

```python
pipe = asc.Tpipe()
que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 4)
num = 4
len = 1024
pipe.init_buffer(que=que, num=num, len=len)
tensor1 = que.alloc_tensor(asc.half)
que.enque(tensor1)
tensor1 = que.deque(asc.half)
que.free_tensor(tensor1)
que.free_all_event()
```
