# asc.language.fwk.TQueBind.enque

#### TQueBind.enque(\*args, \*\*kwargs) → [LocalTensor](../core.md#asc.language.core.LocalTensor) | None

将Tensor push到队列。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline bool EnQue(const LocalTensor<T>& tensor)
```

**参数说明**

- tensor：指定的Tensor

**返回值说明**

- True：表示Tensor加入Queue成功
- False：表示Queue已满，入队失败

**调用示例**

```python
pipe = asc.Tpipe()
que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 2)
num = 4
len = 1024
pipe.init_buffer(que=que, num=num, len=len)
tensor = que.alloc_tensor(asc.half)
que.enque(tensor)
```
