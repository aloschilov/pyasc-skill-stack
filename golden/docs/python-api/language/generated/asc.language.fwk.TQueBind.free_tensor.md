# asc.language.fwk.TQueBind.free_tensor

#### TQueBind.free_tensor(tensor: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

释放Que中的指定Tensor。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void FreeTensor(LocalTensor<T>& tensor)
```

**参数说明**

- tensor：待释放的Tensor。

**调用示例**

```python
pipe = asc.Tpipe()
que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 2)
num = 4
len = 1024
pipe.init_buffer(que=que, num=num, len=len)
tensor = que.alloc_tensor(asc.half)
que.free_tensor(tensor)
```
