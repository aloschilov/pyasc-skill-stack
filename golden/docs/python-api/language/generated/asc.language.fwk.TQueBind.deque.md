# asc.language.fwk.TQueBind.deque

#### TQueBind.deque(dtype: DataType) → [LocalTensor](../core.md#asc.language.core.LocalTensor)

#### TQueBind.deque(tensor: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

#### TQueBind.deque(dtype: DataType, src_user_pos: TPosition, dst_user_pos: TPosition) → [LocalTensor](../core.md#asc.language.core.LocalTensor)

将Tensor从队列中取出，用于后续处理。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline LocalTensor<T> DeQue()
```

```c++
template <typename T>
__aicore__ inline void DeQue(LocalTensor<T>& tensor)
```

**参数说明**

- dtype：Tensor的数据类型。
- tensor：inplace接口需要通过出参的方式返回Tensor。

**返回值说明**

non-inplace接口的返回值为从队列中取出的LocalTensor；inplace接口没有返回值。

**约束说明**

- 对空队列执行deque是一种异常行为，会在CPU调测时报错。
- non-inplace接口，需要将TQueBind的depth模板参数设置为非零值；inplace接口，需要将TQueBind的depth模板参数设置为0。

**调用示例**

- non-inplace接口
  ```python
  pipe = asc.Tpipe()
  que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 4)
  num = 4
  len = 1024
  pipe.init_buffer(que=que, num=num, len=len)
  tensor1 = que.alloc_tensor(asc.half)
  que.enque(tensor1)
  tensor2 = que.deque(asc.half)
  ```
- inplace接口
  ```python
  pipe = asc.Tpipe()
  que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 0)
  num = 4
  len = 1024
  pipe.init_buffer(que=que, num=num, len=len)
  tensor1 = que.alloc_tensor(asc.half)
  que.enque(tensor1)
  que.deque(asc.half, tensor1)
  que.free_tensor(tensor1)
  ```
