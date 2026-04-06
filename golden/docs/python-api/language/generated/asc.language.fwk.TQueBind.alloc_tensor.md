# asc.language.fwk.TQueBind.alloc_tensor

#### TQueBind.alloc_tensor(dtype: DataType) → [LocalTensor](../core.md#asc.language.core.LocalTensor)

#### TQueBind.alloc_tensor(tensor: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

从Que中分配Tensor，Tensor所占大小为init_buffer时设置的每块内存长度。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline LocalTensor<T> AllocTensor()
```

```c++
template <typename T>
__aicore__ inline void AllocTensor(LocalTensor<T>& tensor)
```

**参数说明**

- dtype：Tensor的数据类型。
- tensor：inplace接口需要传入LocalTensor作为内存管理的对象。

**返回值说明**

non-inplace接口返回值为LocalTensor对象，inplace接口没有返回值。

**约束说明**

- non-inplace接口分配的Tensor内容可能包含随机值。
- non-inplace接口，需要将TQueBind的depth模板参数设置为非零值；inplace接口，需要将TQueBind的depth模板参数设置为0。

**调用示例**

- non-inplace接口
  ```python
  pipe = asc.Tpipe()
  que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 2)
  num = 4
  len = 1024
  pipe.init_buffer(que=que, num=num, len=len)
  tensor = que.alloc_tensor(asc.half)
  ```
- inplace接口
  ```python
  pipe = asc.Tpipe()
  que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 2)
  num = 4
  len = 1024
  pipe.init_buffer(que=que, num=num, len=len)
  que.alloc_tensor(asc.half, tensor)
  ```
