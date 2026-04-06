# asc.language.fwk.TQueBind.get_tensor_count_in_que

#### TQueBind.get_tensor_count_in_que() → PlainValue | int

查询Que中已入队的Tensor数量。

**对应的Ascend C函数原型**

```c++
__aicore__ inline int32_t GetTensorCountInQue()
```

**参数说明**

无。

**返回值说明**

Que中已入队的Tensor数量。

**约束说明**

该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。

**调用示例**

```python
# 通过get_tensor_count_in_que查询que中已入队的Tensor数量，当前通过alloc_tensor接口分配了内存，并加入que中，num为1。
pipe = asc.Tpipe()
que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 4)
num = 4
len = 1024
pipe.init_buffer(que=que, num=num, len=len)
tensor1 = que.alloc_tensor(asc.half)
que.enque(tensor1)
num = que.get_tensor_count_in_que()
```
