# asc.language.fwk.TQueBind.vacant_in_que

#### TQueBind.vacant_in_que() → bool

查询队列是否已满。

**对应的Ascend C函数原型**

```c++
__aicore__ inline bool VacantInQue()
```

**参数说明**

无。

**返回值说明**

- True：表示Queue未满，可以继续enque操作
- False：表示Queue已满，不可以继续入队

**约束说明**

该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。

**调用示例**

```python
# 根据vacant_in_que判断当前que是否已满，设置当前队列深度为4
pipe = asc.Tpipe()
que = asc.TQueBind(asc.TPosition.VECOUT, asc.TPosition.GM, 4)
num = 10
len = 1024
pipe.init_buffer(que=que, num=num, len=len)
tensor1 = que.alloc_tensor(asc.half)
tensor2 = que.alloc_tensor(asc.half)
tensor3 = que.alloc_tensor(asc.half)
tensor4 = que.alloc_tensor(asc.half)
tensor5 = que.alloc_tensor(asc.half)
que.enque(tensor1)
que.enque(tensor2)
que.enque(tensor3)
que.enque(tensor4)
ret = que.vacant_in_que()   # 返回False，继续入队操作将报错
```
