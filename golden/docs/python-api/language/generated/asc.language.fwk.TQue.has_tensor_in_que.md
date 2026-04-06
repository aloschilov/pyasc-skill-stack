# asc.language.fwk.TQue.has_tensor_in_que

#### TQue.has_tensor_in_que() → bool

查询Que中目前是否已有入队的Tensor。

**对应的Ascend C函数原型**

```c++
__aicore__ inline bool HasTensorInQue()
```

**参数说明**

无。

**返回值说明**

- True：表示Queue中存在已入队的Tensor
- False：表示Queue为完全空闲

**约束说明**

该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。

**调用示例**

```python
pipe = asc.Tpipe()
que = asc.TQue(asc.TPosition.VECOUT, 4)
num = 4
len = 1024
pipe.init_buffer(que=que, num=num, len=len)
ret = que.has_tensor_in_que()
```
