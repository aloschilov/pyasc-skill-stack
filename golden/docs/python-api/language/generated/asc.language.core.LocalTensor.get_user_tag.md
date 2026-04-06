# asc.language.core.LocalTensor.get_user_tag

#### LocalTensor.get_user_tag() → int

获取指定Tensor块的Tag信息，用户可以根据Tag信息对Tensor进行不同操作。

**对应的Ascend C函数原型**

```c++
__aicore__ inline TTagType GetUserTag() const
```

**参数说明**

无。

**返回值说明**

指定Tensor块的Tag信息。

**调用示例**

```python
tensor1 = que1.deque(asc.half)
tag1 = tensor1.get_user_tag()
tensor2 = que2.deque(asc.half)
tag2 = tensor2.get_user_tag()
tensor3 = que3.alloc_tensor(asc.half)
# 使用Tag控制条件语句执行
if tag1 <= 10 and tag2 >= 9:
    asc.add(tensor3, tensor1, tensor2, count=tile_length)
```
