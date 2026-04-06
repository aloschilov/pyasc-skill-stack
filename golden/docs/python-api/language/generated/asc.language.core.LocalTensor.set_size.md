# asc.language.core.LocalTensor.set_size

#### LocalTensor.set_size(size: int = 0) → None

设置当前LocalTensor Size大小。单位为元素。当用户重用local tensor变量且使用长度发生变化的时候，需要使用此接口重新设置Size。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetSize(const uint32_t size)
```

**参数说明**

- size：元素个数，单位为元素。

**调用示例**

```python
# 将申请的Tensor长度修改为256(单位为元素)
tmp_buffer = temp_queue.alloc_tensor(asc.float)
tmp_buffer.set_size(256)
```
