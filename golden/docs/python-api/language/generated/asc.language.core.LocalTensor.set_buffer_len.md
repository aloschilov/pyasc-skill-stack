# asc.language.core.LocalTensor.set_buffer_len

#### LocalTensor.set_buffer_len(data_len: int) → None

设置Buffer长度。当用户调用operator[]函数创建新LocalTensor时，建议调用该接口设置新LocalTensor长度，便于编译器对内存及同步进行自动优化。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetBufferLen(uint32_t dataLen)
```

**参数说明**

- data_len：Buffer长度，单位为字节。

**调用示例**

```python
# 将申请的Tensor长度修改为1024(单位为字节)
tmp_buffer = temp_queue.alloc_tensor(asc.float)
tmp_buffer.set_buffer_len(1024)
```
