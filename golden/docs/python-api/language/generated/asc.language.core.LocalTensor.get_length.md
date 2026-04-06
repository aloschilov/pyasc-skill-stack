# asc.language.core.LocalTensor.get_length

#### LocalTensor.get_length() → int

获取LocalTensor数据长度。

**对应的Ascend C函数原型**

```c++
__aicore__ inline uint32_t GetLength() const
```

**参数说明**

无。

**返回值说明**

LocalTensor数据长度，单位为字节。

**调用示例**

```python
# 获取localTensor的长度(单位为Byte)
len = input_local.get_length()
```
