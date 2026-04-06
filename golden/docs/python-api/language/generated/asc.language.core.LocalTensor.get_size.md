# asc.language.core.LocalTensor.get_size

#### LocalTensor.get_size() → int

获取当前LocalTensor Size大小。

**对应的Ascend C函数原型**

```c++
__aicore__ inline uint32_t GetSize() const
```

**参数说明**

无。

**返回值说明**

当前LocalTensor Size大小。单位为元素。

**调用示例**

```python
size = input_local.get_size()
```
