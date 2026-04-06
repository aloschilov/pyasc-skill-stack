# asc.language.basic.ListTensorDesc.get_size

#### ListTensorDesc.get_size() → int

获取ListTensor中包含的数据指针的个数。

**对应的Ascend C函数原型**

```c++
__aicore__ inline uint32_t GetSize()
```

**参数说明**

无。

**返回值说明**

数据指针的个数。

**调用示例**

```python
x_desc = asc.ListTensorDesc(data=x, length=0xffffffff, shape_size=0xffffffff)
x_size = x_desc.get_size()
```
