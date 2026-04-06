# asc.language.basic.ListTensorDesc.init

#### ListTensorDesc.init(data: GlobalAddress, length: int = 4294967295, shape_size: int = 4294967295) → None

初始化函数，用于解析对应的内存排布。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void Init(__gm__ void* data, uint32_t length = 0xffffffff, uint32_t shapeSize = 0xffffffff)
```

**参数说明**

- data: 待解析数据的首地址。
- length: 待解析内存的长度。
- shape_size: 数据指针的个数。

**调用示例**

```python
x_desc = asc.ListTensorDesc()
x_desc.init(data=x, length=0xffffffff, shape_size=0xffffffff)
```
