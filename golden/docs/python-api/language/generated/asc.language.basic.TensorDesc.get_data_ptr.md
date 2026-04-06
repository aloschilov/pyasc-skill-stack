# asc.language.basic.TensorDesc.get_data_ptr

#### TensorDesc.get_data_ptr() → GlobalAddress

获取储存Tensor数据地址。

**对应的Ascend C函数原型**

```c++
T* GetDataPtr()
```

**返回值说明**

返回储存Tensor数据地址。T数据类型。。

**调用示例**

```python
tensor_desc = asc.TensorDesc()
data_ptr = tensor_desc.get_data_ptr()
```
