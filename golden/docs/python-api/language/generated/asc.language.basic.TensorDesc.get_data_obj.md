# asc.language.basic.TensorDesc.get_data_obj

#### TensorDesc.get_data_obj() → [GlobalTensor](../core.md#asc.language.core.GlobalTensor)

将数据指针置于GlobalTensor中并返回该GlobalTensor。

**对应的Ascend C函数原型**

```c++
GlobalTensor<T> GetDataObj()
```

**返回值说明**

返回设置了数据指针的GlobalTensor。

**调用示例**

```python
tensor_desc = asc.TensorDesc()
data_obj = tensor_desc.get_data_obj()
```
