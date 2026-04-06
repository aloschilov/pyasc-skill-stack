# asc.language.core.LocalTensor.reinterpret_cast

#### LocalTensor.reinterpret_cast(dtype: DataType) → [LocalTensor](../core.md#asc.language.core.LocalTensor)

将当前Tensor重解释为用户指定的新类型，转换后的Tensor与原Tensor地址及内容完全相同，Tensor的大小（字节数）保持不变。

**对应的Ascend C函数原型**

```c++
template <typename CAST_T>
__aicore__ inline LocalTensor<CAST_T> ReinterpretCast() const
```

**参数说明**

- cast_t：用户指定的新类型。

**返回值说明**

重解释转换后的Tensor。

**调用示例**

```python
# 调用ReinterpretCast将input_local重解释为int16_t类型
interpre_tensor = input_local.reinterpret_cast(asc.int16)
```
