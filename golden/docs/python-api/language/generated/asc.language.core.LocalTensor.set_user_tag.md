# asc.language.core.LocalTensor.set_user_tag

#### LocalTensor.set_user_tag(tag: int = 0) → None

为Tensor添加用户自定义信息，用户可以根据需要设置对应的Tag。后续可通过GetUserTag获取指定Tensor的Tag信息，并根据Tag信息对Tensor进行相应操作。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetUserTag(const TTagType tag)
```

**参数说明**

- tag：设置的Tag信息，类型TTagType对应为int32_t。

**调用示例**

```python
tag = 10
size = input_local.get_size(tag)
```
