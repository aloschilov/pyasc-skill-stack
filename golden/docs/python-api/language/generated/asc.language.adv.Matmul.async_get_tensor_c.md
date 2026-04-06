# asc.language.adv.Matmul.async_get_tensor_c

#### Matmul.async_get_tensor_c(c: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

获取Iterate接口异步计算的结果矩阵。该接口功能已被GetTensorC覆盖，建议直接使用GetTensorC异步接口。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void AsyncGetTensorC(const LocalTensor<DstT>& c)
```

**参数说明**

- c：结果矩阵

**约束说明**

- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
