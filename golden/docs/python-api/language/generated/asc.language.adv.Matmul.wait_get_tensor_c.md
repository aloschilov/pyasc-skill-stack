# asc.language.adv.Matmul.wait_get_tensor_c

#### Matmul.wait_get_tensor_c() → None

当使用GetTensorC异步接口将结果矩阵从GM拷贝到UB，且UB后续需要进行Vector计算时，需要调用WaitGetTensorC进行同步。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void WaitGetTensorC()
```

**参数说明**

无。

**约束说明**

- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

**调用示例**

```python
# 异步模式样例
mm.iterate(sync=False)
# 其他操作
for i in range(single_corem // base_m * single_core_n // base_n):
    mm.get_tensor_c(tensor=ub_cmatrix, sync=False)
    mm.wait_get_tensor_c()
```
