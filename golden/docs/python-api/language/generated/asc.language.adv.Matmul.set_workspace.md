# asc.language.adv.Matmul.set_workspace

#### Matmul.set_workspace(addr: [GlobalTensor](../core.md#asc.language.core.GlobalTensor) | GlobalAddress, size: PlainValue | int | None = None) → None

Iterate计算的异步场景，调用本接口申请一块临时空间来缓存计算结果，然后调用GetTensorC时会在该临时空间中获取C的矩阵分片。
IterateNBatch计算时，调用本接口申请一块临时空间来缓存计算结果，然后根据同步或异步场景进行其它接口的调用。

**对应的Ascend C函数原型**

```c++
template <class T> __aicore__ inline void SetWorkspace(GlobalTensor<T>& addr)
```

```c++
template <class T> __aicore__ inline void SetWorkspace(__gm__ const T* addr, int size)
```

**参数说明**

- addr：用户传入的GM上的workspace空间，GlobalTensor类型。
- addr：用户传入的GM上的workspace空间，GM地址类型。
- size：传入GM地址时，需要配合传入元素个数。

**约束说明**

- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

**调用示例**

```python
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_workspace(workspace_gm)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate(sync=True)
for i in range(single_corem // base_m * single_core_n // base_n):
    mm.get_tensor_c(tensor=gm_c, sync=False)
```
