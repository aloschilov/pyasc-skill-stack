# asc.language.adv.Matmul.set_bias

#### Matmul.set_bias(tensor: BaseTensor) → None

设置矩阵乘的Bias。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetBias(const GlobalTensor<BiasT>& biasGlobal)
```

```c++
__aicore__ inline void SetBias(const LocalTensor<BiasT>& inputBias)
```

**参数说明**

- tensor: Bias矩阵。类型为GlobalTensor或LocalTensor。

**约束说明**

- 在Matmul Tiling计算中，必须配置TCubeTiling结构中的is_bias参数为1，即使能Bias后，才能调用本接口设置Bias矩阵。
- 传入的Bias地址空间大小需要保证不小于single_n。

**调用示例**

```python
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)    # 设置Bias
mm.iterate_all(gm_c)
```
