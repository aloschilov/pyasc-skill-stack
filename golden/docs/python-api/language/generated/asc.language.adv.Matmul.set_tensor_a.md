# asc.language.adv.Matmul.set_tensor_a

#### Matmul.set_tensor_a(scalar: int) → None

#### Matmul.set_tensor_a(tensor: BaseTensor, transpose: bool = False) → None

设置矩阵乘的左矩阵A。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetTensorA(const GlobalTensor<SrcAT>& gm, bool isTransposeA = false)
```

```c++
__aicore__ inline void SetTensorA(const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA = false)
```

```c++
__aicore__ inline void SetTensorA(SrcAT aScalar)
```

**参数说明**

- scalar: A矩阵中设置的值，为标量。
- tensor: A矩阵。类型为GlobalTensor或LocalTensor。
- transpose: A矩阵是否需要转置。

**约束说明**

- 传入的TensorA地址空间大小需要保证不小于single_m \* single_k。

**调用示例**

```python
asc.adv.register_matmul(pipe, workspace, mm, tiling)
# 示例一：左矩阵在Global Memory
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
# 示例二：左矩阵在Local Memory
mm.set_tensor_a(local_a)
# 示例三：设置标量数据
mm.set_tensor_a(scalar_a)
```
