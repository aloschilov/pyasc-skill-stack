# asc.language.adv.Matmul.set_tensor_b

#### Matmul.set_tensor_b(scalar: int) → None

#### Matmul.set_tensor_b(tensor: BaseTensor, transpose: bool = False) → None

设置矩阵乘的右矩阵B。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetTensorB(const GlobalTensor<SrcBT>& gm, bool isTransposeB = false)
```

```c++
__aicore__ inline void SetTensorB(const LocalTensor<SrcBT>& leftMatrix, bool isTransposeB = false)
```

```c++
__aicore__ inline void SetTensorB(SrcBT bScalar)
```

**参数说明**

- scalar: B矩阵中设置的值，为标量。
- tensor: B矩阵。类型为GlobalTensor或LocalTensor。
- transpose: B矩阵是否需要转置。

**约束说明**

- 传入的TensorB地址空间大小需要保证不小于single_k \* single_n。

**调用示例**

```python
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)   # 设置右矩阵B
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
```
