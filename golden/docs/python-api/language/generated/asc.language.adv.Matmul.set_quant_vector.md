# asc.language.adv.Matmul.set_quant_vector

#### Matmul.set_quant_vector(quant_vector: [GlobalTensor](../core.md#asc.language.core.GlobalTensor)) → None

本接口提供对输出矩阵采用向量进行量化或反量化的功能，即对于输入shape为[1, N]的参数向量，
N值为Matmul矩阵计算时M/N/K中的N值，对输出矩阵的每一列都采用该向量中对应列的系数进行量化或反量化。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetQuantVector(const GlobalTensor<uint64_t>& quantTensor)
```

**参数说明**

- quant_vector：量化或反量化运算时的参数向量。

**约束说明**

- 需与set_dequant_type保持一致。
- 本接口必须在iterate或者iterate_all前调用。

**调用示例**

```python
gm_quant = asc.GlobalTensor()
...
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_quant_vector(gm_quant)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
```
