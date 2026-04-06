# asc.language.adv.Matmul.set_sparse_index

#### Matmul.set_sparse_index(index_global: [GlobalTensor](../core.md#asc.language.core.GlobalTensor)) → None

设置稀疏矩阵稠密化过程生成的索引矩阵。
索引矩阵的Format格式要求为NZ格式。
本接口仅支持在纯Cube模式（只有矩阵计算）且MDL模板的场景使用。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetSparseIndex(const GlobalTensor<uint8_t>& indexGlobal)
```

**参数说明**

- index_global：索引矩阵在Global Memory上的首地址，类型为GlobalTensor。

**约束说明**

- 索引矩阵的Format格式要求为NZ格式。
- 本接口仅支持在纯Cube模式（只有矩阵计算）且MDL模板的场景使用。

**调用示例**

```python
@asc.jit(matmul_cube_only=True) # 使能纯Cube模式（只有矩阵计算）
def matmul_kernel(...):
    ...
    asc.adv.register_matmul(pipe, workspace, mm, tiling)
    mm.set_tensor_a(gm_a)
    mm.set_tensor_b(gm_b)
    mm.set_sparse_index(gm_index) # 设置索引矩阵
    mm.set_bias(gm_bias)
    mm.iterate_all(gm_c)
```
