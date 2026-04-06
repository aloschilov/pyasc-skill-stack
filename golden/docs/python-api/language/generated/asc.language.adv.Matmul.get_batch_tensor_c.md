# asc.language.adv.Matmul.get_batch_tensor_c

#### Matmul.get_batch_tensor_c(batch_a: int, batch_b: int, en_sequential_write: bool = False, sync: bool = True) → [GlobalTensor](../core.md#asc.language.core.GlobalTensor)

#### Matmul.get_batch_tensor_c(tensor: [LocalTensor](../core.md#asc.language.core.LocalTensor), batch_a: int, batch_b: int, en_sequential_write: bool = False, sync: bool = True) → None

调用一次get_batch_tensor_c，会获取C矩阵片，该接口可以与iterate_n_batch异步接口配合使用。
用于在调用iterate_n_batch迭代计算后，获取一片std::max(batch_a, batch_b) \* singleCoreM \* singleCoreN大小的矩阵分片。

**对应的Ascend C函数原型**

```c++
template <bool sync = true>
__aicore__ inline GlobalTensor<DstT> GetBatchTensorC(uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
```

```c++
template <bool sync = true>
__aicore__ inline void GetBatchTensorC(const LocalTensor<DstT>& c, uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
```

**参数说明**

- batch_a: 左矩阵的batch数。
- batch_b: 右矩阵的batch数。
- en_sequential_write: 该参数预留，开发者无需关注。
- tensor: C矩阵放置于Local Memory的地址，用于保存矩阵分片。

**约束说明**

- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
- C矩阵片输出到Local Memory，且单核计算的N方向大小single_core_n非32字节对齐的场景，C矩阵的CubeFormat仅支持ND_ALIGN格式，输出C矩阵片时，自动将single_core_n方向上的数据补齐至32字节。

**调用示例**

```python
for_extent = tiling.a_layout_info_b * tiling.a_layout_info_n * g_lay // tiling.batch_num
mm.set_tensor_a(gm_a, is_transpose_a_in)
mm.set_tensor_b(gm_b, is_transpose_b_in)
if tiling.is_bias:
    mm.set_bias(gm_bias)
mm.iterate_n_batch(for_extent, batch_a, batch_b, False, sync=False)
# ...其他计算
for i in range(for_extent):
    mm.get_batch_tensor_c(tensor=ub_cmatrix, sync=False)
```
