# asc.language.adv.Matmul.set_batch_num

#### Matmul.set_batch_num(batch_a: PlainValue | int, batch_b: PlainValue | int) → None

在不改变Tiling的情况下，重新设置多Batch计算的Batch数。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetTail(int tailM = -1, int tailN = -1, int tailK = -1)
```

**参数说明**

- tail_m：重新设置的singleCoreM值。
- tail_n：重新设置的singleCoreN值。
- tail_k：重新设置的singleCoreK值。

**约束说明**

- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
- 本接口仅支持在纯Cube模式（只有矩阵计算）下调用。

**调用示例**

```python
# 纯Cube模式
mm.set_tensor_a(gm_a, is_transpose_a_in)
mm.set_tensor_b(gm_b, is_transpose_b_in)
if tiling.is_bias:
    mm.set_bias(gm_bias)
mm.set_batch_num(batch_a, batch_b)
# 多batch Matmul计算
mm.iterate_batch(tensor=gm_c, en_partial_sum=False, en_atomic=0, en_sequential_write=False)
```
