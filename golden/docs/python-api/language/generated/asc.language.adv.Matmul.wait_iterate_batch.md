# asc.language.adv.Matmul.wait_iterate_batch

#### Matmul.wait_iterate_batch() → None

等待iterate_batch异步接口或iterate_nbatch异步接口返回，支持连续输出到Global Memory。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void WaitIterateBatch()
```

**参数说明**

无。

**约束说明**

- 配套iterate_batchiIterate_n_batch异步接口使用。
- 仅支持连续输出至Global Memory。
- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

**调用示例**

```python
mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type)
mm.set_tensor_a(gm_a[offset_a])
mm.set_tensor_b(gm_b[offset_b])
if tiling.is_bias:
    mm.set_bias(gm_bias[offset_bias])
mm.iterate_batch(tensor=gm_c[offsetc], batch_a=batch_a, batch_b=batch_b, en_sequential_write=False)
mm.wait_iterate_batch()
```
