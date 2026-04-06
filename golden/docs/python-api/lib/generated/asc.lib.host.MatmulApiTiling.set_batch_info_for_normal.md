# asc.lib.host.MatmulApiTiling.set_batch_info_for_normal

#### MatmulApiTiling.set_batch_info_for_normal(self: libhost.MatmulApiTilingBase, batch_a: SupportsInt, batch_b: SupportsInt, m: SupportsInt, n: SupportsInt, k: SupportsInt) → int

设置A/B矩阵的M/N/K轴信息，以及A/B矩阵的Batch数。Layout类型为NORMAL的场景，
调用IterateBatch或者IterateNBatch接口之前，需要在Host侧Tiling实现中通过本接口设置A/B矩阵的M/N/K轴等信息。

**对应的Ascend C函数原型**

```c++
int32_t SetBatchInfoForNormal(int32_t batchA, int32_t batchB, int32_t m, int32_t n, int32_t k)
```

**参数说明**

- batch_a：A矩阵的batch数。
- batch_b：B矩阵的batch数。
- m：A矩阵的M轴信息
- n：B矩阵的N轴信息
- k：A/B矩阵的K轴信息

**返回值说明**

-1表示设置失败； 0表示设置成功。

**约束说明**

Layout类型为NORMAL的场景，调用iterate_batch或者iterate_n_batch接口之前，需要在Host侧Tiling实现中通过本接口设置A/B矩阵的M/N/K轴等信息。

**调用示例**

```python
import asc.lib.host as host
ascendc_platform = host.get_ascendc_platform()
tiling = host.MultiCoreMatmulTiling(ascendc_platform)
m = 32
n = 256
k = 64
tiling.set_dim(1)
tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_shape(m, n, k)
tiling.set_org_shape(m, n, k)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1)
batch_num = 3
tiling.set_batch_info_for_normal(batch_num, batch_num, m, n, k)
tiling.set_buffer_space(-1, -1, -1);
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
