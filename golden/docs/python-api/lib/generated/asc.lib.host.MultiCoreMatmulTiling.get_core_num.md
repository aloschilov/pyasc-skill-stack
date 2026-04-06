# asc.lib.host.MultiCoreMatmulTiling.get_core_num

#### MultiCoreMatmulTiling.get_core_num(self: libhost.MultiCoreMatmulTiling) → object

获得多核切分所使用的BlockNum参数。

**对应的Ascend C函数原型**

```c++
int32_t GetCoreNum(int32_t &dim, int32_t &mDim, int32_t &nDim)
```

**返回值说明**

以元组方式返回(dim, m_dim, n_dim)。

**约束说明**

使用创建的Tiling对象调用该接口，且需在完成Tiling计算（get_tiling）后调用。

**调用示例**

```python
import asc.lib.host as host
ascendc_platform = host.get_ascendc_platform()
tiling = host.MultiCoreMatmulTiling(ascendc_platform)
tiling.set_dim(use_core_nums)
tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_shape(1024, 1024, 1024)
tiling.set_single_shape(1024, 1024, 1024)
tiling.set_org_shape(1024, 1024, 1024)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1)
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
# 获得多核切分后，使用的BlockNum
dim, m_dim, n_dim = 0
ret1 = tiling.get_core_num(dim, m_dim, n_dim)
```
