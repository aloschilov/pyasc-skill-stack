# asc.lib.host.MultiCoreMatmulTiling.get_single_shape

#### MultiCoreMatmulTiling.get_single_shape(self: libhost.MultiCoreMatmulTiling) → object

获取计算后的single_core_m/single_core_n/single_core_k。

**对应的Ascend C函数原型**

```c++
int32_t GetSingleShape(int32_t &shapeM, int32_t &shapeN, int32_t &shapeK)
```

**参数说明**

- flag：是否使能切K轴。

**返回值说明**

以元组方式返回(single_core_m, single_core_m, single_core_k)。

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
# 获取计算后的singleCoreM/singleCoreN/singleCoreK
single_m, single_n, single_k = 0
ret = tiling.get_single_shape(single_m, single_n, single_k)
```
