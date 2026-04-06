# asc.lib.host.MultiCoreMatmulTiling.set_dim

#### MultiCoreMatmulTiling.set_dim(self: libhost.MultiCoreMatmulTiling, dim: SupportsInt) → int

设置多核Matmul时，参与运算的核数。

**对应的Ascend C函数原型**

```c++
int32_t SetDim(int32_t dim)
```

**参数说明**

- dim：多核Matmul tiling计算时，可以使用的核数。

**返回值说明**

-1表示设置失败； 0表示设置成功。

**调用示例**

```python
import asc.lib.host as host
ascendc_platform = host.get_ascendc_platform()
tiling = host.MultiCoreMatmulTiling(ascendc_platform)
tiling.set_dim(use_core_nums) # 设置参与运算的核数
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
```
