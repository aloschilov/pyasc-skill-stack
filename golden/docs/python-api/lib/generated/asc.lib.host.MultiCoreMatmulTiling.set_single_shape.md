# asc.lib.host.MultiCoreMatmulTiling.set_single_shape

#### MultiCoreMatmulTiling.set_single_shape(self: libhost.MultiCoreMatmulTiling, single_m_in: SupportsInt = -1, single_n_in: SupportsInt = -1, single_k_in: SupportsInt = -1) → int

设置Matmul单核计算的形状single_m_in，single_n_in，single_k_in，单位为元素。

**对应的Ascend C函数原型**

```c++
int32_t SetSingleShape(int32_t singleMIn = -1, int32_t singleNIn = -1, int32_t singleKIn = -1)
```

**参数说明**

- single_m_in：设置的single_m_in大小，单位为元素，默认值为-1。-1表示不设置指定的single_m_in，该值由tiling函数自行计算。
- single_n_in：设置的single_n_in大小，单位为元素，默认值为-1。-1表示不设置指定的single_n_in，该值由tiling函数自行计算。
- single_k_in：设置的single_k_in大小，单位为元素，默认值为-1。-1表示不设置指定的single_k_in，该值由tiling函数自行计算。

**返回值说明**

-1表示设置失败； 0表示设置成功。

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
tiling.set_shape(1024, 1024, 1024)    # 设置Matmul单次计算的形状
tiling.set_single_shape(1024, 1024, 1024) # 设置单核计算的形状
tiling.set_org_shape(1024, 1024, 1024)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1)
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
