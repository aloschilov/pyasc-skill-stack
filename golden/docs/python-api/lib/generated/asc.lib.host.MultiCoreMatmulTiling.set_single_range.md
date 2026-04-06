# asc.lib.host.MultiCoreMatmulTiling.set_single_range

#### MultiCoreMatmulTiling.set_single_range(self: libhost.MultiCoreMatmulTiling, max_m: SupportsInt = -1, max_n: SupportsInt = -1, max_k: SupportsInt = -1, min_m: SupportsInt = -1, min_n: SupportsInt = -1, min_k: SupportsInt = -1) → int

设置single_core_m/single_core_n/single_core_k的最大值与最小值。

**对应的Ascend C函数原型**

```c++
int32_t SetSingleRange(int32_t maxM = -1, int32_t maxN = -1, int32_t maxK = -1, int32_t minM = -1, int32_t minN = -1, int32_t minK = -1)
```

**参数说明**

- max_m：设置最大的single_core_m值，默认值为-1，表示不设置指定的single_core_m最大值，该值由Tiling函数自行计算。
- max_n：设置最大的single_core_n值，默认值为-1，表示不设置指定的single_core_n最大值，该值由Tiling函数自行计算。
- max_k：设置最大的single_core_k值，默认值为-1，表示不设置指定的single_core_k最大值，该值由Tiling函数自行计算。
- min_m：设置最小的single_core_m值，默认值为-1，表示不设置指定的single_core_m最小值，该值由Tiling函数自行计算。
- min_n：设置最小的single_core_n值，默认值为-1，表示不设置指定的single_core_n最小值，该值由Tiling函数自行计算。
- min_k：设置最小的single_core_k值，默认值为-1，表示不设置指定的single_core_k最小值，该值由Tiling函数自行计算。

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
tiling.set_shape(1024, 1024, 1024)
tiling.set_single_range(1024, 1024, 1024, 1024, 1024, 1024) # 设置single_core_m/single_core_n/single_core_k的最大值与最小值
tiling.set_org_shape(1024, 1024, 1024)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1)
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
