# asc.lib.host.MatmulApiTiling.get_base_m

#### MatmulApiTiling.get_base_m(self: libhost.MatmulApiTilingBase) → int

获取Tiling计算得到的baseM值。

**对应的Ascend C函数原型**

```c++
int32_t GetBaseM()
```

**返回值说明**

返回值为Tiling计算得到的baseM值。

**约束说明**

使用创建的Tiling对象调用该接口，且需在完成Tiling计算（GetTiling）后调用。

**调用示例**

```python
import asc.lib.host as host
ascendc_platform = host.get_ascendc_platform()
tiling = host.MatmulApiTiling(ascendc_platform)
tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_shape(1024, 1024, 1024)
tiling.set_org_shape(1024, 1024, 1024)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1)
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
bask_m = tiling.get_base_m()
```
