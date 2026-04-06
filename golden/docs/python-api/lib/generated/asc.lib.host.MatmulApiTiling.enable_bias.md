# asc.lib.host.MatmulApiTiling.enable_bias

#### MatmulApiTiling.enable_bias(self: libhost.MatmulApiTilingBase, is_bias_in: bool = False) → int

设置Bias是否参与运算，设置的信息必须与Kernel侧保持一致。

**对应的Ascend C函数原型**

```c++
int32_t EnableBias(bool isBiasIn = false)
```

**参数说明**

- is_bias_in：设置是否有Bias参与运算。

**返回值说明**

-1表示设置失败；0表示设置成功。

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
tiling.enable_bias(True)
tiling.set_buffer_space(-1, -1, -1)
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
