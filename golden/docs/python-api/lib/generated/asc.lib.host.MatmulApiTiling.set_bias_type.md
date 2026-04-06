# asc.lib.host.MatmulApiTiling.set_bias_type

#### MatmulApiTiling.set_bias_type(self: libhost.MatmulApiTilingBase, pos: libhost.TPosition, type: libhost.CubeFormat, data_type: libhost.DataType) → int

设置Bias的位置，数据格式，数据类型，是否转置等信息，这些信息需要和kernel侧的设置保持一致。

**对应的Ascend C函数原型**

```c++
int32_t SetBiasType(TPosition pos, CubeFormat type, DataType dataType)
```

**参数说明**

- pos：Bias矩阵所在的buffer位置。
- type：Bias矩阵的数据格式。
- data_type：Bias矩阵的数据类型。

**返回值说明**

-1表示设置失败； 0表示设置成功。

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
```
