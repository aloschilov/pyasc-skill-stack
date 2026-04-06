# asc.lib.host.MatmulApiTiling.get_tiling

#### MatmulApiTiling.get_tiling(self: libhost.MatmulApiTilingBase, tiling: object) → int

获取Tiling参数。

**对应的Ascend C函数原型**

```c++
int64_t GetTiling(optiling::TCubeTiling &tiling)
int64_t GetTiling(TCubeTiling &tiling)
```

**参数说明**

- tiling：Tiling结构体存储最终的tiling结果。

**约束说明**

在Tiling计算失败的场景，若需查看Tiling计算失败的原因，请将日志级别设置为WARNING级别，并在日志中搜索关键字“MatmulApi Tiling”。

**返回值说明**

如果返回值不为-1，则代表Tiling计算成功，用户可以使用该Tiling结构的值。如果返回值为-1，则代表Tiling计算失败，该Tiling结果无法使用。

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
