# asc.lib.host.MatmulApiTiling.set_traverse

#### MatmulApiTiling.set_traverse(self: libhost.MatmulApiTilingBase, traverse: libhost.MatrixTraverse) → int

设置固定的Matmul计算方向，M轴优先还是N轴优先。

**对应的Ascend C函数原型**

```c++
int32_t SetTraverse(MatrixTraverse traverse)
```

**参数说明**

- traverse：设置固定的Matmul计算方向。可选值：MatrixTraverse::FIRSTM/MatrixTraverse::FIRSTN。

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
tiling.set_traverse(host.MatrixTraverse.FIRSTM)
tiling.set_buffer_space(-1, -1, -1)
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
