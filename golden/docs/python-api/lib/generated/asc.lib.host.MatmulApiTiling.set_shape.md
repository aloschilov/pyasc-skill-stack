# asc.lib.host.MatmulApiTiling.set_shape

#### MatmulApiTiling.set_shape(self: libhost.MatmulApiTilingBase, m: SupportsInt, n: SupportsInt, k: SupportsInt) → int

设置Matmul计算的形状m、n、k，该形状可以为原始完整矩阵或其局部矩阵，单位为元素。该形状的矩阵乘可以由单核或多核计算完成。

**对应的Ascend C函数原型**

```c++
int32_t SetShape(int32_t m, int32_t n, int32_t k)
```

**参数说明**

- m：设置Matmul计算的M方向大小，单位为元素。
- n：设置Matmul计算的N方向大小，单位为元素。
- k：设置Matmul计算的K方向大小，单位为元素。

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
