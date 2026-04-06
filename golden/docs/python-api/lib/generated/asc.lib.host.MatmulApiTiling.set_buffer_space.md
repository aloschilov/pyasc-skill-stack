# asc.lib.host.MatmulApiTiling.set_buffer_space

#### MatmulApiTiling.set_buffer_space(self: libhost.MatmulApiTilingBase, l1_size: SupportsInt = -1, l0_c_size: SupportsInt = -1, ub_size: SupportsInt = -1, bt_size: SupportsInt = -1) → int

设置Matmul计算时可用的L1 Buffer/L0C Buffer/Unified Buffer/BiasTable Buffer空间大小，单位为字节。

**对应的Ascend C函数原型**

```c++
int32_t SetBufferSpace(int32_t l1Size = -1, int32_t l0CSize = -1, int32_t ubSize = -1, int32_t btSize = -1)
```

**参数说明**

- l1_size：设置Matmul计算时，能够使用的L1 Buffer大小，单位为字节。默认值-1，表示使用AI处理器L1 Buffer大小。
- l0_c_size：设置Matmul计算时，能够使用的L0C Buffer大小，单位为字节。默认值-1，表示使用AI处理器L0C Buffer大小。
- ub_size：设置Matmul计算时，能够使用的UB Buffer大小，单位为字节。默认值-1，表示使用AI处理器UB Buffer大小。
- bt_size：设置Matmul计算时，能够使用的BiasTable Buffer大小，单位为字节。默认值-1，表示使用AI处理器BiasTable Buffer大小。

**返回值说明**

-1表示设置失败； 0表示设置成功。

**调用示例**

```python
import asc.lib.host as host
ascendc_platform = host.get_ascendc_platform()
tiling = host.MatmulApiTiling(ascendc_platform)
tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
# 设置B矩阵，buffer位置为GM，数据格式为ND，数据类型为bfloat16，默认不转置
tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_shape(1024, 1024, 1024)
tiling.set_org_shape(1024, 1024, 1024)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1, -1)   # 设置计算时可用的L1/L0C/UB/BT空间大小
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
