# asc.lib.host.MatmulApiTiling.set_dequant_type

#### MatmulApiTiling.set_dequant_type(self: libhost.MatmulApiTilingBase, dequant_type: libhost.DequantType) → int

该接口用于设置量化或反量化的模式。

**对应的Ascend C函数原型**

```c++
int32_t SetDequantType(DequantType dequantType)
```

**参数说明**

- dequant_type：设置量化或反量化时的模式。

**返回值说明**

1表示设置失败； 0表示设置成功。

**约束说明**

本接口支持的同一系数的量化/反量化模式、向量的量化/反量化模式分别与Kernel侧接口set_quant_scalar和set_quant_vector对应，本接口设置的量化/反量化模式必须与Kernel侧使用的接口保持一致。

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
tiling.set_dequant_type(host.DequantType.SCALAR)  # 设置同一系数的量化/反量化模式
# tiling.set_dequant_type(host.DequantType.TENSOR)  # 设置向量的量化/反量化模式
tiling.set_buffer_space(-1, -1, -1)
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
