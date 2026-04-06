# asc.lib.host.MatmulApiTiling.set_org_shape

#### MatmulApiTiling.set_org_shape(\*args, \*\*kwargs)

设置Matmul计算时的原始完整的形状M、N、K或Ka/Kb，单位均为元素个数。

**对应的Ascend C函数原型**

```c++
int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKIn)
int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKaIn, int32_t orgKbIn)
```

**参数说明**
- org_m_in：设置原始完整的形状M大小，单位为元素。
- org_n_in：设置原始完整的形状N大小，单位为元素。
- org_k_in：设置原始完整的形状K大小，单位为元素。原始完整形状Ka=Kb时可设置。
- org_ka_in：设置矩阵A原始完整的形状Ka大小，单位为元素。
- org_kb_in：  设置矩阵B原始完整的形状Kb大小，单位为元素。

**返回值说明**

-1表示设置失败； 0表示设置成功。

**约束说明**

参数org_ka_in和org_kb_in可以不相等，即原始矩阵形状Ka和Kb不相等，并不是实际Matmul计算时的K，此参数只用于辅助Matmul API搬运时的偏移计算。

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
