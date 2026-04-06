# asc.lib.host.MatmulApiTiling.set_c_layout

#### MatmulApiTiling.set_c_layout(self: libhost.MatmulApiTilingBase, b: SupportsInt, s: SupportsInt, n: SupportsInt, g: SupportsInt, d: SupportsInt) → int

设置C矩阵的Layout轴信息，包括B、S、N、G、D轴。对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用IterateBatch接口之前，
需要在Host侧Tiling实现中通过本接口设置C矩阵的Layout轴信息。

**对应的Ascend C函数原型**

```c++
int32_t SetCLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d)
```

**参数说明**

- b：C矩阵Layout的B轴信息。
- s：C矩阵Layout的S轴信息。
- n：C矩阵Layout的N轴信息。
- g：C矩阵Layout的G轴信息。
- d：C矩阵Layout的D轴信息。

**返回值说明**

-1表示设置失败； 0表示设置成功。

**约束说明**

对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用iterate_batch接口之前，需要在Host侧Tiling实现中通过本接口设置C矩阵的Layout轴信息。

**调用示例**

```python
import asc.lib.host as host
ascendc_platform = host.get_ascendc_platform()
tiling = host.MultiCoreMatmulTiling(ascendc_platform)
m = 32
n = 256
k = 64
tiling.set_dim(1)
tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_shape(m, n, k)
tiling.set_org_shape(m, n, k)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1)
a_bnum = 2
a_snum = 32
a_gnum = 3
a_dnum = 64
b_bnum = 2
b_snum = 256
b_gnum = 3
b_dnum = 64
c_bnum = 2
c_snum = 32
c_gnum = 3
c_dnum = 256
batch_num = 3
tiling.set_a_layout(a_bnum, a_snum, 1, a_gnum, a_dnum)
tiling.set_b_layout(b_bnum, b_snum, 1, b_gnum, b_dnum)
tiling.set_c_layout(c_bnum, c_snum, 1, c_gnum, c_dnum) # 设置C矩阵排布
tiling.set_batch_num(batch_num)
tiling.set_buffer_space(-1, -1, -1);
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
