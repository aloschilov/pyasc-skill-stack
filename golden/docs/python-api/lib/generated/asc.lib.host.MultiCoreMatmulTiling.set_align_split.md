# asc.lib.host.MultiCoreMatmulTiling.set_align_split

#### MultiCoreMatmulTiling.set_align_split(self: libhost.MultiCoreMatmulTiling, align_m: SupportsInt, align_n: SupportsInt, align_k: SupportsInt) → int

多核切分时， 设置single_core_m/single_core_n/single_core_k的对齐值。比如设置single_core_m的对齐值为64（单位为元素），切分出的singleCoreM为64的倍数。

**对应的Ascend C函数原型**

```c++
int32_t SetAlignSplit(int32_t alignM, int32_t alignN, int32_t alignK)
```

**参数说明**

- align_m：single_core_m的对齐值。若传入-1或0，表示不设置指定的single_core_m的对齐值，该值由Tiling函数自行计算。
- align_n：single_core_n的对齐值。若传入-1或0，表示不设置指定的single_core_n的对齐值，该值由Tiling函数自行计算。
- align_k：single_core_k的对齐值。若传入-1或0，表示不设置指定的single_core_k的对齐值，该值由Tiling函数自行计算。

**返回值说明**

-1表示设置失败； 0表示设置成功。

**调用示例**

```python
import asc.lib.host as host
ascendc_platform = host.get_ascendc_platform()
tiling = host.MultiCoreMatmulTiling(ascendc_platform)
tiling.set_dim(use_core_nums)
tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_shape(1024, 1024, 1024)
tiling.set_align_split(-1, 64, -1);  # 设置single_core_m/single_core_n/single_core_k的对齐值
tiling.set_org_shape(1024, 1024, 1024)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1)
tiling_data = host.TCubeTiling()
ret1 = tiling.get_tiling(tiling_data)
```
