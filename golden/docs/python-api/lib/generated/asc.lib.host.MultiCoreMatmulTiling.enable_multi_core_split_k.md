# asc.lib.host.MultiCoreMatmulTiling.enable_multi_core_split_k

#### MultiCoreMatmulTiling.enable_multi_core_split_k(self: libhost.MultiCoreMatmulTiling, flag: bool) → None

多核场景，通过该接口使能切K轴。不调用该接口的情况下，默认不切K轴。在GetTiling接口调用前使用。

**对应的Ascend C函数原型**

```c++
void EnableMultiCoreSplitK(bool flag)
```

**参数说明**

- flag：是否使能切K轴。

**约束说明**

- 如果在算子中使用该接口，获取C矩阵结果时仅支持输出到Global Memory。
- 如果在算子中使用该接口，需在Kernel侧代码中首次将C矩阵分片的结果写入Global Memory之前，先清零Global Memory，随后在获取C矩阵分片的结果时，再开启AtomicAdd累加。如果不预先清零Global Memory，可能会因为累加Global Memory中原始的无效数据而产生精度问题。

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
tiling.set_org_shape(1024, 1024, 1024)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1)
tiling.enable_multi_core_split_k(true);  // 使能切K轴
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
