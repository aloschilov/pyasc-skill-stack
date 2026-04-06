# asc.lib.host.MatmulApiTiling.set_sparse

#### MatmulApiTiling.set_sparse(self: libhost.MatmulApiTilingBase, is_sparce_in: bool = False) → int

设置Matmul的使用场景是否为Sparse Matmul场景。

**对应的Ascend C函数原型**

```c++
int32_t SetSparse(bool isSparseIn = false)
```

**参数说明**

- is_sparse_in：设置是否为Sparse Matmul稀疏场景。

**返回值说明**

-1表示设置失败； 0表示设置成功。

**约束说明**

本接口必须在get_tiling接口前调用。

**调用示例**

```python
import asc.lib.host as host
ascendc_platform = host.get_ascendc_platform()
tiling = host.MatmulApiTiling(ascendc_platform)
tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_sparse(True)
tiling.set_shape(1024, 1024, 1024)
tiling.set_org_shape(1024, 1024, 1024)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1)
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
